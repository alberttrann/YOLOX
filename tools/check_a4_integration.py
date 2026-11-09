"""Local CUDA/CPU preflight of A4 construction, losses, and gradient isolation."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from yolox.exp import get_exp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s_fourier_a4.py')
    p.add_argument('--device',default='cuda')
    p.add_argument('--height',type=int,default=64)
    p.add_argument('--width',type=int,default=96)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--backward',action='store_true')
    a=p.parse_args()
    if a.height%32 or a.width%32:raise ValueError('Input must be stride-32 aligned')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    e=get_exp(a.model_exp,None);m=e.get_model().to(a.device)
    print('A4_PARAMS',sum(p.numel() for p in m.parameters()),'VARIANT',e.fourier_variant)
    x=torch.randn(a.batch_size,3,a.height,a.width,device=a.device)
    m.eval()
    with torch.no_grad():pred=m(x)
    if pred.shape[-1]!=e.num_classes+5 or not torch.isfinite(pred).all():
        raise RuntimeError('Invalid inference shape or nonfinite output')
    print('A4_INFERENCE',list(pred.shape))
    if a.backward:
        m.train()
        targets=torch.zeros(a.batch_size,8,5,device=a.device)
        for b in range(a.batch_size):
            targets[b,0]=torch.tensor([float(b%e.num_classes),a.width/2,a.height/2,
                                       min(16,a.width/2),min(16,a.height/2)],device=a.device)
        out=m(x,targets)
        if not torch.isfinite(out['total_loss']):raise RuntimeError('Nonfinite detection loss')
        backbone=next(p for n,p in m.named_parameters() if 'backbone.backbone.dark3' in n)
        proto=next(p for n,p in m.named_parameters() if 'memory_banks' in n and 'prototypes' in n)
        g=torch.autograd.grad(out['mem_loss'],[backbone,proto],retain_graph=True,allow_unused=True)
        if g[0] is not None and torch.count_nonzero(g[0]):
            raise RuntimeError('Memory auxiliary gradient leaked into backbone')
        m.zero_grad(set_to_none=True)
        out['total_loss'].backward()
        active=[(n,p) for n,p in m.named_parameters() if 'spectral_adapters' in n and p.grad is not None and p.grad.abs().sum()>0]
        if not active:raise RuntimeError('Fourier adapter has zero gradient from detection loss')
        print('A4_LOSSES',{k:float(out[k].detach()) for k in ('total_loss','det_loss','mem_loss','iou_loss','cls_loss','conf_loss')})
        print('A4_GRAD_ISOLATION_PASS','spectral_nonzero_parameters',len(active))
    print('A4_INTEGRATION_PASS')


if __name__=='__main__':main()
