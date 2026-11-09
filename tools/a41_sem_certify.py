"""Live-repository A4.1 semantic-head ABI, optimizer and gradient check.

No training steps, checkpoint writes, or COCO evaluation are performed.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from yolox.exp import get_exp


def grad_norm(loss,parameters,retain_graph=True):
    gs=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,
                           allow_unused=True)
    total=sum(g.detach().float().square().sum() for g in gs if g is not None)
    return float(total.sqrt()) if not isinstance(total,int) else 0.0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-exp',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--height',type=int,default=64)
    ap.add_argument('--width',type=int,default=96)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--output',default='audits/a41_sem_certificate.json')
    ap.add_argument('--warn-grad-ratio',type=float,default=0.20)
    a=ap.parse_args()
    if a.height%32 or a.width%32:raise ValueError('Canvas must align to stride32')
    if a.batch_size<1:raise ValueError('Positive batch size required')
    exp=get_exp(a.model_exp,None)
    model=exp.get_model().to(a.device).train()
    if not hasattr(model,'aux_sem_head') or not hasattr(model,'sem_supervision'):
        raise RuntimeError('Experiment did not build auxiliary semantic detector')
    # Optimizer check must happen AFTER exp.get_model() has registered all heads.
    opt=exp.get_optimizer(a.batch_size)
    opt_ids={id(p) for group in opt.param_groups for p in group['params']}
    missing=[n for n,p in model.named_parameters() if p.requires_grad and id(p) not in opt_ids]
    if missing:raise RuntimeError('Trainable parameters absent from optimizer: '+str(missing[:20]))
    x=torch.rand(a.batch_size,3,a.height,a.width,device=a.device)
    t=torch.zeros(a.batch_size,5,5,device=a.device)
    for b in range(a.batch_size):
        t[b,0]=torch.tensor([0,a.width*0.45,a.height*0.5,
                               max(8,a.width*0.20),max(8,a.height*0.25)],device=a.device)
    output=model(x,t)
    for name in ('total_loss','iou_loss','conf_loss','cls_loss','mem_loss','sem_aux_loss'):
        if name not in output or not torch.isfinite(output[name]).all():
            raise RuntimeError('Missing/nonfinite loss '+name)
    sem_params=[p for p in model.aux_sem_head.parameters() if p.requires_grad]
    shared=[p for n,p in model.backbone.named_parameters()
            if p.requires_grad and ('spi_highway.projector' in n or
                                    'C3_p3' in n or 'backbone.dark2' in n)]
    if not shared:raise RuntimeError('Could not locate expected A4.1 shared parameters')
    sem=output['sem_aux_loss']; det=output['total_loss']-sem
    sem_head_norm=grad_norm(sem,sem_params)
    shared_sem=grad_norm(sem,shared)
    shared_det=grad_norm(det,shared)
    ratio=shared_sem/max(shared_det,1e-12)
    if sem_head_norm<=0:raise RuntimeError('Semantic decoder has no auxiliary gradients')
    if shared_sem<=0 and model.backbone_grad_scale>0:
        raise RuntimeError('Semantic objective does not reach shared features')
    if not torch.isfinite(torch.tensor([sem_head_norm,shared_sem,shared_det])).all():
        raise RuntimeError('Nonfinite family gradient')
    output['total_loss'].backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite total-backward gradients')
    model.eval()
    def fail_if_sem_called(*_args,**_kwargs):
        raise RuntimeError('Training-only semantic head ran during inference')
    hook=model.aux_sem_head.register_forward_pre_hook(fail_if_sem_called)
    with torch.no_grad():
        preds=model(x)
    hook.remove()
    if not torch.isfinite(preds).all():raise RuntimeError('Nonfinite detections')
    result={
        'status':'PASS','exp':exp.exp_name,'mode':exp.a41_sem_mode,
        'device':a.device,'inputs':[a.batch_size,3,a.height,a.width],
        'optimizer_missing':missing,'aux_trainable_parameters':sum(p.numel() for p in sem_params),
        'total_loss':float(output['total_loss'].detach()),
        'semantic_weighted':float(sem.detach()),
        'semantic_decoder_grad_l2':sem_head_norm,
        'shared_sem_grad_l2':shared_sem,'shared_det_grad_l2':shared_det,
        'semantic_to_detection_shared_grad_ratio':ratio,
        'grad_ratio_warning':ratio>a.warn_grad_ratio,
        'inference_sem_head_executed':False,
        'predictions_shape':list(preds.shape),
    }
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
    if ratio>a.warn_grad_ratio:
        print('WARNING: semantic gradient exceeds diagnostic budget; do not promote weight')

if __name__=='__main__':main()
