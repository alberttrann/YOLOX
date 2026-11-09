"""Pretraining CUDA/CPU integration gate. Uses actual installed YOLOX runtime.

Examples:
 python tools/check_fourier_integration.py --variant spectral --device cpu
 python tools/check_fourier_integration.py --variant spectral_memory --device cuda --backward
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from yolox.exp import get_exp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['off','spatial','spectral','spectral_memory'],default='spectral')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--height',type=int,default=64)
    p.add_argument('--width',type=int,default=96)
    p.add_argument('--backward',action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--minimum-load-ratio',type=float,default=None,
                   help='Optional explicit checkpoint validation threshold')
    a=p.parse_args()
    exp=get_exp(f'exps/default/tde_yolox_s_fourier_{a.variant}.py',None)
    model=exp.get_model().to(a.device)
    print('EXP',exp.exp_name, 'VARIANT',a.variant)
    print('GEOMETRY',json.dumps(exp._preflight(),sort_keys=True))
    nparams=sum(p.numel() for p in model.parameters())
    active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('PARAMETERS total=',nparams,'trainable=',active)
    assert not any('ttt_lrs' in k or 'taxonomic_margins' in k for k in model.state_dict())
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        state=ck.get('ema_model') or ck.get('model') or ck if isinstance(ck,dict) else ck
        state={k.removeprefix('module.'):v for k,v in state.items()}
        model_state=model.state_dict()
        # Same BN-inclusive dark2 remapping as FourierTrainer.resume_train.
        selected={}
        prefix='backbone.backbone.dark2.'
        for k,v in state.items():
            possible=[k]
            if k.startswith(prefix) and '.backbone_stage.' not in k:
                possible.append(k.replace(prefix,prefix+'backbone_stage.',1))
            for new_key in possible:
                if new_key in model_state and model_state[new_key].shape==v.shape:
                    selected[new_key]=v
        total=sum(v.numel() for v in model_state.values() if v.is_floating_point())
        loaded=sum(model_state[k].numel() for k in selected if model_state[k].is_floating_point())
        ratio=loaded/max(total,1)
        print('CHECKPOINT matched_keys',len(selected),'float_element_coverage',round(ratio,6))
        if a.minimum_load_ratio is not None and ratio<a.minimum_load_ratio:
            raise RuntimeError('Checkpoint coverage below explicitly requested threshold')
        model.load_state_dict({**model_state,**selected},strict=True)
        print('NOTE: compatible parameter warm-start only; do not resume optimizer state')
    x=torch.randn(a.batch_size,3,a.height,a.width,device=a.device)
    model.eval()
    with torch.no_grad():
        out=model(x)
    print('INFERENCE',tuple(out.shape),'finite',bool(torch.isfinite(out).all()))
    assert torch.isfinite(out).all()
    if a.backward:
        model.train()
        labels=torch.zeros(a.batch_size,8,5,device=a.device)
        labels[0,0]=torch.tensor([0.,a.width/2,a.height/2,16.,16.],device=a.device)
        loss=model(x,labels)['total_loss']
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for n,p in model.named_parameters() if 'spectral_adapters' in n and p.grad is not None]
        if a.variant!='off':
            assert grads and any(g.abs().sum()>0 for g in grads)
        print('BACKWARD loss',float(loss.detach()))
    print('PASS FULL INTEGRATION')

if __name__=='__main__':main()
