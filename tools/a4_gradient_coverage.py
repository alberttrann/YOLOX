"""P1: real-image classification/regression gradient conflicts + GT coverage.

Uses exact head loss tensors; uses a subset of shared parameters to bound cost.
No optimizer steps are performed. Non-finite losses and missing gradients fail.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from yolox.exp import get_exp


def audit(model,images,targets,include=('backbone.backbone.dark3','backbone.C3_p3','backbone.C3_p4')):
    model.train()
    result=model(images,targets)
    names=[]
    all_params=list(model.named_parameters())
    for prefix in include:
        matches=[(n,p) for n,p in all_params if p.requires_grad and prefix in n]
        names.extend(matches[:6])
    if not names:raise RuntimeError('No shared parameters matched requested audit prefixes')
    params=[p for _,p in names]
    keys=['iou_loss','conf_loss','cls_loss','mem_loss','l1_loss']
    gradients={}
    norms={}
    for key in keys:
        loss=result[key]
        if not isinstance(loss,torch.Tensor) or not loss.requires_grad:
            norms[key]=0.;continue
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        vec=torch.cat([(g.float().reshape(-1) if g is not None else torch.zeros_like(p).float().reshape(-1))
                       for p,g in zip(params,grads)]).detach()
        gradients[key]=vec
        norms[key]=float(torch.linalg.vector_norm(vec))
    cos={}
    for ia,k in enumerate(keys):
        for q in keys[ia+1:]:
            if k in gradients and q in gradients and norms[k]>0 and norms[q]>0:
                cos[f'{k}__{q}']=float(torch.dot(gradients[k],gradients[q]) /
                    (norms[k]*norms[q]))
    return {'losses':{k:float(result[k].detach()) for k in keys},
            'grad_norms':norms,'grad_cosines':cos,
            'sampled_parameters':[n for n,_ in names],
            'assignment':dict(model.head.last_assignment_stats),
            'coverage':dict(getattr(model.head,'last_coverage_by_size',{})),
            'total_loss':float(result['total_loss'].detach())}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s_fourier_a4.py')
    p.add_argument('--checkpoint',required=False)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--batches',type=int,default=4)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    if a.batches<1:raise ValueError('batches must be positive')
    device=torch.device(a.device)
    exp=get_exp(a.model_exp,None)
    model=exp.get_model().to(device)
    if a.checkpoint and getattr(exp,'a4_filter_basis',None) is not None:
        from a4_verify_checkpoint import verify
        verify(a.model_exp,a.checkpoint)
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if not isinstance(ck,dict) or not isinstance(ck.get('model'),dict):
            raise RuntimeError('Need raw model checkpoint for gradient diagnostic')
        model.load_state_dict(ck['model'],strict=True)
    loader=exp.get_data_loader(batch_size=a.batch_size,is_distributed=False,
                               no_aug=True,cache_img=None)
    records=[]
    for i,batch in enumerate(loader):
        if i>=a.batches:break
        if len(batch)<2:raise RuntimeError('Data loader did not return images and targets')
        images,targets=batch[:2]
        info=audit(model,images.to(device),targets.to(device))
        records.append(info)
        print('GRADIENT',i,info['grad_norms'],'COVERAGE',info['coverage'],flush=True)
    if not records:raise RuntimeError('No batches audited')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps({'model_exp':a.model_exp,'records':records},indent=2),encoding='utf-8')


if __name__=='__main__':main()
