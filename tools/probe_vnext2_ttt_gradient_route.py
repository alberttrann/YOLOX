#!/usr/bin/env python3
"""vNext.2 TTT spatial-gradient ROUTING ABLATION, strictly read only.

This probes how a parameter-free neighbor-fill changes the masked-target
fast-weight gradient. It does NOT install or use a new TTT training objective.
No checkpoint/weights are modified or exported.

Run within YOLOX repo, with previous diagnose_vnext2_v2.py available:
python tools/probe_vnext2_ttt_gradient_route.py \
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth \
  --model-exp exps/default/tde_yolox_s.py \
  --data-exp exps/default/test_acdc_fog.py --indices 0 10 37 \
  --output YOLOX_outputs/vnext2_ttt_gradient_route_fog.json
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _gnorm(grads):
    return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads))


def _cosine(a, b):
    dot=sum(float((x.detach()*y.detach()).sum()) for x,y in zip(a,b))
    na, nb=_gnorm(a),_gnorm(b)
    return dot / (na*nb) if na > 0 and nb > 0 else None


def _local_visible_mean(restored, mask):
    """Nonlearned 3x3 masked-neighbor mean, with per-location valid counts.

    Dividing identically padded local sums/counts avoids edge bias.
    Each masked location only borrows visible neighbor features; it does NOT
    see its target feature and is never given ground-truth during inference.
    """
    visible=1.-mask
    numerator=F.avg_pool2d(restored*visible, 3, stride=1, padding=1)
    denominator=F.avg_pool2d(visible,3,stride=1,padding=1)
    contextual=numerator / denominator.clamp_min(1e-6)
    return contextual,denominator


def probe_gradient_route(dark2, native_features):
    if native_features.shape[0]!=1:
        raise ValueError('B=1 required; each image must have independent fast weights')
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Cannot differentiate inside torch.inference_mode')
    with torch.enable_grad(),torch.autocast(device_type=native_features.device.type,enabled=False):
        source=native_features.detach().float()
        with torch.no_grad():
            target=dark2._get_noise_clipped_target(source)
            mask=dark2._get_relative_contrast_mask(target,dark2.mask_ratio).float()
            normalized=F.group_norm(source,8,eps=1e-5)
        s=dark2.residual_scale.detach().float().clone().requires_grad_(True)
        b=dark2.residual_bias.detach().float().clone().requires_grad_(True)
        restored=source+dark2.residual_gain*(s.tanh()*normalized+b.tanh())
        token=dark2.projector.mask_token.detach().float()
        available=1.-mask
        contextual,counts=_local_visible_mean(restored,mask)
        has_neighbor=(counts>1e-6).to(mask.dtype)
        context_fill=contextual*has_neighbor + token*(1.-has_neighbor)
        # Same forward values, gradients cut ONLY through masked-site context.
        inputs={
            'original_token': restored*available + token*mask,
            'neighbor_fill_detached_control': restored*available + context_fill.detach()*mask,
            'neighbor_fill_differentiable': restored*available + context_fill*mask,
        }
        outputs={}
        gradients={}
        N=int(source.numel())
        for name, x in inputs.items():
            prediction=dark2.projector(x).float()
            error=(prediction-target).square()
            masked=(error*mask).sum()/N
            visible=(error*available).sum()/N
            full=masked+visible
            gm=torch.autograd.grad(masked,(s,b),retain_graph=True)
            gv=torch.autograd.grad(visible,(s,b),retain_graph=True)
            gt=torch.autograd.grad(full,(s,b),retain_graph=True)
            gradnorm_m,gradnorm_v=_gnorm(gm),_gnorm(gv)
            if not all(torch.isfinite(g).all().item() for g in (*gm,*gv,*gt)):
                raise FloatingPointError(f'Nonfinite gradients in {name}')
            gradients[name]=(gm,gv,gt)
            selected=prediction[0].flatten(1)[:,mask[0,0].bool().flatten()]
            variance=float(selected.var(dim=-1,unbiased=False).mean().detach()) if selected.shape[1]>1 else None
            outputs[name]={
                'full_loss':float(full.detach()),
                'masked_loss_contribution':float(masked.detach()),
                'visible_loss_contribution':float(visible.detach()),
                'masked_region_mse':float(masked.detach())/max(float(mask.mean()),1e-12),
                'visible_region_mse':float(visible.detach())/max(float(available.mean()),1e-12),
                'masked_gradient_norm':gradnorm_m,
                'visible_gradient_norm':gradnorm_v,
                'full_gradient_norm':_gnorm(gt),
                'masked_to_visible_gradient_ratio':gradnorm_m/max(gradnorm_v,1e-15),
                'gradient_cosine':_cosine(gm,gv),
                'masked_prediction_spatial_variance':variance,
                'gradient_additivity_error':_gnorm(tuple(x+y-z for x,y,z in zip(gm,gv,gt))),
            }
        ctx_g=gradients['neighbor_fill_differentiable'][0]
        detached_g=gradients['neighbor_fill_detached_control'][0]
        difference=tuple(a-b for a,b in zip(ctx_g,detached_g))
        outputs['gradient_path_isolation']={
            'direct_neighbor_route_gradient_norm':_gnorm(difference),
            'masked_gradient_with_route_norm':_gnorm(ctx_g),
            'direct_route_fraction_of_contextual_masked_grad':_gnorm(difference)/max(_gnorm(ctx_g),1e-15),
            'forward_equivalence_between_context_and_detached':float((inputs['neighbor_fill_differentiable']-inputs['neighbor_fill_detached_control']).abs().max().detach()),
        }
        outputs['mask_and_context']={
            'mask_fraction':float(mask.mean().detach()),
            'fraction_masked_sites_with_visible_neighbor':float((has_neighbor*mask).sum().detach()/mask.sum().clamp_min(1)),
            'model_type':'Original vNext.2 saved projector, unchanged',
            'neighbor_radius_cells':1,
            'context_kernel':3,
        }
        return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    parser.add_argument('--data-exp',required=True)
    parser.add_argument('--weights',choices=['ema','raw'],default='ema')
    parser.add_argument('--indices',type=int,nargs='+',default=[0,10,37])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from diagnose_vnext2_v2 import get_image
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    data_exp=get_exp(args.data_exp,None)
    for key in ('data_dir','val_ann','test_ann','test_size'):
        setattr(exp,key,getattr(data_exp,key))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint.get('ema_model' if args.weights=='ema' else 'model')
    if not isinstance(state,dict):
        raise RuntimeError(f'Missing checkpoint state: {args.weights}')
    model=exp.get_model().to(device).eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    if hasattr(model,'set_meta_training_state'):
        model.set_meta_training_state(max(1,int(checkpoint.get('start_epoch',1))),int(exp.max_epoch))
    model.eval_ttt_enabled=False
    dark2=model.backbone.backbone.dark2
    if not getattr(dark2,'residual_adapter',False):
        raise RuntimeError('Expected residual adapter on dark2')
    dataset=exp.get_eval_dataset(legacy=False)
    saved={key:value.clone() for key,value in model.state_dict().items()}
    native={}
    hook=dark2.backbone_stage.register_forward_hook(lambda _m,_a,output:native.update(feature=output.detach()))
    report={'protocol':{'checkpoint':args.checkpoint,'weights':args.weights,'start_epoch':checkpoint.get('start_epoch'),'model_exp':args.model_exp,'data_exp':args.data_exp,'read_only':True,'architectural_change':False,'trained_objective_change':False},'images':[]}
    try:
        for index in args.indices:
            if not 0<=index<len(dataset):
                raise IndexError(f'index {index} outside dataset length {len(dataset)}')
            inp=get_image(dataset,index,device)
            with torch.no_grad():
                model(inp)
            r=probe_gradient_route(dark2,native['feature'])
            r['image_index']=index
            report['images'].append(r)
            b=r['original_token'];c=r['neighbor_fill_differentiable'];z=r['gradient_path_isolation'];ctx=r['mask_and_context']
            print(f'IMAGE={index} COVERAGE={ctx["fraction_masked_sites_with_visible_neighbor"]:.3f} '
                  f'BASE_MASK_GRAD={b["masked_gradient_norm"]:.4e} '
                  f'CONTEXT_MASK_GRAD={c["masked_gradient_norm"]:.4e} '
                  f'CONTEXT_TOTAL_LOSS={c["full_loss"]:.6g} '
                  f'DIRECT_ROUTE_GRAD={z["direct_neighbor_route_gradient_norm"]:.4e} '
                  f'CONTEXT_MASK_MSE={c["masked_region_mse"]:.6g}')
    finally:
        hook.remove()
    changed=[key for key,val in model.state_dict().items() if not torch.equal(val,saved[key])]
    if changed:
        raise RuntimeError('Model state changed: '+str(changed))
    report['state_mutations']=changed
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print('SAVED',path)
    print('CAUTION: diagnostic neighbor-fill is NOT a trained model or validated detection improvement.')


if __name__=='__main__':
    main()
