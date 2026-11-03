#!/usr/bin/env python3
"""Read-only masked-vs-visible reconstruction gradient decomposition for vNext.2.

Never alters a checkpoint or any model parameter. Evaluates the saved EMA/raw
projector and frozen native dark2 features on real validation images.

 python tools/diagnose_vnext2_ttt_objective.py `
   --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
   --model-exp exps/default/tde_yolox_s.py `
   --data-exp exps/default/test_acdc_fog.py `
   --indices 0 10 37 `
   --output YOLOX_outputs/vnext2_ttt_mask_decomposition_fog.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def grad_magnitude(grads):
    return float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)))


def masked_spatial_variance(feature, mask):
    """Variance across masked spatial sites per channel, then mean channels."""
    out = []
    for idx in range(feature.shape[0]):
        f = feature[idx].detach().float().flatten(1)
        sel = mask[idx, 0].detach().flatten() > 0.5
        if sel.sum().item() >= 2:
            out.append(float(f[:, sel].var(dim=1, unbiased=False).mean()))
    return sum(out) / len(out) if out else None


def probe_inner_objective(dark2, native_features):
    """Decompose exact vNext.2 evaluation inner MSE by spatial region.

    For sample B=1, the region gradients add up to the full gradient. Loss
    contribution uses total tensor-element denominator (not region mean), so
    masked+unmasked exactly reconstructs the old inner training objective.
    """
    if native_features.shape[0] != 1:
        raise ValueError('Only B=1 is supported to avoid shared-batch confounding')
    if torch.is_inference_mode_enabled():
        raise RuntimeError('TTT gradient probe cannot run within inference_mode')
    with torch.enable_grad(), torch.autocast(device_type=native_features.device.type, enabled=False):
        source = native_features.detach().float()
        with torch.no_grad():
            target = dark2._get_noise_clipped_target(source)
            mask = dark2._get_relative_contrast_mask(target, dark2.mask_ratio).float()
            norm = F.group_norm(source, 8, eps=1e-5)
        s = dark2.residual_scale.detach().float().clone().requires_grad_(True)
        b = dark2.residual_bias.detach().float().clone().requires_grad_(True)
        residual = dark2.residual_gain * (s.tanh() * norm + b.tanh())
        corrupted = (source + residual) * (1.0-mask) + dark2.projector.mask_token.detach().float()*mask
        predicted = dark2.projector(corrupted).float()
        squared = (predicted-target).square()
        N = squared.numel()
        # Use a common denominator, so full = masked + visible.
        masked = (squared * mask).sum() / N
        visible = (squared * (1.0-mask)).sum() / N
        full = masked + visible
        g_masked = torch.autograd.grad(masked, (s,b), retain_graph=True)
        g_visible = torch.autograd.grad(visible, (s,b), retain_graph=True)
        g_total = torch.autograd.grad(full, (s,b))
        sum_grad_error = float(torch.sqrt(sum((x+y-z).square().sum()
                              for x,y,z in zip(g_masked,g_visible,g_total))))
        gmasked = grad_magnitude(g_masked)
        gvisible = grad_magnitude(g_visible)
        gfull = grad_magnitude(g_total)
        cross = sum((x.detach()*y.detach()).sum() for x,y in zip(g_masked,g_visible))
        cosine = float(cross/(gmasked*gvisible)) if gmasked and gvisible else None
        frac = float(mask.mean())
        masked_recon = float(masked.detach()) / max(frac,1e-12)
        visible_recon = float(visible.detach()) / max(1.-frac,1e-12)
        report = {
            'masked_fraction': frac,
            'loss_total': float(full.detach()),
            'loss_masked_contribution': float(masked.detach()),
            'loss_visible_contribution': float(visible.detach()),
            'masked_region_mse': masked_recon,
            'visible_region_mse': visible_recon,
            'masked_grad_norm': gmasked,
            'visible_grad_norm': gvisible,
            'full_grad_norm': gfull,
            'grad_masked_to_visible_norm_ratio': gmasked/max(gvisible, 1e-15),
            'gradient_cosine_between_regions': cosine,
            'gradient_additivity_absolute_error': sum_grad_error,
            'masked_output_spatial_variance': masked_spatial_variance(predicted,mask),
            'masked_target_spatial_variance': masked_spatial_variance(target,mask),
            'interpretation': ('Spatially identical masked-token inputs lack local-neighbor '
                'information in the 1x1 projector; masked-region gradients may still '
                'arrive indirectly through GroupNorm/GRN global statistics. '
                'This read-only probe does not imply TTT improves detection.'),
        }
        if not all(torch.isfinite(g).all().item() for g in (*g_masked,*g_visible,*g_total)):
            raise FloatingPointError('Nonfinite TTT region gradient')
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--data-exp',required=True)
    p.add_argument('--weights',choices=['ema','raw'],default='ema')
    p.add_argument('--indices',type=int,nargs='+',default=[0,10,37])
    p.add_argument('--output',required=True)
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Requires existing vNext.2 v2.1 diagnostic helpers, not vNext.3 source.
    from diagnose_vnext2_v2 import get_image
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    data_exp=get_exp(args.data_exp,None)
    for key in ('data_dir','val_ann','test_ann','test_size'):
        setattr(exp,key,getattr(data_exp,key))
    ck=torch.load(args.checkpoint, map_location='cpu',weights_only=False)
    if not isinstance(ck,dict):
        raise RuntimeError('Expected vNext.2 checkpoint dictionary')
    state=ck.get('ema_model' if args.weights=='ema' else 'model')
    if not isinstance(state,dict):
        raise RuntimeError(f'Missing checkpoint {args.weights} state')
    model=exp.get_model().to(device).eval().requires_grad_(False)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    if hasattr(model,'set_meta_training_state'):
        model.set_meta_training_state(max(1,int(ck.get('start_epoch',1))),int(exp.max_epoch))
    model.eval_ttt_enabled=False
    dark2=model.backbone.backbone.dark2
    if not dark2.residual_adapter:
        raise RuntimeError('Expected residual TTT vNext.2 dark2')
    data=exp.get_eval_dataset(legacy=False)
    out={'checkpoint':args.checkpoint,'weights':args.weights,'checkpoint_start_epoch':ck.get('start_epoch'),
         'model_exp':args.model_exp,'data_exp':args.data_exp,'images':[]}
    current={}
    hook=dark2.backbone_stage.register_forward_hook(lambda _,__,native: current.update(native=native.detach()))
    before={k:v.clone() for k,v in model.state_dict().items()}
    try:
        for index in args.indices:
            if index<0 or index>=len(data):
                raise IndexError(f'image index {index} out of 0..{len(data)-1}')
            image=get_image(data,index,device)
            with torch.no_grad():
                _=model(image)
            item=probe_inner_objective(dark2,current['native'])
            item['image_index']=index
            out['images'].append(item)
    finally:
        hook.remove()
    changes=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
    if changes:
        raise RuntimeError(f'Unexpected persistent EMA state mutations: {changes}')
    out['state_mutations']=changes
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    for item in out['images']:
        print(f"IMAGE={item['image_index']} MASK={item['masked_fraction']:.4f} "
              f"LOSS_MASKED={item['loss_masked_contribution']:.6g} "
              f"LOSS_VISIBLE={item['loss_visible_contribution']:.6g} "
              f"GRAD_MASKED={item['masked_grad_norm']:.6g} "
              f"GRAD_VISIBLE={item['visible_grad_norm']:.6g} "
              f"GRAD_COS={item['gradient_cosine_between_regions']}")
    print('SAVED',output)


if __name__=='__main__':
    main()
