#!/usr/bin/env python3
"""Audited vNext.2 task-alignment probe: source-clean labels only, NO TRAINING.

Runs frozen native vNext.2 backbone+TDE_Head with native SimOTA matching,
replays EXACT native assignments/IoU/soft obj targets for zero-state and TTT
passes, and reports per-image gradient alignment and observed detector effects.

The original model/checkpoint are strictly loaded, then an UNTRAINED separate
SafeSpatialTTT sidecar is instantiated with fixed seed. This is integration
and gradient-connectivity evidence, not a benchmark improvement claim.

PowerShell from YOLOX root:
python tools/audit_safe_ttt_yolox.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py --weights ema `
  --indices 0 10 37 --output audits/safe_ttt_clean_audit.json

Optional --meta-probe checks actual detector->SSL mixed derivative once per
image, which may use substantial memory (run --indices 0 first on your GPU).
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from safe_ttt_model import SafeSpatialTTT, alignment_surrogate
from fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity
from native_dark2_router import Dark2AuditRouter


def gradient_stats(ssl,task):
    a=ssl.detach().float();b=task.detach().float()
    if a.shape!=b.shape or a.ndim!=2:
        raise RuntimeError('Expected independent B x rank gradients')
    numerator=(a*b).sum(1)
    na=a.norm(dim=1);nb=b.norm(dim=1)
    denom=(na*nb).clamp_min(1e-15)
    cosine=torch.where((na>1e-11)&(nb>1e-11),numerator/denom,torch.nan)
    return {'ssl_norm':na.tolist(),'task_norm':nb.tolist(),
            'dot':numerator.tolist(),'cosine':[
                None if not math.isfinite(v) else v for v in cosine.tolist()]}


def loss_view(recon):
    return {'total':recon['total'],**recon['components'],
      **{f'{scale}_{kind}':loss for scale,part in recon['scales'].items() for kind,loss in part.items()},
      **{f'{scale}_total':sum(part.values()) for scale,part in recon['scales'].items()},
      **{f'fpn{level}_{kind}':loss for level,part in recon['levels'].items() for kind,loss in part.items()},
      **{f'fpn{level}_total':sum(part.values()) for level,part in recon['levels'].items()}}


def finite_metrics(report):
    for group in (report['native_losses'],report['updated_losses'],report['zero_state_losses']):
        for value in group.values():
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite task metric detected')


def head_loss_pass(model,images,labels):
    # Set the TDE_Head's *root training boolean* only; child BN, memory and
    # convolution modules remain eval(). Never call model.train() here.
    return model.head(model.backbone(images,ttt_prob=0.0),labels,images)


def audit_one(model,safe,images,labels,*,meta_probe=False,gradient_profile="compact"):
    if model.training or model.backbone.training or not model.head.training:
        raise RuntimeError('Expected model.eval(), backbone.eval(), head.root.training=True')
    if any(m.training for m in model.head.modules() if m is not model.head):
        raise RuntimeError('Head child module in training mode; refusing BN running-stat mutation')
    size=images.shape[0]
    if size!=1:
        raise RuntimeError('Per-image audit currently requires B=1 to keep failure attribution simple')
    # The hook is temporary. No legacy TTT operation can run in this context.
    with Dark2AuditRouter(model,safe) as router, NativeMatchReplay(model.head) as fixed:
        router.set_mode('native')
        with torch.no_grad():
            native_head=head_loss_pass(model,images,labels)
        router.require_one_hit(size)
        fixed.finalize()
        native_feat=router.last_native.detach().float()
        native_logits=fixed.head_inputs['outputs'].detach().clone()
        native_recon=reconstruct_tde_loss(model.head,fixed)
        native_parity=assert_head_loss_parity(native_head,native_recon)
        native_view={k:float(v.detach()) for k,v in loss_view(native_recon).items()}
        matching={'fg':native_recon['counts'],'normalizer':native_recon['num_foreground_normalizer'],
             'matched_images':native_recon['frozen_gt_matches'],
             'soft_objectness_fraction':native_recon['objectness_soft_target_fraction']}

        router.set_mode('fixed_z',z=torch.zeros(size,safe.rank,device=images.device,dtype=torch.float32,requires_grad=True))
        z0=router.z
        fixed.replay()
        with torch.enable_grad():
            z0_head=head_loss_pass(model,images,labels)
        router.require_one_hit(size);fixed.finalize()
        z0_rec=reconstruct_tde_loss(model.head,fixed)
        z0_logits=fixed.head_inputs['outputs'].detach().clone()
        zero_parity=assert_head_loss_parity(z0_head,z0_rec)
        z0_view=loss_view(z0_rec)
        if not torch.allclose(native_head[0].detach(),z0_head[0].detach(),atol=5e-5,rtol=1e-5):
            raise RuntimeError('At z=0 the detector output MUST equal native detector loss')
        mask=safe_ttt_mask(safe,native_feat)
        loss_ssl=safe.ssl_loss(native_feat,z0,mask)
        with torch.enable_grad():
            g_ssl,=torch.autograd.grad(loss_ssl.sum(),z0,retain_graph=True)
        if not torch.isfinite(g_ssl).all():raise FloatingPointError('Nonfinite SSL compact gradient')
        step=safe.compute_step(g_ssl.detach()).detach()
        result_grad={}
        gdet_total=None
        if gradient_profile not in ('compact','full'):
            raise ValueError('gradient_profile must be compact or full')
        compact_names={'total','box','classification','positive_objectness',
                'background_objectness','small_total','medium_total','large_total',
                'fpn8_total','fpn16_total','fpn32_total'}
        for name,objective in z0_view.items():
            if gradient_profile=='compact' and name not in compact_names:
                continue
            if not objective.requires_grad:
                raise RuntimeError(f'{name} is detached from compact state')
            g,=torch.autograd.grad(objective,z0,retain_graph=True)
            stat=gradient_stats(g_ssl,g)
            if name=='total':gdet_total=g.detach().clone()
            stat['taylor_prediction_delta']=float((g.detach()*step).sum())
            result_grad[name]=stat

        router.set_mode('episode',meta_train=False)
        fixed.replay()
        with torch.no_grad():
            after_head=head_loss_pass(model,images,labels)
        router.require_one_hit(size);fixed.finalize()
        after_rec=reconstruct_tde_loss(model.head,fixed)
        after_logits=fixed.head_inputs['outputs'].detach().clone()
        after_parity=assert_head_loss_parity(after_head,after_rec)
        after_view={k:float(v.detach()) for k,v in loss_view(after_rec).items()}
        for k in result_grad:
            result_grad[k]['loss_before']=native_view[k]
            result_grad[k]['loss_after']=after_view[k]
            result_grad[k]['observed_delta']=after_view[k]-native_view[k]
            result_grad[k]['taylor_prediction_error']=result_grad[k]['observed_delta']-result_grad[k]['taylor_prediction_delta']
        prediction_effects={}
        for name,sl in (('bbox',slice(0,4)),('objectness_logit',slice(4,5)),('class_logits',slice(5,None))):
            ref=native_logits[:,:,sl].float(); new=after_logits[:,:,sl].float()
            zed=z0_logits[:,:,sl].float()
            if not torch.allclose(ref,zed,atol=1e-5,rtol=1e-6):
                raise RuntimeError(f'Zero-fast-state predictions mismatch native: {name}')
            diff=new-ref
            prediction_effects[name]={
                'max_absolute_difference':float(diff.abs().max()),
                'mean_absolute_difference':float(diff.abs().mean()),
                'relative_l2':float(diff.square().sum().sqrt()/ref.square().sum().clamp_min(1e-12).sqrt()),
                'fraction_exactly_changed':float((diff!=0).float().mean()),
            }
        report={
            'prediction_effects':prediction_effects,
            'native_losses':native_view,
            'zero_state_losses':{k:float(v.detach()) for k,v in loss_view(z0_rec).items()},
            'updated_losses':after_view,
            'gradients':result_grad,
            'matching':matching,
            'native_parity':native_parity,
            'zero_state_parity':zero_parity,
            'adapted_parity':after_parity,
            'gradient_profile':gradient_profile,
            'fast_state_norm':router.last_stats['fast_state_norm'].cpu().tolist(),
            'step_policy':safe.step_policy,
            'low_signal_abstention':router.last_stats['low_signal_abstention'].cpu().tolist(),
            'ssl_initial':loss_ssl.detach().cpu().tolist(),
            'grad_norm':g_ssl.detach().norm(dim=1).cpu().tolist(),
            'relative_feature_change_postcast':router.last_stats['relative_feature_change_postcast'].cpu().tolist(),
            'postcast_changed_fraction':router.last_stats['postcast_changed_fraction'].cpu().tolist(),
        }
        finite_metrics(report)
        # Optional exact downstream detection->SSL mixed derivative for the real head.
        if meta_probe:
            router.set_mode('episode',meta_train=True)
            fixed.replay()
            with torch.enable_grad():
                metah=head_loss_pass(model,images,labels)
                router.require_one_hit(size);fixed.finalize()
                metar=reconstruct_tde_loss(model.head,fixed)
                assert_head_loss_parity(metah,metar)
                pred_params=list(safe.predictor.parameters())
                mixed=torch.autograd.grad(metar['total'],pred_params,allow_unused=True)
                # The surrogate uses EXACT native fixed-assignment task gradient,
                # not a dynamically rematched target or an adversarial label.
                z_sur=torch.zeros(size,safe.rank,device=images.device,dtype=torch.float32,requires_grad=True)
                ssl_sur=safe.ssl_loss(native_feat,z_sur,mask)
                g_sur,=torch.autograd.grad(ssl_sur.sum(),z_sur,create_graph=True)
                surrogate=alignment_surrogate(g_sur,gdet_total.detach())
                gs=torch.autograd.grad(surrogate,pred_params,allow_unused=True)
            norm=math.sqrt(sum(float(g.detach().square().sum()) for g in mixed if g is not None))
            surrogate_norm=math.sqrt(sum(float(g.detach().square().sum()) for g in gs if g is not None))
            dot=sum(float((x.detach()*y.detach()).sum()) for x,y in zip(mixed,gs) if x is not None and y is not None)
            meta_surrogate_cos=dot/(norm*surrogate_norm) if norm>0 and surrogate_norm>0 else None
            # Fair NEGATIVE CONTROL: same numerical adapted z1 and same detector
            # head, but detach only the SSL-gradient update's dependence on
            # predictor weights, NOT the adapter's outer parameter gradients.
            with torch.enable_grad(),torch.autocast(device_type=images.device.type,enabled=False):
                zc=torch.zeros(size,safe.rank,device=images.device,dtype=torch.float32,requires_grad=True)
                gl,=torch.autograd.grad(safe.ssl_loss(native_feat,zc,mask).sum(),zc,create_graph=False)
                detached_z1=(zc+safe.compute_step(gl.detach())).detach()
            router.set_mode('fixed_z',z=detached_z1)
            fixed.replay()
            with torch.enable_grad():
                control_head=head_loss_pass(model,images,labels)
                router.require_one_hit(size);fixed.finalize()
                control_rec=reconstruct_tde_loss(model.head,fixed)
                assert_head_loss_parity(control_head,control_rec)
                control_grad=torch.autograd.grad(control_rec['total'],pred_params,allow_unused=True)
            if any(g is not None for g in control_grad):
                raise RuntimeError('Detached inner-gradient control unexpectedly reaches SSL predictor')
            if not torch.allclose(control_rec['total'].detach(),metar['total'].detach(),atol=5e-5,rtol=1e-5):
                raise RuntimeError('Meta vs detached control numerical forward mismatch')
            report['meta_probe']={'detector_to_ssl_predictor_grad_norm':norm,
                 'nonzero_predictor_grad_tensors':sum(int(g is not None and bool(g.detach().abs().sum()>0)) for g in mixed),
                 'alignment_surrogate_grad_norm':surrogate_norm,
                 'exact_meta_vs_alignment_surrogate_grad_cosine':meta_surrogate_cos,
                 'surrogate_value':float(surrogate.detach()),
                 'detached_control_predictor_grad_tensors':sum(int(g is not None) for g in control_grad),
                 'detached_control_forward_loss_abs_diff':float((control_rec['total']-metar['total']).detach().abs())}
            if norm==0.0:
                raise RuntimeError('Detection did not deliver mixed gradient into SSL predictor on actual TDE head')
        return report


def safe_ttt_mask(safe,native_feat):
    # Avoid using saliency computed from the target itself to define masks.
    from safe_ttt_model import checkerboard_mask
    return checkerboard_mask(native_feat,phase=0)


def get_clean_train_example(dataset,index,device):
    if index<0 or index>=len(dataset):raise IndexError(f'Invalid clean-train index: {index}')
    example=dataset[index]
    if not isinstance(example,(tuple,list)) or len(example)<2:
        raise RuntimeError('Expected COCODataset item (image, labels, info, id)')
    img,labels=example[:2]
    image=torch.as_tensor(img,device=device).float()
    labels=torch.as_tensor(labels,device=device).float()
    if image.ndim!=3 or image.shape[0]!=3:
        raise RuntimeError('COCO TrainTransform must output CHW RGB/BGR image')
    if labels.ndim!=2 or labels.shape[-1]!=5:
        raise RuntimeError('COCO TrainTransform must provide [max_gt, 5] classes/cxcywh')
    if not torch.isfinite(image).all() or not torch.isfinite(labels).all():
        raise FloatingPointError('Invalid clean training data')
    if (labels[:,3:]<0).any():raise RuntimeError('Negative object size')
    return image.unsqueeze(0),labels.unsqueeze(0)


def setup_model(exp,checkpoint,device,weights,seed):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict):raise RuntimeError('Expected YOLOX checkpoint dictionary')
    name='ema_model' if weights=='ema' else 'model'
    state=ck.get(name)
    if not isinstance(state,dict):raise RuntimeError(f'Missing {name} checkpoint state')
    model=exp.get_model().to(device)
    # Critical: strict model checkpoint load BEFORE creating the separate sidecar.
    clean_state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(clean_state,strict=True)
    model.eval().requires_grad_(False)
    epoch=int(ck.get('start_epoch',1))
    if hasattr(model,'set_meta_training_state'):
        model.set_meta_training_state(max(1,epoch),int(exp.max_epoch))
    else:raise RuntimeError('Expected model.set_meta_training_state')
    if not hasattr(model.head,'get_losses_with_targets'):
        raise RuntimeError('The current head is not compatible with TDE source target semantics')
    model.head.training=True
    # Unlike model.train(), does NOT enable BN/statistics updates.
    torch.manual_seed(seed)
    return model,ck


def get_size_from_native(model,images):
    with Dark2AuditRouter(model,None) as router:
        router.set_mode('native')
        with torch.no_grad():_ = model.backbone(images,ttt_prob=0.0)
        router.require_one_hit(images.shape[0])
        return int(router.last_native.shape[1])


def record_model_hash(model):
    # Very cheap immutable-state checksum across all persisted parameters and buffers.
    # Tensor equality is also verified at each image; hash is for provenance.
    h=hashlib.sha256()
    for key,v in model.state_dict().items():
        h.update(key.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--weights',choices=('ema','raw'),default='ema')
    p.add_argument('--indices',type=int,nargs='+',default=[0])
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--step-policy',choices=('legacy_floor','normalized'),default='normalized')
    p.add_argument('--min-signal-norm',type=float,default=1e-8)
    p.add_argument('--basis-gain',type=float,default=1.0)
    p.add_argument('--meta-probe',action='store_true')
    p.add_argument('--gradient-profile',choices=('compact','full'),default='compact')
    args=p.parse_args()
    torch.manual_seed(args.seed)
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Refusing audit: exp.train_ann is not explicitly a CLEAN source training split')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,ck=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    dataset=exp.get_dataset(cache=False)
    before=record_model_hash(model)
    results={'protocol':{'source':'BDD clean TRAIN only','checkpoint':args.checkpoint,
       'model_exp':args.model_exp,'weights':args.weights,'checkpoint_epoch':ck.get('start_epoch'),
       'seed':args.seed,'rank':args.rank,'step_policy':args.step_policy,
       'min_signal_norm':args.min_signal_norm,'basis_gain':args.basis_gain,
       'meta_probe':args.meta_probe,'gradient_profile':args.gradient_profile,
       'untrained_safe_ttt_stage':True,'strict_native_load':True,
       'match_policy':'frozen_native_assignments_and_all_class_obj_targets',
       'fixed_objectness_negative_group':'background separate',
       'model_sha256':before,'device':str(device)},'images':[]}
    for index in args.indices:
        torch.manual_seed(args.seed+index)
        random.seed(args.seed+index)
        try:
            import numpy as np
            np.random.seed((args.seed+index) % (2**32))
        except ImportError:
            pass
        imgs,labels=get_clean_train_example(dataset,index,device)
        channels=get_size_from_native(model,imgs)
        # A fresh random sidecar for each image, same fixed seed (controls).
        torch.manual_seed(args.seed)
        safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
                     min_signal_norm=args.min_signal_norm,basis_gain=args.basis_gain).to(device=device,dtype=torch.float32).eval()
        data=audit_one(model,safe,imgs,labels,meta_probe=args.meta_probe,gradient_profile=args.gradient_profile)
        data['image_index']=index
        results['images'].append(data)
        print('IMAGE',index,'GT',int((labels.sum(-1)>0).sum()),
             'FG',data['matching']['fg']['foreground'],
             'TASK_BEFORE',round(data['native_losses']['total'],6),
             'AFTER',round(data['updated_losses']['total'],6),
             'TASK_DELTA',round(data['updated_losses']['total']-data['native_losses']['total'],8),
             'COS',data['gradients']['total']['cosine'],
             'MIXED_META',data.get('meta_probe',{}).get('detector_to_ssl_predictor_grad_norm','not run'))
        del safe
    after=record_model_hash(model)
    if before!=after:raise RuntimeError('Native model persistent weights/buffers changed during audit')
    results['protocol']['model_sha256_after']=after
    results['model_state_mutations']=[]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print('SAVED',out)
    print('IMPORTANT: sidecar is random/untrained; these are mechanical & directional source-label diagnostics, not domain-generalization results')

if __name__=='__main__':main()
