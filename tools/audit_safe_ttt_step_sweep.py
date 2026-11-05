#!/usr/bin/env python3
"""READ-ONLY, SOURCE-CLEAN finite-step TTT direction/amplitude audit.

Requires existing YOLOX sidecar tools in tools/:
  audit_safe_ttt_yolox.py, fixed_detection_targets.py, native_dark2_router.py,
  safe_ttt_model.py, and pilot_safe_ttt_meta.py.

Runs a frozen native checkpoint and, optionally, a source-only-trained sidecar
on DISJOINT clean TRAIN examples.  Captures one native SimOTA assignment and
replays exactly the same target for every candidate fast-state strength.

The GT labels are used solely to evaluate candidate directions OFFLINE. The
oracle best scale MUST NOT be used at deployment or tuned on ACDC data.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch

from audit_safe_ttt_yolox import (setup_model, get_size_from_native,
    head_loss_pass, record_model_hash)
from fixed_detection_targets import (NativeMatchReplay,reconstruct_tde_loss,
    assert_head_loss_parity)
from native_dark2_router import Dark2AuditRouter
from safe_ttt_model import SafeSpatialTTT
from pilot_safe_ttt_meta import load_source_example

COMPONENTS=('box','classification','positive_objectness','background_objectness')
SCALES=('small','medium','large')


def section_view(result):
    """Python floats and fixed native-match counts; never track train graphs."""
    return {
        'total': float(result['total'].detach()),
        'components': {k:float(v.detach()) for k,v in result['components'].items()},
        'scales': {k:{j:float(x.detach()) for j,x in v.items()}
                   for k,v in result['scales'].items()},
        'levels': {str(k):{j:float(x.detach()) for j,x in v.items()}
                   for k,v in result['levels'].items()},
        'counts': dict(result['counts']),
    }


def component_deltas(a,b):
    return {k: b['components'][k]-a['components'][k] for k in COMPONENTS}


def evaluate_acceptance(native,updated,eps_abs,eps_rel):
    """Descriptive fixed-target test, NOT a guaranteed AP/AR safety gate."""
    base=native['total'];delta=updated['total']-base
    threshold=max(eps_abs,abs(base)*eps_rel)
    present={}
    for size in SCALES:
        if native['counts'].get(size,0)==0:
            continue
        x=sum(native['scales'][size].values())
        y=sum(updated['scales'][size].values())
        tolerance=max(eps_abs,abs(x)*eps_rel)
        present[size]={'delta':y-x,'within_tolerance': y-x <= tolerance,
                       'positive_anchor_count':native['counts'][size]}
    bgdelta=updated['components']['background_objectness']-native['components']['background_objectness']
    bgthreshold=max(eps_abs,abs(native['components']['background_objectness'])*eps_rel)
    return {
        'total_delta':delta,
        'total_tolerance':threshold,
        'meaningfully_beneficial':delta < -threshold,
        'meaningfully_harmful':delta > threshold,
        'scale_nonregressing':all(x['within_tolerance'] for x in present.values()),
        'background_nonregressing':bgdelta<=bgthreshold,
        'present_scale_groups':present,
        'background_delta':bgdelta,
        'strict_surrogate_pass':(delta < -threshold and
                all(x['within_tolerance'] for x in present.values()) and
                bgdelta<=bgthreshold),
    }


def summarize(images, scales):
    aggregate={}
    for scale in scales:
        records=[a['candidates'][str(scale)] for a in images]
        # For each scale, ignore None from images without relevant groups.
        deltas=[r['total_delta'] for r in records]
        ns=len(deltas)
        aggregate[str(scale)]={
           'images':ns,
           'mean_total_delta':sum(deltas)/max(ns,1),
           'median_total_delta':sorted(deltas)[ns//2] if ns%2 else
               0.5*(sorted(deltas)[ns//2-1]+sorted(deltas)[ns//2]),
           'n_beneficial':sum(bool(r['meaningfully_beneficial']) for r in records),
           'n_harmful':sum(bool(r['meaningfully_harmful']) for r in records),
           'n_scale_nonregressing':sum(bool(r['scale_nonregressing']) for r in records),
           'n_strict_surrogate_pass':sum(bool(r['strict_surrogate_pass']) for r in records),
           'mean_changed_feature_relative_l2':sum(a['feature_effects'][str(scale)] for a in images)/max(ns,1),
        }
    aggregate['oracle_diagnostic_only']={
      'n_images_with_any_beneficial_nonzero_step':sum(any(
          x['candidates'][str(s)]['meaningfully_beneficial'] for s in scales if s>0) for x in images),
      'n_images_with_any_strict_surrogate_step':sum(any(
          x['candidates'][str(s)]['strict_surrogate_pass'] for s in scales if s>0) for x in images),
    }
    return aggregate


def eval_one(model,safe,dataset,index,device,seed,scales,eps_abs,eps_rel):
    images,labels=load_source_example(dataset,index,device,seed)
    if images.shape[0]!=1:raise RuntimeError('Expected batch=1')
    with Dark2AuditRouter(model,safe) as router, NativeMatchReplay(model.head) as matcher:
        router.set_mode('native')
        with torch.no_grad():
            direct=head_loss_pass(model,images,labels)
        router.require_one_hit(1);matcher.finalize()
        original=reconstruct_tde_loss(model.head,matcher)
        assert_head_loss_parity(direct,original)
        native=section_view(original)
        feat=router.last_native.detach()
        if feat.shape[1]!=safe.channels:
            raise RuntimeError('Dark2 channels do not match sidecar')
        with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
            z0=torch.zeros(1,safe.rank,device=device,dtype=torch.float32,requires_grad=True)
            # Same unlabeled SSL gradient that inference uses.
            ssl=safe.ssl_loss(feat.detach().float(),z0)
            grad,=torch.autograd.grad(ssl.sum(),z0)
            if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite inner SSL gradient')
            base_step=safe.compute_step(grad).detach()
        # Exact source-label detector gradient at native z=0 with frozen
        # native assignments. This is an OFFLINE diagnostic only.
        router.set_mode('fixed_z', z=z0)
        matcher.replay()
        with torch.enable_grad():
            z_head=head_loss_pass(model,images,labels)
        router.require_one_hit(1);matcher.finalize()
        z_recon=reconstruct_tde_loss(model.head,matcher)
        assert_head_loss_parity(z_head,z_recon)
        if not torch.allclose(z_recon['total'].detach(),original['total'].detach(),atol=5e-5,rtol=1e-5):
            raise RuntimeError('Zero fast state does not reproduce native loss')
        det_grad,=torch.autograd.grad(z_recon['total'],z0)
        if not torch.isfinite(det_grad).all():raise FloatingPointError('Nonfinite detector gradient')
        norm_ssl=float(grad.norm().detach()); norm_det=float(det_grad.norm().detach())
        dot=float((grad.detach()*det_grad.detach()).sum())
        cosine=dot/(norm_ssl*norm_det) if norm_ssl>1e-12 and norm_det>1e-12 else None
        saved_feature=feat.detach().float()
        details={'index':index, 'native':native,
          'ssl_loss':float(ssl.sum().detach()),
          'ssl_grad_norm':norm_ssl,
          'detector_grad_norm':norm_det,
          'ssl_detector_grad_dot':dot,
          'ssl_detector_grad_cosine':cosine,
          'base_step_norm':float(base_step.norm().detach()),
          'low_signal_abstention':bool((base_step.abs().max()==0).item()),
          'candidates':{},'feature_effects':{},'total_gt':int((labels.sum(-1)>0).sum()),
        }
        for scalar in scales:
            key=str(scalar)
            z=base_step*scalar
            router.set_mode('fixed_z',z=z)
            matcher.replay()
            with torch.no_grad():
                candidate=head_loss_pass(model,images,labels)
            router.require_one_hit(1);matcher.finalize()
            replay=reconstruct_tde_loss(model.head,matcher)
            assert_head_loss_parity(candidate,replay)
            view=section_view(replay)
            acc=evaluate_acceptance(native,view,eps_abs,eps_rel)
            acc['component_deltas']=component_deltas(native,view)
            acc['taylor_predicted_delta']=float((det_grad.detach()*z).sum())
            acc['taylor_error']=acc['total_delta']-acc['taylor_predicted_delta']
            acc['fpn_level_total_deltas']={
                key2:sum(view['levels'][key2].values())-sum(val.values())
                for key2,val in native['levels'].items()}
            # The correction magnitude after final native dtype must be checked;
            # FP16 can quantize away tiny changes.
            with torch.no_grad():
                adapted=safe.adapter(feat.float(),z).to(feat.dtype).float()
                d=adapted-saved_feature
                change=float(d.square().mean().sqrt() /
                      saved_feature.square().mean().clamp_min(1e-8).sqrt())
            details['candidates'][key]=acc
            details['feature_effects'][key]=change
        # Keep an oracle upper bound only as a diagnosis; never select per-image
        # optimal scale using GT at deployment.
        ordered=sorted((details['candidates'][str(s)]['total_delta'],s)
                        for s in scales)
        details['label_oracle_best_step']=float(ordered[0][1])
        details['label_oracle_best_delta']=float(ordered[0][0])
        return details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--weights',choices=['ema','raw'],default='ema')
    p.add_argument('--safe-weights',default=None,help='Pilot sidecar; omit for seeded random sidecar control')
    p.add_argument('--indices',nargs='+',type=int,default=[10,37,53,79,101,137,181,223])
    p.add_argument('--exclude-indices',nargs='*',type=int,default=[0,1])
    p.add_argument('--step-scales',nargs='+',type=float,default=[0.0,0.25,0.5,1.0,2.0])
    p.add_argument('--step-policy',choices=['normalized','legacy_floor'],default='normalized')
    p.add_argument('--basis-gain',type=float,default=1.0)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--eps-abs',type=float,default=1e-5)
    p.add_argument('--eps-rel',type=float,default=1e-5)
    p.add_argument('--device',default='auto')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not args.indices or len(set(args.indices))!=len(args.indices):p.error('Need unique source indices')
    if set(args.indices)&set(args.exclude_indices):p.error('Source audit overlaps forbidden training indices')
    if 0.0 not in args.step_scales or 1.0 not in args.step_scales or len(set(args.step_scales))!=len(args.step_scales) or any(x<0 or not math.isfinite(x) for x in args.step_scales):
        p.error('Include exactly one zero and one unit step; use unique finite nonnegative step scales')
    if args.eps_abs<0 or args.eps_rel<0:p.error('Tolerance must be nonnegative')
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Refuse non-clean-source training data')
    dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,_=setup_model(exp,args.checkpoint,dev,args.weights,args.seed)
    dataset=exp.get_dataset(cache=False)
    if any(i<0 or i>=len(dataset) for i in args.indices):
        p.error(f'Index outside source dataset range 0..{len(dataset)-1}')
    x,_=load_source_example(dataset,args.indices[0],dev,args.seed)
    channels=get_size_from_native(model,x)
    torch.manual_seed(args.seed)
    if args.safe_weights:
        ck=torch.load(args.safe_weights,map_location='cpu',weights_only=False)
        config=ck.get('configuration',{})
        if not isinstance(config,dict) or not config.get('clean_train_only'):
            raise RuntimeError('Unsafe or unknown pilot sidecar provenance')
        if os.path.normcase(os.path.abspath(args.checkpoint))!=os.path.normcase(os.path.abspath(config['checkpoint'])):
            raise RuntimeError('Native checkpoint differs from pilot')
        if config.get('weights')!=args.weights or record_model_hash(model)!=config.get('native_sha256'):
            raise RuntimeError('Native model hash or EMA/raw mismatch')
        args.rank=int(config['rank']);args.basis_gain=float(config['basis_gain'])
        args.step_policy=str(config['step_policy'])
        safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
            basis_gain=args.basis_gain,min_signal_norm=float(config.get('min_signal_norm',1e-8))).to(dev).eval().requires_grad_(False)
        safe.load_state_dict(ck['safe_ttt'],strict=True)
    else:
        safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
                             basis_gain=args.basis_gain).to(dev).eval().requires_grad_(False)
    before_hash=record_model_hash(model)
    report={'protocol':{'checkpoint':args.checkpoint,'weights':args.weights,
        'safe_weights':args.safe_weights,'model_hash':before_hash,'indices':args.indices,
        'step_scales':args.step_scales,'step_policy':args.step_policy,'basis_gain':args.basis_gain,
        'rank':args.rank,'seed':args.seed,'eps_abs':args.eps_abs,'eps_rel':args.eps_rel,
        'labels_used_for_diagnostic':True,'target_domain_labels_used':False,
        'interpretation':'Source-only oracle sweep; never deploy GT-selected scale'}, 'images':[]}
    for i in args.indices:
        r=eval_one(model,safe,dataset,i,dev,args.seed,args.step_scales,args.eps_abs,args.eps_rel)
        report['images'].append(r)
        diffs=' '.join(f'{s}:{r["candidates"][str(s)]["total_delta"]:+.3g}' for s in args.step_scales)
        print(f'IMAGE={i} GT={r["total_gt"]} GSSL={r["ssl_grad_norm"]:.3g} '
              f'STEP={r["base_step_norm"]:.3g} DELTA [ {diffs} ]')
    if record_model_hash(model)!=before_hash:raise RuntimeError('Native model state mutated')
    report['model_state_mutations']=[]
    report['summary']=summarize(report['images'],args.step_scales)
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError(f'Refusing to overwrite: {dst}')
    dst.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('SAVED',dst)
    print('WARNING: Source GT used only for diagnostic; oracle steps are NOT deployable TTT policy')


if __name__=='__main__':
    main()
