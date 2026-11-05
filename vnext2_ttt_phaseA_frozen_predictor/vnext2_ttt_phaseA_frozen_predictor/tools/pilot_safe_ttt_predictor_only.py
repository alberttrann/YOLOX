#!/usr/bin/env python3
"""vNext.2 SafeSpatialTTT Phase A: source-clean, predictor-only exact meta-pilot.

Starts with seeded initial sidecar. Keeps native YOLOX, spatial adapter, mask
 token and online step parameter frozen. Optimizes only the SSL predictor with
 detection loss after the true differentiable inner adaptation step.

This is not DG training, an epoch run, or an AP/AR improvement claim.
Requires existing tools/pilot_safe_ttt_meta.py and the other safe-TTT audit
modules. No adverse-condition GT or data is accessed.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from audit_safe_ttt_yolox import setup_model, get_size_from_native, record_model_hash
from pilot_safe_ttt_meta import (
    fixed_native_targets_step, load_source_example, evaluate_source_holdout,
)
from safe_ttt_model import SafeSpatialTTT

FROZEN_GROUPS=('adapter.','mask_token','step_logit')


def select_predictor_only(model):
    """Explicit, fail-closed trainable-parameter selection."""
    selected=[]
    frozen=[]
    for name, parameter in model.named_parameters():
        if name.startswith('predictor.'):
            parameter.requires_grad_(True)
            selected.append((name,parameter))
        elif name.startswith('adapter.') or name in ('mask_token','step_logit'):
            parameter.requires_grad_(False)
            frozen.append((name,parameter))
        else:
            raise RuntimeError('Unknown parameter in SafeSpatialTTT: '+name)
    if not selected or not frozen:
        raise RuntimeError('Expected both predictor and frozen spatial/step parameters')
    if any(name in ('mask_token','step_logit') for name,_ in selected):
        raise RuntimeError('Unexpected parameter selection')
    return selected,frozen


def exact_frozen_snapshot(model):
    return {key:value.detach().cpu().clone()
            for key,value in model.state_dict().items()
            if not key.startswith('predictor.')}


def check_frozen(model,snapshot):
    now=model.state_dict()
    changes=[]
    for key,reference in snapshot.items():
        if key not in now or not torch.equal(now[key].detach().cpu(),reference):
            changes.append(key)
    if changes:
        raise RuntimeError('Non-predictor sidecar state mutated: '+repr(changes))


def predictor_delta(model,snapshot):
    return math.sqrt(sum(float((p.detach().cpu()-snapshot[name]).square().sum())
                         for name,p in model.named_parameters() if name.startswith('predictor.')))


def summarize_holdout(pre,post,eps_abs=1e-5,eps_rel=1e-5):
    if len(pre)!=len(post) or not pre:
        raise RuntimeError('Missing/mismatched holdout')
    per=[]
    for a,b in zip(pre,post):
        if a['index']!=b['index']:
            raise RuntimeError('Holdout order mismatch')
        delta0=float(a['delta']);delta1=float(b['delta'])
        if not math.isfinite(delta0) or not math.isfinite(delta1):
            raise FloatingPointError('Nonfinite holdout delta')
        tolerance=max(eps_abs,eps_rel*abs(float(a['native'])))
        per.append({'index':int(a['index']),
                    'native_loss':float(a['native']),
                    'initial_adapt_minus_native':delta0,
                    'trained_adapt_minus_native':delta1,
                    'trained_minus_initial':delta1-delta0,
                    'train_state_improves_vs_initial':bool(delta1-delta0 < -tolerance),
                    'train_state_harms_vs_initial':bool(delta1-delta0 > tolerance),
                    'initial_helpful':bool(delta0 < -tolerance),
                    'trained_helpful':bool(delta1 < -tolerance),
                    'initial_cosine':float(a['cosine']),
                    'trained_cosine':float(b['cosine']),
                    'threshold':tolerance})
    mean=lambda key:statistics.mean(row[key] for row in per)
    return {'sample_size':len(per), 'mean_initial_delta':mean('initial_adapt_minus_native'),
            'mean_trained_delta':mean('trained_adapt_minus_native'),
            'mean_trained_minus_initial':mean('trained_minus_initial'),
            'median_trained_minus_initial':statistics.median(row['trained_minus_initial'] for row in per),
            'mean_cosine_initial':mean('initial_cosine'),
            'mean_cosine_trained':mean('trained_cosine'),
            'train_state_better_count':sum(r['train_state_improves_vs_initial'] for r in per),
            'train_state_worse_count':sum(r['train_state_harms_vs_initial'] for r in per),
            'initial_beneficial_count':sum(r['initial_helpful'] for r in per),
            'trained_beneficial_count':sum(r['trained_helpful'] for r in per),
            'per_image':per,
            'warning':'Fixed-matching surrogate; tiny source-TRAIN holdout, not held-out clean val AP, not adverse DG.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--weights',choices=('ema','raw'),default='ema')
    p.add_argument('--train-indices',nargs='+',type=int,required=True)
    p.add_argument('--holdout-indices',nargs='+',type=int,required=True)
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--nonregression-weight',type=float,default=0.5)
    p.add_argument('--objective-mode',choices=('detector_total','scale_balanced'),default='detector_total')
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--basis-gain',type=float,default=1.0)
    p.add_argument('--step-policy',choices=('legacy_floor','normalized'),default='normalized')
    p.add_argument('--min-signal-norm',type=float,default=1e-8)
    p.add_argument('--device',default='auto')
    p.add_argument('--output',required=True)
    p.add_argument('--save-sidecar',default=None)
    args=p.parse_args()
    if not (1 <= args.steps <= 64):p.error('--steps must be in 1..64; this is a mini-pilot, not an epoch run')
    if not (0<args.lr<=1e-2) or not math.isfinite(args.lr):p.error('Need finite 0 < LR <= 0.01')
    if not math.isfinite(args.nonregression_weight) or args.nonregression_weight<0:p.error('Invalid nonregression weight')
    if not args.train_indices or not args.holdout_indices:
        p.error('Supply disjoint nonempty train and holdout index lists')
    all_ids=args.train_indices+args.holdout_indices
    if any(i<0 for i in all_ids) or len(all_ids)!=len(set(all_ids)):
        p.error('Indices must be nonnegative and unique')
    # Freeze index set before training; repeating a tiny pair of images is not
    # a credible test of source-diverse meta-learning.
    if len(args.train_indices)<min(args.steps,8):
        p.error('Use >= min(steps,8) distinct source examples, not just two images')
    if len(args.holdout_indices)<4:
        p.error('Use >=4 disjoint heldout source examples')
    output=Path(args.output).resolve()
    sidecar_path=Path(args.save_sidecar).resolve() if args.save_sidecar else None
    if output.exists() or (sidecar_path is not None and sidecar_path.exists()):
        raise FileExistsError('Refusing to overwrite existing output/sidecar')
    if sidecar_path is not None and sidecar_path==output:
        p.error('Output report and sidecar paths must differ')
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Source-clean TRAIN annotation required. This script forbids adverse labels.')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,_=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    dataset=exp.get_dataset(cache=False)
    if max(all_ids)>=len(dataset):
        raise IndexError(f'Source example outside dataset length {len(dataset)}')
    model_hash=record_model_hash(model)
    x,_=load_source_example(dataset,args.train_indices[0],device,args.seed)
    channels=get_size_from_native(model,x)
    torch.manual_seed(args.seed)
    safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
         min_signal_norm=args.min_signal_norm,basis_gain=args.basis_gain).to(device).train()
    initial_predictor={name:p.detach().cpu().clone() for name,p in safe.named_parameters()
                       if name.startswith('predictor.')}
    chosen,frozen=select_predictor_only(safe)
    frozen_snapshot=exact_frozen_snapshot(safe)
    opt=torch.optim.AdamW([p for _,p in chosen],lr=args.lr,weight_decay=0)
    before=evaluate_source_holdout(model,safe,dataset,args.holdout_indices,device,args.seed)
    steps=[]
    for iteration in range(args.steps):
        index=args.train_indices[iteration % len(args.train_indices)]
        xb,yb=load_source_example(dataset,index,device,args.seed)
        train_metrics=fixed_native_targets_step(model,safe,xb,yb,opt,
            nonregression_weight=args.nonregression_weight,
            objective_mode=args.objective_mode)
        train_metrics['step']=iteration+1
        train_metrics['index']=index
        if train_metrics['adapter_grad_norm']>0:
            raise RuntimeError('Frozen adapter unexpectedly received a gradient')
        check_frozen(safe,frozen_snapshot)
        steps.append(train_metrics)
        print('STEP',iteration+1,'IDX',index,
              'META_PRED_GRAD',f'{train_metrics["meta_predictor_grad_norm"]:.5g}',
              'ADAPT_DET',f'{train_metrics["adapted_detection_loss"]:.7g}')
    after=evaluate_source_holdout(model,safe,dataset,args.holdout_indices,device,args.seed)
    check_frozen(safe,frozen_snapshot)
    if record_model_hash(model)!=model_hash:
        raise RuntimeError('Native detector mutated')
    pred_movement=predictor_delta(safe,initial_predictor)
    if not pred_movement>0:
        raise RuntimeError('Predictor did not update')
    comparison=summarize_holdout(before,after)
    config={'checkpoint':args.checkpoint,'native_sha256':model_hash,'weights':args.weights,
        'model_exp':args.model_exp,'seed':args.seed,'train_indices':args.train_indices,
        'holdout_indices':args.holdout_indices,'steps':args.steps,'lr':args.lr,
        'rank':args.rank,'basis_gain':args.basis_gain,'step_policy':args.step_policy,
        'min_signal_norm':args.min_signal_norm,'objective_mode':args.objective_mode,
        'nonregression_weight':args.nonregression_weight,'clean_train_only':True,
        'trainable_families':['predictor'],'frozen_families':['adapter','mask_token','step_logit'],
        'source_gt_used_only_for_meta_training_and_offline_holdout':True}
    report={'protocol':config,'steps':steps,'predictor_parameter_delta_l2':pred_movement,
            'frozen_state_changed_keys':[],'native_model_mutations':[],
            'heldout':comparison,'before_holdout':before,'after_holdout':after,
            'limitations':['NO ACDC validation','NO AP/AR metric','not guaranteed safe','pilot with fixed source matches']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print('HELDOUT mean initial',f'{comparison["mean_initial_delta"]:+.6g}',
          'mean trained',f'{comparison["mean_trained_delta"]:+.6g}',
          'trained_minus_initial',f'{comparison["mean_trained_minus_initial"]:+.6g}')
    print('HELDOUT better',comparison['train_state_better_count'],'worse',comparison['train_state_worse_count'],
          'n',comparison['sample_size'])
    print('PREDICTOR_L2_CHANGE',f'{pred_movement:.6g}','FROZEN_PARAMS',len(frozen),'CHANGED',0)
    print('SAVED',output)
    if sidecar_path is not None:
        sidecar_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'safe_ttt':{k:v.detach().cpu() for k,v in safe.state_dict().items()},
            'configuration':config,'note':'source-only predictor-only meta pilot'},sidecar_path)
        print('SAVED_SIDECAR',sidecar_path)
    print('CAUTION: pilot does not establish AP, AR, adverse DG, or deployable safety.')

if __name__=='__main__':main()
