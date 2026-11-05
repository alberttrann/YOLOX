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
import os
import tempfile
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


def scalar_image_cosine(value):
    """Convert the audit's per-image [cosine] format without hiding batch errors.

    None represents an undefined cosine when an underlying gradient is zero.
    A multi-item list is a bug for this single-image holdout, not a quantity
    that should be silently averaged or indexed without checking.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError('Expected exactly one cosine for a single-image holdout; got '+str(len(value)))
        value = value[0]
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError('Cosine must be a number or None')
    cosine = float(value)
    if not math.isfinite(cosine) or abs(cosine) > 1.0001:
        raise ValueError('Invalid cosine value: '+repr(value))
    return cosine


def optional_mean(values):
    valid = [float(x) for x in values if x is not None]
    return statistics.mean(valid) if valid else None


def save_sidecar_early(path, state, configuration, steps, before_holdout, movement):
    """Persist trained weights BEFORE holdout evaluation/report formatting.

    A temporary file is atomically renamed only after the complete torch.save.
    Existing destinations are refused, as with the original pilot. On a crash
    after saving, --evaluate-only-sidecar can regenerate the final report.
    """
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('Refusing to overwrite sidecar: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=str(path.parent))
    os.close(fd)
    try:
        torch.save({
            'safe_ttt': {k:v.detach().cpu().clone() for k,v in state.items()},
            'configuration': configuration,
            'steps': steps,
            'before_holdout': before_holdout,
            'predictor_parameter_delta_l2': movement,
            'note': 'Phase A predictor-only pilot; saved before post-training holdout evaluation',
        }, temp)
        if path.exists():
            raise FileExistsError('Refusing to overwrite sidecar: '+str(path))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def validate_report_only_checkpoint(payload, expected):
    if not isinstance(payload, dict) or not isinstance(payload.get('safe_ttt'), dict):
        raise ValueError('Not a Phase A sidecar checkpoint')
    saved=payload.get('configuration')
    if not isinstance(saved,dict):
        raise ValueError('Missing sidecar experiment configuration')
    for key in ('native_sha256','weights','model_exp','seed','train_indices',
                'holdout_indices','steps','lr','rank','basis_gain','step_policy',
                'min_signal_norm','objective_mode','nonregression_weight'):
        if saved.get(key) != expected.get(key):
            raise ValueError(f'Report-only protocol mismatch at {key}: '
                             f'{saved.get(key)!r} != {expected.get(key)!r}')
    if not isinstance(payload.get('steps'),list) or len(payload['steps'])!=expected['steps']:
        raise ValueError('Checkpoint does not contain a complete optimizer-step trace')
    if not isinstance(payload.get('before_holdout'),list):
        raise ValueError('Checkpoint does not contain initial holdout results')


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
                    'initial_cosine':scalar_image_cosine(a['cosine']),
                    'trained_cosine':scalar_image_cosine(b['cosine']),
                    'threshold':tolerance})
    mean=lambda key:statistics.mean(row[key] for row in per)
    return {'sample_size':len(per), 'mean_initial_delta':mean('initial_adapt_minus_native'),
            'mean_trained_delta':mean('trained_adapt_minus_native'),
            'mean_trained_minus_initial':mean('trained_minus_initial'),
            'median_trained_minus_initial':statistics.median(row['trained_minus_initial'] for row in per),
            'mean_cosine_initial':optional_mean(r['initial_cosine'] for r in per),
            'mean_cosine_trained':optional_mean(r['trained_cosine'] for r in per),
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
    p.add_argument('--evaluate-only-sidecar',default=None,
        help='Regenerate holdout report from an already saved Phase A sidecar; no training')
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
    if output.exists() or (args.evaluate_only_sidecar is None and sidecar_path is not None and sidecar_path.exists()):
        raise FileExistsError('Refusing to overwrite existing output/sidecar')
    if args.evaluate_only_sidecar is not None and sidecar_path is not None:
        p.error('Use --evaluate-only-sidecar WITHOUT --save-sidecar')
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
    config={'checkpoint':args.checkpoint,'native_sha256':model_hash,'weights':args.weights,
        'model_exp':args.model_exp,'seed':args.seed,'train_indices':args.train_indices,
        'holdout_indices':args.holdout_indices,'steps':args.steps,'lr':args.lr,
        'rank':args.rank,'basis_gain':args.basis_gain,'step_policy':args.step_policy,
        'min_signal_norm':args.min_signal_norm,'objective_mode':args.objective_mode,
        'nonregression_weight':args.nonregression_weight,'clean_train_only':True,
        'trainable_families':['predictor'],'frozen_families':['adapter','mask_token','step_logit'],
        'source_gt_used_only_for_meta_training_and_offline_holdout':True}

    if args.evaluate_only_sidecar is None:
        # The initial holdout uses the seeded, untrained sidecar.
        before=evaluate_source_holdout(model,safe,dataset,args.holdout_indices,device,args.seed)
        opt=torch.optim.AdamW([p for _,p in chosen],lr=args.lr,weight_decay=0)
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
                  'ADAPT_DET',f'{train_metrics["adapted_detection_loss"]:.7g}',flush=True)
        check_frozen(safe,frozen_snapshot)
        if record_model_hash(model)!=model_hash:
            raise RuntimeError('Native detector mutated')
        pred_movement=predictor_delta(safe,initial_predictor)
        if not pred_movement>0:
            raise RuntimeError('Predictor did not update')
        # Crucial: save finished weights BEFORE any post-training evaluation
        # or report formatting; avoids loss of all trained state on an error.
        if sidecar_path is not None:
            save_sidecar_early(sidecar_path,safe.state_dict(),config,steps,before,pred_movement)
            print('SAVED_SIDECAR_EARLY',sidecar_path,flush=True)
    else:
        # Report recovery from weights saved before a previous reporting failure.
        source=Path(args.evaluate_only_sidecar).resolve()
        if not source.is_file():
            raise FileNotFoundError('Missing saved Phase A checkpoint: '+str(source))
        payload=torch.load(source,map_location='cpu',weights_only=False)
        validate_report_only_checkpoint(payload,config)
        safe.load_state_dict(payload['safe_ttt'],strict=True)
        check_frozen(safe,frozen_snapshot)
        steps=payload['steps']
        before=payload['before_holdout']
        pred_movement=predictor_delta(safe,initial_predictor)
        if not pred_movement>0:
            raise RuntimeError('No saved predictor update found')
        if abs(pred_movement-float(payload['predictor_parameter_delta_l2']))>1e-7:
            raise RuntimeError('Saved predictor delta mismatch')
        print('RECOVERED_SIDECAR',source,'NO_TRAINING',flush=True)

    after=evaluate_source_holdout(model,safe,dataset,args.holdout_indices,device,args.seed)
    check_frozen(safe,frozen_snapshot)
    if record_model_hash(model)!=model_hash:
        raise RuntimeError('Native detector mutated')
    comparison=summarize_holdout(before,after)
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
    print('CAUTION: pilot does not establish AP, AR, adverse DG, or deployable safety.')

if __name__=='__main__':main()
