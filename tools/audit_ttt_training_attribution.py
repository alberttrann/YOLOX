#!/usr/bin/env python3
"""vNext.2 SafeSpatialTTT — deterministic 4-family source-clean training attribution.

READ-ONLY: factorial swaps between (a) the seeded initial sidecar, and
(b) the two-step-trained sidecar. No labels enter the online SSL update.
Source TRAIN ground truth is used only for the fixed-native-target diagnostic.

Requires tools from the already delivered safe-TTT YOLOX audit package:
  audit_safe_ttt_yolox.py, pilot_safe_ttt_meta.py, safe_ttt_model.py,
  native_dark2_router.py, fixed_detection_targets.py

The factorial attribution exposes which parameter family changed the finite
adaptation effect. It CANNOT establish that any family is intrinsically good
or bad after real training; mixed hybrids may be off the trained distribution.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics

import torch

from audit_safe_ttt_yolox import setup_model,get_size_from_native,head_loss_pass,record_model_hash
from pilot_safe_ttt_meta import load_source_example
from fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity
from native_dark2_router import Dark2AuditRouter
from safe_ttt_model import SafeSpatialTTT

GROUPS=('adapter','predictor','mask_token','step_logit')


def family(key):
    if key.startswith('adapter.'):
        return 'adapter'
    if key.startswith('predictor.'):
        return 'predictor'
    if key in ('mask_token','step_logit'):
        return key
    raise ValueError('Unknown parameter/buffer: '+key)


def check_states(initial,trained):
    if list(initial)!=list(trained):
        raise RuntimeError('Initial/trained sidecar state_dict keys differ')
    changes={g:{'changed_tensors':0,'total_tensors':0,'max_abs_change':0.,'delta_l2':0.,'relative_l2':0.,'initial_l2':0.} for g in GROUPS}
    for key in initial:
        group=family(key)
        i=initial[key].detach().float().cpu()
        t=trained[key].detach().float().cpu()
        if i.shape!=t.shape or not (torch.isfinite(i).all() and torch.isfinite(t).all()):
            raise RuntimeError('Nonfinite or mismatched tensor in '+key)
        d=t-i
        row=changes[group]
        row['total_tensors']+=1
        row['changed_tensors']+=int(bool((d!=0).any()))
        row['max_abs_change']=max(row['max_abs_change'],float(d.abs().max()))
        row['delta_l2']+=float(d.square().sum())
        row['initial_l2']+=float(i.square().sum())
    for g,row in changes.items():
        row['delta_l2']=math.sqrt(row['delta_l2'])
        row['initial_l2']=math.sqrt(row['initial_l2'])
        row['relative_l2']=row['delta_l2']/max(row['initial_l2'],1e-12)
    return changes


def combine(initial,trained,chosen):
    """Full strict-key replacement, never blended numerical interpolation."""
    if any(g not in GROUPS for g in chosen):
        raise ValueError('Unknown group choice')
    chosen=frozenset(chosen)
    if list(initial)!=list(trained):
        raise RuntimeError('Mismatched sidecar state keys')
    return {key:(trained if family(key) in chosen else initial)[key].detach().clone()
            for key in initial}


def subsets():
    return [tuple(s) for n in range(len(GROUPS)+1) for s in itertools.combinations(GROUPS,n)]


def shapley(values):
    """Exact Shapley decomposition for a 4-group finite intervention game.

    values: keys are frozenset(group names), values are scalar measured effects.
    Invariant: sum(phi_g)==value(all)-value(empty).
    """
    n=len(GROUPS)
    phi={g:0. for g in GROUPS}
    for g in GROUPS:
        rest=[x for x in GROUPS if x!=g]
        for k in range(n):
            for subset in itertools.combinations(rest,k):
                S=frozenset(subset)
                weight=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
                phi[g]+=weight*(values[S|{g}]-values[S])
    residual=sum(phi.values())-(values[frozenset(GROUPS)]-values[frozenset()])
    if abs(residual)>1e-9:
        raise AssertionError(f'Shapley additivity failed: {residual}')
    return phi


def as_float(x):
    return float(x.detach()) if isinstance(x,torch.Tensor) else float(x)


def one_image(model, hybrids, dataset, index, device, seed, strength,eps_abs=1e-5,eps_rel=1e-5):
    images,labels=load_source_example(dataset,index,device,seed)
    if images.shape[0]!=1:raise RuntimeError('B=1 only')
    safe=next(iter(hybrids.values()))
    results={}
    with Dark2AuditRouter(model,safe) as router, NativeMatchReplay(model.head) as match:
        router.set_mode('native')
        with torch.no_grad():
            raw=head_loss_pass(model,images,labels)
        router.require_one_hit(1); match.finalize()
        native=reconstruct_tde_loss(model.head,match)
        assert_head_loss_parity(raw,native)
        native_loss=as_float(native['total'])
        feature=router.last_native.detach().float()
        # Measure a second native pass with identical model/input/assignments;
        # don't equate an arbitrary tolerance with actual numeric noise.
        router.set_mode('native');match.replay()
        with torch.no_grad():
            repeated_raw=head_loss_pass(model,images,labels)
        router.require_one_hit(1);match.finalize()
        repeated=reconstruct_tde_loss(model.head,match)
        assert_head_loss_parity(repeated_raw,repeated)
        repeat_error=abs(as_float(repeated['total'])-native_loss)
        for combo,sidecar in hybrids.items():
            router.safe=sidecar
            with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
                z0=torch.zeros(1,sidecar.rank,device=device,dtype=torch.float32,requires_grad=True)
                ssl=sidecar.ssl_loss(feature,z0).sum()
                gssl,=torch.autograd.grad(ssl,z0)
                if not torch.isfinite(gssl).all():raise FloatingPointError('Nonfinite SSL gradient')
                step=sidecar.compute_step(gssl).detach()*strength
            # Measure detector gradient at z=0 for this *specific adapter basis*.
            router.set_mode('fixed_z',z=z0);match.replay()
            with torch.enable_grad():
                z_raw=head_loss_pass(model,images,labels)
                router.require_one_hit(1);match.finalize()
                z_result=reconstruct_tde_loss(model.head,match)
                assert_head_loss_parity(z_raw,z_result)
                gdet,=torch.autograd.grad(z_result['total'],z0)
            if abs(as_float(z_result['total'])-native_loss)>1e-5:
                raise RuntimeError('Zero-state/native mismatch; cannot attribute result')
            if not torch.isfinite(gdet).all():raise FloatingPointError('Nonfinite detector gradient')
            norm_ssl=as_float(gssl.norm());norm_det=as_float(gdet.norm())
            cosine=as_float((gssl.detach()*gdet.detach()).sum())/max(norm_ssl*norm_det,1e-20)
            router.set_mode('fixed_z',z=step);match.replay()
            with torch.no_grad():
                after_raw=head_loss_pass(model,images,labels)
                router.require_one_hit(1);match.finalize()
                after=reconstruct_tde_loss(model.head,match)
                assert_head_loss_parity(after_raw,after)
            difference=as_float(after['total'])-native_loss
            # Independently rerun the original and full-trained interventions
            # to reveal numerical variability of *adapted* forward itself.
            adapter_repeat_error=None
            if not combo or combo==GROUPS:
                router.set_mode('fixed_z',z=step);match.replay()
                with torch.no_grad():
                    repeat_after_raw=head_loss_pass(model,images,labels)
                    router.require_one_hit(1);match.finalize()
                    repeat_after=reconstruct_tde_loss(model.head,match)
                    assert_head_loss_parity(repeat_after_raw,repeat_after)
                adapter_repeat_error=abs(as_float(repeat_after['total'])-as_float(after['total']))
            comp={key:as_float(value-native['components'][key]) for key,value in after['components'].items()}
            scales={key:sum(as_float(v-native['scales'][key][k]) for k,v in sub.items())
                    for key,sub in after['scales'].items()}
            nonreg=all(value<=max(eps_abs,eps_rel*abs(sum(as_float(x) for x in native['scales'][key].values())))
                       for key,value in scales.items() if native['counts'][key]>0)
            bg=comp['background_objectness']<=max(eps_abs,eps_rel*abs(as_float(native['components']['background_objectness'])))
            tol=max(eps_abs,eps_rel*abs(native_loss))
            with torch.no_grad():
                adapted=sidecar.adapter(feature,step)
                rel_l2=as_float((adapted-feature).square().mean().sqrt()/feature.square().mean().clamp_min(1e-8).sqrt())
            results['+'.join(combo) or 'initial']={
                'total_delta':difference,'cosine':cosine,'adapted_repeat_error':adapter_repeat_error,
                'ssl_gradient_norm':norm_ssl,'detector_gradient_norm':norm_det,
                'state_step_norm':as_float(step.norm()),'feature_relative_l2':rel_l2,
                'taylor_predicted_delta':as_float((gdet.detach()*step).sum()),
                'component_deltas':comp,'scale_deltas':scales,
                'strict_surrogate_pass':bool(difference < -tol and nonreg and bg),
                'beneficial':bool(difference < -tol),
                'scale_nonregressing':bool(nonreg),'background_nonregressing':bool(bg),
            }
    return {'index':index,'native_loss':native_loss,'native_retest_error':repeat_error,
            'foreground':native['counts']['foreground'],'groups':results}


def run(args):
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Require source clean TRAIN dataset; do not use adverse labels')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,_=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    orig_hash=record_model_hash(model)
    # Reconstruct initialization with same seeding order as pilot_safe_ttt_meta.py
    dataset=exp.get_dataset(cache=False)
    images,_=load_source_example(dataset,args.indices[0],device,args.seed)
    channels=get_size_from_native(model,images)
    ckpt=torch.load(args.trained_sidecar,map_location='cpu',weights_only=False)
    config=ckpt.get('configuration')
    if not isinstance(config,dict) or not config.get('clean_train_only'):
        raise RuntimeError('Missing valid source-only pilot provenance')
    if os.path.normcase(os.path.abspath(args.checkpoint))!=os.path.normcase(os.path.abspath(config['checkpoint'])):
        raise RuntimeError('Native checkpoint path mismatch')
    if config['weights']!=args.weights or config['native_sha256']!=orig_hash:
        raise RuntimeError('Native checkpoint hash or weight branch mismatch')
    if config.get('seed') is not None and args.seed!=config.get('seed'):
        raise RuntimeError('Seed differs from training init; refusing invalid attribution')
    # The original 2026-11-04 pilot omitted the seed from saved configuration.
    # To reconstruct its initialization, require the original untrained report
    # with the exact seed and verify the actual 0.5-step outputs below.
    if config.get('seed') is None and not args.reference_initial:
        raise RuntimeError('Pilot did not persist seed. Supply --reference-initial '
             'with original untrained sweep JSON so the reconstruction can be validated.')
    if not set(args.indices).isdisjoint(set(config.get('train_indices',[]))):
        raise RuntimeError('Analysis indices overlap meta-training indices')
    torch.manual_seed(args.seed)
    initial=SafeSpatialTTT(channels,rank=int(config['rank']),step_policy=config['step_policy'],
        basis_gain=float(config['basis_gain']),min_signal_norm=float(config.get('min_signal_norm',1e-8))).to(device).eval().requires_grad_(False)
    trained=copy.deepcopy(initial)
    trained.load_state_dict(ckpt['safe_ttt'],strict=True)
    init_state=initial.state_dict();trained_state=trained.state_dict()
    changes=check_states(init_state,trained_state)
    combos=subsets()
    hybrids={}
    for combo in combos:
        model_c=copy.deepcopy(initial)
        model_c.load_state_dict(combine(init_state,trained_state,combo),strict=True)
        model_c.eval().requires_grad_(False)
        hybrids[combo]=model_c
    report={'protocol':{'source_clean':True,'index_list':args.indices,'seed':args.seed,
        'native_sha256':orig_hash,'training_checkpoint':args.trained_sidecar,
        'checkpoint':args.checkpoint,'weights':args.weights,'strength':args.strength,
        'trained_on':config['train_indices'],'analysis_groups':GROUPS,
        'offline_gt_only':True,'note':'Factorial hybrid parameter-state interventions, not causal generalization guarantees'},
        'parameter_changes':changes,'images':[]}
    for idx in args.indices:
        r=one_image(model,hybrids,dataset,idx,device,args.seed,args.strength,args.eps_abs,args.eps_rel)
        report['images'].append(r)
        print('IMAGE',idx,'native_repeat_err',f'{r["native_retest_error"]:.3g}',
              'untrained',f'{r["groups"]["initial"]["total_delta"]:+.6g}',
              'trained',f'{r["groups"]["+".join(GROUPS)]["total_delta"]:+.6g}')
    if record_model_hash(model)!=orig_hash:
        raise RuntimeError('Native model checkpoint state mutated')
    for refpath,groupkey,display in (
        (args.reference_initial,'initial','untrained'),
        (args.reference_trained,'+'.join(GROUPS),'trained'),
    ):
        if refpath:
            ref=json.loads(Path(refpath).read_text(encoding='utf8'))
            proto=ref.get('protocol',{})
            if (proto.get('model_hash')!=orig_hash or proto.get('weights')!=args.weights
                    or proto.get('seed')!=args.seed or proto.get('indices')!=args.indices
                    or args.strength not in proto.get('step_scales',[])):
                raise RuntimeError('Earlier step-sweep protocol differs from current attribution')
            records={int(rec['index']):rec for rec in ref['images']}
            for rec in report['images']:
                expected=float(records[rec['index']]['candidates'][str(args.strength)]['total_delta'])
                observed=float(rec['groups'][groupkey]['total_delta'])
                if abs(expected-observed)>args.reference_tolerance:
                    raise RuntimeError(f'{display} reproduction failed image {rec["index"]}: '
                       f'current {observed:+.8g} vs earlier {expected:+.8g}; '
                       f'tolerance {args.reference_tolerance:g}')
            print('PASS prior-sweep reproducibility:',display,'max_abs_tolerance=',args.reference_tolerance)
    mean_games={}
    for combo in combos:
        key='+'.join(combo) or 'initial'
        mean_games[frozenset(combo)]=statistics.mean(im['groups'][key]['total_delta'] for im in report['images'])
    phi=shapley(mean_games)
    report['attribution']={'shapley_mean_loss_effects':phi,
        'initial_mean_delta':mean_games[frozenset()],
        'trained_mean_delta':mean_games[frozenset(GROUPS)],
        'train_minus_init':mean_games[frozenset(GROUPS)]-mean_games[frozenset()],
        'caution':'Shapley distributes interactions among parameter-family swaps; it does not prove which family caused overfitting'}
    print('GROUP SHAPLEY (positive = deterioration):',', '.join(f'{k}={v:+.6g}' for k,v in phi.items()))
    print('TRAINED-INITIAL MEAN LOSS DELTA:',f'{report["attribution"]["train_minus_init"]:+.6g}')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Refuse overwrite: '+str(out))
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print('SAVED',out)
    print('This is a source-only attribution probe, not proof of ACDC generalization.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    ap.add_argument('--weights',choices=['ema','raw'],default='ema')
    ap.add_argument('--trained-sidecar',required=True)
    ap.add_argument('--indices',nargs='+',type=int,default=[10,37,53,79,101,137,181,223])
    ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--strength',type=float,default=0.5)
    ap.add_argument('--eps-abs',type=float,default=1e-5)
    ap.add_argument('--eps-rel',type=float,default=1e-5)
    ap.add_argument('--reference-initial',default=None,help='Optional prior untrained sweep JSON')
    ap.add_argument('--reference-trained',default=None,help='Optional prior trained sweep JSON')
    ap.add_argument('--reference-tolerance',type=float,default=2e-5)
    ap.add_argument('--device',default='auto')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if len(args.indices)==0 or len(args.indices)!=len(set(args.indices)) or min(args.indices)<0:
        ap.error('Need unique nonnegative indices')
    if not math.isfinite(args.strength) or args.strength<0 or args.strength>2:
        ap.error('Strength must be finite and between 0 and 2')
    if args.reference_tolerance<0 or not math.isfinite(args.reference_tolerance):
        ap.error('Reference tolerance must be finite and nonnegative')
    if args.eps_abs<0 or args.eps_rel<0:
        ap.error('Nonnegative tolerances required')
    run(args)


if __name__=='__main__':
    main()
