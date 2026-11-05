#!/usr/bin/env python3
"""vNext.2 Safe TTT Phase C: repaired detector-unit constraints and episode accumulation.

Native YOLOX, spatial basis, mask token, online step controller are all frozen.
The *test-time* inner loss is the SAME per-image unlabeled masked prediction as
before. The OUTER training objective supervises that inner step through the
real frozen-matching TDE head, on a geometry-preserving photometric view.

Optional (disabled by default) clean-reference teacher consistency is source-
training-only. This is a bounded pilot, NOT an AP or OOD generalization claim.
Run only using clean TRAIN labels. Never use ACDC to select settings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from audit_safe_ttt_yolox import setup_model, get_size_from_native, record_model_hash, head_loss_pass
from fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity
from native_dark2_router import Dark2AuditRouter
from pilot_safe_ttt_meta import load_source_example
from pilot_safe_ttt_predictor_only import select_predictor_only,exact_frozen_snapshot,check_frozen,predictor_delta
from safe_ttt_model import SafeSpatialTTT
from ttt_source_views import sample_appearance,apply_appearance,image_fingerprint,spec_dict
from ttt_phaseC_constraints import ConstraintConfig, detector_unit_objective

SCALES=('small','medium','large')
OBJECTIVES=('detector_total','scale_balanced')


def reference_consistency(clean_logits, adapted_logits, native_matches,
                          image_shape):
    """Optional clean teacher penalty on GT-associated anchors only.

    This is an *outer* source-label training regularizer. The SSL inner update
    never sees the clean reference, labels, or this loss. No teacher loss in TTT
    inference. Default coefficient is ZERO because epoch-2 teacher is weak.
    """
    if clean_logits is None:
        raise ValueError('No clean teacher predictions')
    if clean_logits.shape!=adapted_logits.shape or clean_logits.ndim!=3:
        raise RuntimeError('Unmatched clean/shifted detector anchor grids')
    if len(native_matches.stored)!=1 or clean_logits.shape[0]!=1:
        raise RuntimeError('One source image and one frozen native assignment required')
    fg=native_matches.stored[0].match[1].to(device=adapted_logits.device,dtype=torch.bool)
    if fg.numel()!=adapted_logits.shape[1]:
        raise RuntimeError('Native foreground match mask incompatible with teacher')
    if not bool(fg.any()):
        return adapted_logits.sum()*0.0,0
    teacher=clean_logits[0,fg].detach().float()
    student=adapted_logits[0,fg].float()
    # A relative box coordinate loss; smooth bounded derivatives.
    h,w=image_shape[-2:]
    norm=student.new_tensor([w,h,w,h]).reshape(1,4)
    box=F.smooth_l1_loss(student[:,:4]/norm,teacher[:,:4]/norm)
    # Avoid hard teacher targets/temperature sharpening that amplify errors.
    probs=(student[:,4:].sigmoid()-teacher[:,4:].sigmoid()).square().mean()
    return box+probs,int(fg.sum())


def _view_meta_backward(model,safe,images,labels,*,view_weight,
                        clean_view, constraint_config,
                        teacher_logits=None,teacher_weight=0.0):
    """Accumulate ONE view's exact second-order predictor gradients.

    Caller takes optimizer step only after both views; the two computational
    graphs are released independently, avoiding double-view peak VRAM.
    """
    with Dark2AuditRouter(model,safe) as router,NativeMatchReplay(model.head) as replay:
        router.set_mode('native')
        with torch.no_grad():native_head=head_loss_pass(model,images,labels)
        router.require_one_hit(1);replay.finalize()
        native=reconstruct_tde_loss(model.head,replay)
        native_parity=assert_head_loss_parity(native_head,native)
        native_logits=replay.head_inputs['outputs'].detach().clone()
        router.set_mode('episode',meta_train=True)
        replay.replay()
        with torch.enable_grad():
            adapted_head=head_loss_pass(model,images,labels)
            router.require_one_hit(1);replay.finalize()
            adapted=reconstruct_tde_loss(model.head,replay)
            adapted_parity=assert_head_loss_parity(adapted_head,adapted)
            loss,details=detector_unit_objective(native,adapted,
               clean_view=clean_view,config=constraint_config)
            if teacher_weight:
                consistency,nteacher=reference_consistency(
                    teacher_logits,replay.head_inputs['outputs'],replay,images.shape)
                loss=loss+teacher_weight*consistency
            else:
                consistency=loss*0.0; nteacher=0
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError('Nonfinite per-view meta objective')
            if view_weight:
                (view_weight*loss).backward()
        return ({'native_detection_loss':float(native['total'].detach()),
                 'adapted_detection_loss':float(adapted['total'].detach()),
                 'adapted_minus_native':float((adapted['total']-native['total']).detach()),
                 'meta_objective':float(loss.detach()),
                 'source_teacher_consistency':float(consistency.detach()),
                 'source_teacher_positive_anchors':nteacher,
                 'native_parity':native_parity,'adapted_parity':adapted_parity,
                 'nonregression':details},native_logits)


def paired_accumulate_one(model,safe,clean,shifted,labels,*,
                         constraint_config,clean_weight,teacher_weight,accumulation_count):
    """Backprop one pair with normalized contribution, but do not optimizer.step.

    Each pair contributes one independent, source-labeled supervised meta-episode.
    Train-only GT never enters the inner adaptation. Caller zeros gradients once
    and updates only after accumulation_count distinct episodes.
    """
    if clean.shape!=shifted.shape or clean.shape[0]!=1 or labels.shape[0]!=1:
        raise RuntimeError('Phase C requires B=1 geometry-aligned clean/shifted pair')
    if torch.equal(clean,shifted):
        raise RuntimeError('Shifted view identical to clean image')
    if not math.isfinite(clean_weight) or clean_weight <=0:
        raise ValueError('Clean view weight must be positive')
    if not math.isfinite(teacher_weight) or teacher_weight <0:
        raise ValueError('Invalid teacher weight')
    if not isinstance(accumulation_count,int) or accumulation_count <=0:
        raise ValueError('Invalid accumulation size')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Native detector must be frozen')
    if model.training or model.backbone.training or not model.head.training:
        raise RuntimeError('Model/head training-state contract changed')
    # Normalized view weighting, so equal weights mean an actual 50/50 mixture.
    denom=(1.0+clean_weight)*accumulation_count
    clean_report,clean_logits=_view_meta_backward(model,safe,clean,labels,
        view_weight=clean_weight/denom,clean_view=True,
        constraint_config=constraint_config)
    shifted_report,_=_view_meta_backward(model,safe,shifted,labels,
        view_weight=1.0/denom,clean_view=False,
        constraint_config=constraint_config,teacher_logits=clean_logits,
        teacher_weight=teacher_weight)
    return {'clean':clean_report,'shifted':shifted_report}


def optimizer_step_predictor(safe,optimizer,*,max_grad_norm):
    selected=[p for p in safe.predictor.parameters() if p.requires_grad]
    if not selected:raise RuntimeError('No trainable predictor')
    norm=math.sqrt(sum(float(p.grad.detach().float().square().sum())
                       for p in selected if p.grad is not None))
    if not math.isfinite(norm) or norm<=0:
        raise RuntimeError('No finite detection-to-SSL meta-gradient')
    if any(p.grad is not None for n,p in safe.named_parameters() if not n.startswith('predictor.')):
        raise RuntimeError('Frozen sidecar parameters received gradient')
    preclip=float(torch.nn.utils.clip_grad_norm_(selected,max_grad_norm,error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {'predictor_gradient_l2':norm,'preclip_gradient_norm':preclip}


def paired_eval_one(model,safe,images,labels):
    """Read-only clean-source evaluator using frozen native matching and TTT.

    The native YOLOX is frozen; the sidecar's parameters are never updated.
    Both passes see the SAME input/labels, with only the episodic fast state
    differing. No source labels enter the inner adaptation.
    """
    with Dark2AuditRouter(model,safe) as router,NativeMatchReplay(model.head) as replay:
        router.set_mode('native')
        with torch.no_grad():head0=head_loss_pass(model,images,labels)
        router.require_one_hit(1); replay.finalize()
        nat=reconstruct_tde_loss(model.head,replay)
        err0=assert_head_loss_parity(head0,nat)
        router.set_mode('episode',meta_train=False)
        replay.replay()
        with torch.no_grad():head1=head_loss_pass(model,images,labels)
        router.require_one_hit(1); replay.finalize()
        ada=reconstruct_tde_loss(model.head,replay)
        err1=assert_head_loss_parity(head1,ada)
        before=float(nat['total'].detach());after=float(ada['total'].detach())
        components={k:float((ada['components'][k]-nat['components'][k]).detach()) for k in nat['components']}
        scales={}
        for name in SCALES:
            count=nat['counts'][name]
            if count:
                factor=nat['num_foreground_normalizer']/count
                scales[name]={
                  'count':int(count),
                  'native_per_matched_positive':float((sum(nat['scales'][name].values())*factor).detach()),
                  'adapted_minus_native_per_positive':float((sum(ada['scales'][name].values())-sum(nat['scales'][name].values())).detach()*factor),
                  'components':{c:float((ada['scales'][name][c]-nat['scales'][name][c]).detach()*factor)
                                for c in ('box','classification','positive_objectness')},
                }
        bg_factor=nat['num_foreground_normalizer']/max(1,nat['counts']['background'])
        bg={'count':int(nat['counts']['background']),
            'delta_per_background_anchor':float((ada['components']['background_objectness']-
                   nat['components']['background_objectness']).detach()*bg_factor)}
        stats=router.last_stats
        return {'native':before,'adapted':after,'delta':after-before,
                'counts':{k:int(v) for k,v in nat['counts'].items()},
                'components':components,'scales':scales,'background':bg,
                'fast_state_l2':float(stats['fast_state_norm'][0]),
                'relative_feature_change_postcast':float(stats['relative_feature_change_postcast'][0]),
                'changed_feature_fraction':float(stats['postcast_changed_fraction'][0]),
                'loss_parity':{'native':err0,'adapted':err1}}


def evaluate_views(model,safe,dataset,indices,device,seed,strength):
    records=[]
    safe.eval()
    for index in indices:
        images,labels=load_source_example(dataset,index,device,seed)
        spec=sample_appearance(seed,index,strength=strength)
        shifted=apply_appearance(images,spec)
        record={'index':int(index),'appearance':spec_dict(spec),
                'clean_sha256':image_fingerprint(images),
                'shifted_sha256':image_fingerprint(shifted),
                'clean':paired_eval_one(model,safe,images,labels),
                'shifted':paired_eval_one(model,safe,shifted,labels)}
        records.append(record)
    return records


def summarize_views(before,after,*,constraint_config,abs_threshold=2e-5,relative_threshold=1e-5):
    """Holdout safety diagnostics in the SAME units as training constraints.

    The strict criterion is intentionally stringent; numerical significance
    must be established separately, and no source-label gate is deployable.
    """
    if len(before)!=len(after) or not before:
        raise RuntimeError('Unmatched source holdouts')
    result={}
    for view in ('clean','shifted'):
        entries=[]
        for b,a in zip(before,after):
            if b['index']!=a['index'] or b['clean_sha256']!=a['clean_sha256'] or b['shifted_sha256']!=a['shifted_sha256']:
                raise RuntimeError('Holdout identities or appearance changed')
            pre=b[view]; post=a[view]
            if abs(pre['native']-post['native'])>2e-5:
                raise RuntimeError('Native detector benchmark changed')
            tol=max(abs_threshold,relative_threshold*abs(post['native']))
            pd=post['delta']-pre['delta']
            scale_regressions=[]
            for s,v in post['scales'].items():
                if v['adapted_minus_native_per_positive']>constraint_config.scale_margin:
                    scale_regressions.append(s)
            bg_total=post['components']['background_objectness']
            bg_bad=bg_total>constraint_config.background_margin
            strict=(post['delta'] < -tol and not scale_regressions and not bg_bad)
            entries.append({'index':a['index'],'initial_delta':pre['delta'],
               'trained_delta':post['delta'],'paired_delta':pd,'tolerance':tol,
               'better_than_initial':pd<-tol,'worse_than_initial':pd>tol,
               'adaptation_helpful':post['delta']<-tol,'adaptation_harmful':post['delta']>tol,
               'strict_all_present_scales_and_bg':strict,
               'scale_regressions':scale_regressions,'background_regression':bg_bad,
               'background_delta_detector_units':bg_total})
        result[view]={'mean_initial_delta':statistics.mean(r['initial_delta'] for r in entries),
            'mean_trained_delta':statistics.mean(r['trained_delta'] for r in entries),
            'mean_paired_delta':statistics.mean(r['paired_delta'] for r in entries),
            'better_than_initial':sum(r['better_than_initial'] for r in entries),
            'worse_than_initial':sum(r['worse_than_initial'] for r in entries),
            'trained_helpful':sum(r['adaptation_helpful'] for r in entries),
            'trained_harmful':sum(r['adaptation_harmful'] for r in entries),
            'trained_strict':sum(r['strict_all_present_scales_and_bg'] for r in entries),
            'background_violations_detector_units':sum(r['background_regression'] for r in entries),
            'scale_coverage':{s:sum(s in v[view]['scales'] for v in after) for s in SCALES},
            'per_image':entries}
    return result


def _atomic_save_sidecar(path,model,config,trace,before,movement):
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('Refusing to overwrite sidecar '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=str(path.parent))
    os.close(fd)
    try:
        torch.save({'safe_ttt':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'configuration':config,'steps':trace,'before_holdout':before,
                    'predictor_parameter_delta_l2':movement,
                    'note':'Phase B source-paired predictor-only meta pilot; not a validated DG model'},tmp)
        if path.exists():raise FileExistsError('Refusing to overwrite sidecar '+str(path))
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def _expected_config(args,model_hash,guard_config):
    return {'checkpoint':str(Path(args.checkpoint).resolve()),
      'model_exp':args.model_exp,'native_sha256':model_hash,'weights':args.weights,
      'seed':args.seed,'rank':args.rank,'basis_gain':args.basis_gain,
      'step_policy':args.step_policy,'min_signal_norm':args.min_signal_norm,
      'train_indices':args.train_indices,'holdout_indices':args.holdout_indices,
      'steps':args.steps,'episodes_per_step':args.episodes_per_step,
      'lr':args.lr,'strength':args.strength,'clean_weight':args.clean_weight,
      'teacher_weight':args.teacher_weight,
      'constraint_config':vars(guard_config),
      'protocol':'Phase C detector-unit guards, clean+shifted per episode, predictor only',
      'source_labels_only_in_outer':True,
      'test_time_labels_used':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    ap.add_argument('--weights',choices=('ema','raw'),default='ema')
    ap.add_argument('--train-start',type=int,default=300)
    ap.add_argument('--train-count',type=int,default=32)
    ap.add_argument('--holdout-start',type=int,default=500)
    ap.add_argument('--holdout-count',type=int,default=16)
    ap.add_argument('--steps',type=int,default=8,
                    help='Number of predictor optimizer updates, not image episodes')
    ap.add_argument('--episodes-per-step',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--strength',type=float,default=0.85)
    ap.add_argument('--rank',type=int,default=8)
    ap.add_argument('--basis-gain',type=float,default=1.0)
    ap.add_argument('--step-policy',choices=('legacy_floor','normalized'),default='normalized')
    ap.add_argument('--min-signal-norm',type=float,default=1e-8)
    ap.add_argument('--clean-weight',type=float,default=1.0,
       help='Relative weight; clean-weight=1 means equal clean/shifted gradient contribution')
    ap.add_argument('--teacher-weight',type=float,default=0.0)
    ap.add_argument('--background-weight',type=float,default=0.5,
       help='Extra BG hinge gradient weight in original detector foreground-normalized units')
    ap.add_argument('--scale-weight',type=float,default=0.25,
       help='Average scale-specific per-matched-positive hinge weight')
    ap.add_argument('--clean-total-weight',type=float,default=1.0)
    ap.add_argument('--background-margin',type=float,default=0.0,
       help='Allowed BG regression in detector loss units (NOT per-negative units)')
    ap.add_argument('--scale-margin',type=float,default=0.0,
       help='Allowed regression per matched positive, using scale-normalized loss')
    ap.add_argument('--clean-total-margin',type=float,default=0.0)
    ap.add_argument('--grad-clip',type=float,default=0.05,
       help='Predictor global gradient-norm cap; preclip norm is reported')
    ap.add_argument('--device',default='auto')
    ap.add_argument('--output',required=True)
    ap.add_argument('--save-sidecar',required=True)
    args=ap.parse_args()
    if args.steps<1 or args.steps>64 or args.episodes_per_step<1 or args.episodes_per_step>16:
        ap.error('Invalid steps/episode accumulation')
    if args.steps*args.episodes_per_step>args.train_count:
        ap.error('Training image count must be at least steps * episodes-per-step')
    if args.holdout_count<4 or args.train_count<1:
        ap.error('At least 4 holdout and 1 training image required')
    if max(args.lr,args.clean_weight,args.teacher_weight,args.strength,args.grad_clip)>10000:
        ap.error('Implausibly large parameter')
    for v,name in ((args.lr,'lr'),(args.clean_weight,'clean-weight'),
                   (args.strength,'strength'),(args.grad_clip,'grad-clip')):
        if not math.isfinite(v) or v<=0:ap.error('Invalid '+name)
    if args.strength>1 or args.lr>1e-2:
        ap.error('Appearance strength must <=1 and LR <=1e-2')
    if not math.isfinite(args.teacher_weight) or args.teacher_weight<0 or args.teacher_weight>1:
        ap.error('Teacher weight must be in [0,1]')
    guard_config=ConstraintConfig(
        background_weight=args.background_weight,scale_weight=args.scale_weight,
        clean_total_weight=args.clean_total_weight,
        background_margin=args.background_margin,scale_margin=args.scale_margin,
        clean_total_margin=args.clean_total_margin).validate()
    args.train_indices=list(range(args.train_start,args.train_start+args.train_count))
    args.holdout_indices=list(range(args.holdout_start,args.holdout_start+args.holdout_count))
    ids=args.train_indices+args.holdout_indices
    if min(ids)<0 or len(set(ids))!=len(ids):ap.error('Indices must be unique and nonnegative')
    out=Path(args.output).resolve();side=Path(args.save_sidecar).resolve()
    if out==side or out.exists() or side.exists():
        raise FileExistsError('Refusing overwrite or conflicting output names')
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Source CLEAN TRAIN only; nonclean annotation rejected')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,ck=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    dataset=exp.get_dataset(cache=False)
    if max(ids)>=len(dataset):raise IndexError('Index outside clean train dataset')
    native_hash=record_model_hash(model)
    x,_=load_source_example(dataset,args.train_indices[0],device,args.seed)
    channels=get_size_from_native(model,x)
    torch.manual_seed(args.seed)
    safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
                        min_signal_norm=args.min_signal_norm,basis_gain=args.basis_gain).to(device).train()
    init_pred={n:p.detach().cpu().clone() for n,p in safe.named_parameters() if n.startswith('predictor.')}
    selected,_=select_predictor_only(safe)
    snap=exact_frozen_snapshot(safe)
    config=_expected_config(args,native_hash,guard_config)
    before=evaluate_views(model,safe,dataset,args.holdout_indices,device,args.seed,args.strength)
    optimizer=torch.optim.AdamW([p for _,p in selected],lr=args.lr,weight_decay=0.)
    trace=[]
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        episodes=[]
        for offset in range(args.episodes_per_step):
            index=args.train_indices[step*args.episodes_per_step+offset]
            xi,yi=load_source_example(dataset,index,device,args.seed)
            spec=sample_appearance(args.seed,index,strength=args.strength)
            xa=apply_appearance(xi,spec)
            report=paired_accumulate_one(model,safe,xi,xa,yi,
                constraint_config=guard_config,clean_weight=args.clean_weight,
                teacher_weight=args.teacher_weight,accumulation_count=args.episodes_per_step)
            episodes.append({'index':index,'GT':int((yi.sum(-1)>0).sum()),
                'clean_sha256':image_fingerprint(xi),
                'shifted_sha256':image_fingerprint(xa),
                'appearance':spec_dict(spec),'metrics':report})
        g=optimizer_step_predictor(safe,optimizer,max_grad_norm=args.grad_clip)
        check_frozen(safe,snap)
        trace.append({'step':step+1,'episodes':episodes,**g})
        print('STEP',step+1,'IMAGES',[e['index'] for e in episodes],
             'PRED_META_GRAD',f'{g["predictor_gradient_l2"]:.6g}',
             'SHIFT_MEAN',f'{statistics.mean(e["metrics"]["shifted"]["adapted_minus_native"] for e in episodes):+.6g}',
             'BG_VIOLATIONS',sum(e['metrics']['shifted']['nonregression']['background_guard_active'] for e in episodes),
             flush=True)
    if record_model_hash(model)!=native_hash:
        raise RuntimeError('Native YOLOX mutated during training')
    move=predictor_delta(safe,init_pred)
    if move<=0:raise RuntimeError('No predictor learning detected')
    _atomic_save_sidecar(side,safe,config,trace,before,move)
    print('SAVED_SIDECAR',side,flush=True)
    after=evaluate_views(model,safe,dataset,args.holdout_indices,device,args.seed,args.strength)
    check_frozen(safe,snap)
    if record_model_hash(model)!=native_hash:
        raise RuntimeError('Native YOLOX mutated during evaluation')
    summary=summarize_views(before,after,constraint_config=guard_config)
    payload={'protocol':config,'source_epoch':ck.get('start_epoch'),
             'initial_holdout':before,'trained_holdout':after,'summary':summary,
             'steps':trace,'predictor_parameter_delta_l2':move,
             'limitations':['Source training labels used only in outer meta objective',
                            'No claim of AP/AR, adverse generalization or test-time safety',
                            'Numerical variation of tiny loss differences requires follow-up',
                            'No learned reliability gate']}
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('No overwrite')
    out.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    for view in ('clean','shifted'):
        q=summary[view]
        print(view.upper(),'mean_initial',f'{q["mean_initial_delta"]:+.7g}',
              'mean_trained',f'{q["mean_trained_delta"]:+.7g}',
              'paired',f'{q["mean_paired_delta"]:+.7g}',
              'BETTER/WORSE',q['better_than_initial'],q['worse_than_initial'],
              'BG_VIOLATIONS',q['background_violations_detector_units'],
              'STRICT',q['trained_strict'],'/',args.holdout_count,flush=True)
    print('SAVED_REPORT',out)
    print('CAUTION: source-only fixed-target surrogate; no AP/AR or unseen-domain claims')

if __name__=='__main__':main()
