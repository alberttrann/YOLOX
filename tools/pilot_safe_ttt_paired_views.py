#!/usr/bin/env python3
"""vNext.2 Safe TTT Phase B: paired-view, predictor-only source meta-training.

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

SCALES=('small','medium','large')
OBJECTIVES=('detector_total','scale_balanced')


def meta_objective(native, adapted, *, objective_mode, nonregression_weight,
                   background_weight=0.1):
    """Matched-target objective; per-positive scales + background kept distinct."""
    if objective_mode not in OBJECTIVES:
        raise ValueError('Unsupported objective mode')
    zero=adapted['total']*0.0
    sc_losses={}
    scale_before={}
    scale_after={}
    for scale in SCALES:
        count=native['counts'][scale]
        if count <= 0:continue
        factor=native['num_foreground_normalizer']/max(count,1)
        before=sum(native['scales'][scale].values()).detach()*factor
        after=sum(adapted['scales'][scale].values())*factor
        scale_before[scale]=before
        scale_after[scale]=after
        sc_losses[scale]=F.relu(after-before)
    per_scale=sum(sc_losses.values(),zero)/max(1,len(sc_losses))
    bg_factor=native['num_foreground_normalizer']/max(1,native['counts']['background'])
    bg_before=native['components']['background_objectness'].detach()*bg_factor
    bg_after=adapted['components']['background_objectness']*bg_factor
    bg_nonreg=F.relu(bg_after-bg_before)
    if objective_mode=='scale_balanced':
        positive=(sum(scale_after.values(),zero)/len(scale_after) if scale_after else zero)
        det_obj=positive+background_weight*bg_after
    else:
        det_obj=adapted['total']
    objective=det_obj+nonregression_weight*(per_scale+background_weight*bg_nonreg)
    return objective, {'per_scale':{k:float(v.detach()) for k,v in sc_losses.items()},
                      'background_violation':float(bg_nonreg.detach()),
                      'scale_count':{k:int(native['counts'][k]) for k in SCALES},
                      'background_count':int(native['counts']['background']),
                      'detector_primary':float(det_obj.detach())}


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


def _view_meta_backward(model,safe,images,labels,*,view_weight,objective_mode,
                        nonregression_weight,teacher_logits=None,teacher_weight=0.0):
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
            loss,details=meta_objective(native,adapted,
               objective_mode=objective_mode,nonregression_weight=nonregression_weight)
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


def paired_meta_step(model,safe,clean,shifted,labels,optimizer,*,
                     objective_mode='detector_total',nonregression_weight=0.5,
                     clean_weight=0.5,teacher_weight=0.0,max_grad_norm=5.0):
    """One paired episode; TWO outer task views, ONE predictor optimizer step.

    The clean-view loss explicitly protects clean performance. The shifted-view
    loss teaches adaptation on photometric variation with exactly unchanged GT
    geometry. Mixed derivatives are retained in both, but graphs are backprop'd
    sequentially to bound VRAM. The SSL INNER step remains unlabeled and uses
    only the image currently being adapted.
    """
    if clean.shape!=shifted.shape or clean.shape[0]!=1 or labels.shape[0]!=1:
        raise RuntimeError('Phase B requires geometry-aligned B=1 clean/shifted pair')
    if torch.equal(clean,shifted):
        raise RuntimeError('Appearance branch is numerically identical to clean input')
    if not (0.0 <= clean_weight <= 1.0) or not math.isfinite(clean_weight):
        raise ValueError('clean weight must be finite in [0,1]')
    if teacher_weight and clean_weight==0:
        raise ValueError('Teacher consistency requires the reference clean-view branch')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Native detector is not frozen')
    if model.training or model.backbone.training or not model.head.training:
        raise RuntimeError('Native setup must use eval() with only head.root.training=True')
    opt_params=[p for p in safe.predictor.parameters() if p.requires_grad]
    if not opt_params:
        raise RuntimeError('No trainable SSL predictor parameters')
    optimizer.zero_grad(set_to_none=True)
    clean_report=None;teacher_logits=None
    if clean_weight:
        clean_report,teacher_logits=_view_meta_backward(model,safe,clean,labels,
                        view_weight=clean_weight,objective_mode=objective_mode,
                        nonregression_weight=nonregression_weight)
    shifted_report,_=_view_meta_backward(model,safe,shifted,labels,
                        view_weight=1.0,objective_mode=objective_mode,
                        nonregression_weight=nonregression_weight,
                        teacher_logits=teacher_logits,teacher_weight=teacher_weight)
    prednorm=math.sqrt(sum(float(p.grad.detach().float().square().sum())
                           for p in opt_params if p.grad is not None))
    if not math.isfinite(prednorm) or prednorm<=0:
        raise RuntimeError('No finite detection-to-SSL predictor meta-gradient')
    if any(p.grad is not None for name,p in safe.named_parameters()
           if not name.startswith('predictor.')):
        raise RuntimeError('Frozen sidecar parameters received gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(opt_params,max_grad_norm,error_if_nonfinite=True))
    optimizer.step()
    return {**shifted_report,
            'shifted':shifted_report,'clean':clean_report,
            'clean_outer_weight':clean_weight,
            'predictor_gradient_l2':prednorm,'preclip_gradient_norm':norm}

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


def summarize_views(before,after,*,abs_threshold=2e-5,relative_threshold=1e-5):
    if len(before)!=len(after) or not before:
        raise RuntimeError('Unmatched source holdouts')
    summaries={}
    for view in ('clean','shifted'):
        rec=[]
        for pre,post in zip(before,after):
            if pre['index']!=post['index'] or pre['clean_sha256']!=post['clean_sha256'] or pre['shifted_sha256']!=post['shifted_sha256']:
                raise RuntimeError('Holdout changed images, ordering or appearance specifications')
            a=pre[view];b=post[view]
            if abs(a['native']-b['native'])>2e-5:
                raise RuntimeError('Native benchmark mutated or has an unstable forward pass')
            d=b['delta']-a['delta']
            tol=max(abs_threshold,relative_threshold*abs(a['native']))
            scale_regressions=[]
            for scale,entry in b['scales'].items():
                s=entry['adapted_minus_native_per_positive']
                smargin=max(abs_threshold,relative_threshold*abs(entry['native_per_matched_positive']))
                if s>smargin:scale_regressions.append(scale)
            bg=b['background']['delta_per_background_anchor']
            bg_tol=abs_threshold
            strict=(b['delta'] < -tol and not scale_regressions and bg<=bg_tol)
            rec.append({'index':int(pre['index']),
                'initial_delta':a['delta'],'trained_delta':b['delta'],'paired_delta':d,
                'tolerance':tol,'better_than_initial':d<-tol,'worse_than_initial':d>tol,
                'adaptation_helpful':b['delta']<-tol,
                'adaptation_harmful':b['delta']>tol,
                'strict_all_present_scales_and_bg':strict,
                'scale_regressions':scale_regressions,
                'background_regression':bool(bg>bg_tol)})
        summaries[view]={'mean_initial_delta':statistics.mean(r['initial_delta'] for r in rec),
            'mean_trained_delta':statistics.mean(r['trained_delta'] for r in rec),
            'mean_paired_delta':statistics.mean(r['paired_delta'] for r in rec),
            'better_than_initial':sum(r['better_than_initial'] for r in rec),
            'worse_than_initial':sum(r['worse_than_initial'] for r in rec),
            'trained_helpful':sum(r['adaptation_helpful'] for r in rec),
            'trained_harmful':sum(r['adaptation_harmful'] for r in rec),
            'trained_strict':sum(r['strict_all_present_scales_and_bg'] for r in rec),
            'scale_coverage':{s:sum(s in r0[view]['scales'] for r0 in after) for s in SCALES},
            'per_image':rec}
    return summaries


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


def _expected_config(args,model_hash):
    return {'checkpoint':str(Path(args.checkpoint).resolve()),
      'model_exp':args.model_exp,'native_sha256':model_hash,'weights':args.weights,
      'seed':args.seed,'rank':args.rank,'basis_gain':args.basis_gain,
      'step_policy':args.step_policy,'min_signal_norm':args.min_signal_norm,
      'train_indices':args.train_indices,'holdout_indices':args.holdout_indices,
      'clean_train_only':True,
      'steps':args.steps,'lr':args.lr,'strength':args.strength,
      'objective_mode':args.objective_mode,'nonregression_weight':args.nonregression_weight,
      'teacher_weight':args.teacher_weight,'clean_weight':args.clean_weight,'trainable_families':['predictor'],
      'frozen_families':['adapter','mask_token','step_logit'],
      'inner_test_time_labels':False,'source_train_labels_used_only_in_outer':True,
      'identical_bbox_geometry_across_views':True,'test_time_teacher_required':False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    ap.add_argument('--weights',choices=('ema','raw'),default='ema')
    ap.add_argument('--train-indices',nargs='+',type=int,default=None)
    ap.add_argument('--holdout-indices',nargs='+',type=int,default=None)
    ap.add_argument('--train-start',type=int,default=300)
    ap.add_argument('--train-count',type=int,default=32)
    ap.add_argument('--holdout-start',type=int,default=500)
    ap.add_argument('--holdout-count',type=int,default=16)
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--strength',type=float,default=0.85)
    ap.add_argument('--rank',type=int,default=8)
    ap.add_argument('--basis-gain',type=float,default=1.0)
    ap.add_argument('--step-policy',choices=('legacy_floor','normalized'),default='normalized')
    ap.add_argument('--min-signal-norm',type=float,default=1e-8)
    ap.add_argument('--objective-mode',choices=OBJECTIVES,default='detector_total')
    ap.add_argument('--nonregression-weight',type=float,default=0.5)
    ap.add_argument('--clean-weight',type=float,default=0.5,
         help='Weight on a second, clean-view supervised meta-loss; 0 disables this branch')
    ap.add_argument('--teacher-weight',type=float,default=0.0,
         help='Optional source clean prediction consistency; DEFAULT ZERO pending teacher quality validation')
    ap.add_argument('--device',default='auto')
    ap.add_argument('--output',required=True)
    ap.add_argument('--save-sidecar',required=True)
    ap.add_argument('--evaluate-only-sidecar',default=None,
                    help='Regenerate report from an already saved Phase B checkpoint without retraining')
    args=ap.parse_args()
    args.train_indices=(list(range(args.train_start,args.train_start+args.train_count))
                        if args.train_indices is None else args.train_indices)
    args.holdout_indices=(list(range(args.holdout_start,args.holdout_start+args.holdout_count))
                        if args.holdout_indices is None else args.holdout_indices)
    all_indices=args.train_indices+args.holdout_indices
    if args.steps<1 or args.steps>64 or args.steps>len(args.train_indices):
        ap.error('Use 1..64 steps, each on a distinct source-train image')
    if not args.train_indices or len(args.holdout_indices)<4:
        ap.error('Need nonempty train and at least 4 disjoint source holdout images')
    if min(all_indices)<0 or len(set(all_indices))!=len(all_indices):
        ap.error('Train/holdout source indices must be nonnegative and unique')
    for value,name,minimum,maximum in ((args.lr,'lr',0,0.01),(args.nonregression_weight,'nonregression weight',0,10),
                                      (args.teacher_weight,'teacher weight',0,1),(args.clean_weight,'clean view weight',0,1),(args.strength,'strength',0,1)):
        if not math.isfinite(value) or not (minimum <= value <= maximum):
            ap.error('Invalid '+name)
    if args.lr<=0 or args.strength<=0:ap.error('LR and appearance strength must be positive')
    if args.teacher_weight and not args.clean_weight:ap.error('Teacher requires a clean-view branch')
    out=Path(args.output).resolve();side=Path(args.save_sidecar).resolve()
    if out==side or out.exists():raise FileExistsError('Refusing existing report or same-name sidecar')
    if side.exists() and args.evaluate_only_sidecar is None:
        raise FileExistsError('Refusing to overwrite existing sidecar')
    if args.evaluate_only_sidecar is not None and side!=Path(args.evaluate_only_sidecar).resolve():
        ap.error('--save-sidecar must point to the existing --evaluate-only-sidecar path')
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Source CLEAN TRAIN only. Refusing nonclean training split.')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,ck=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    dataset=exp.get_dataset(cache=False)
    if max(all_indices)>=len(dataset):raise IndexError('Source index outside dataset length')
    model_hash=record_model_hash(model)
    x,_=load_source_example(dataset,args.train_indices[0],device,args.seed)
    channels=get_size_from_native(model,x)
    torch.manual_seed(args.seed)
    safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
                        min_signal_norm=args.min_signal_norm,basis_gain=args.basis_gain).to(device).train()
    initial_pred={name:p.detach().cpu().clone() for name,p in safe.named_parameters() if name.startswith('predictor.')}
    selected,frozen=select_predictor_only(safe)
    snap=exact_frozen_snapshot(safe)
    config=_expected_config(args,model_hash)
    if args.evaluate_only_sidecar is not None:
        payload=torch.load(side,map_location='cpu',weights_only=False)
        if not isinstance(payload,dict) or payload.get('configuration')!=config:
            raise RuntimeError('Recovered sidecar configuration/hash/indices disagree: refusing report')
        safe.load_state_dict(payload['safe_ttt'],strict=True)
        check_frozen(safe,snap)
        before=payload['before_holdout'];trace=payload['steps']
        movement=predictor_delta(safe,initial_pred)
        if len(trace)!=args.steps or abs(movement-payload['predictor_parameter_delta_l2'])>1e-7:
            raise RuntimeError('Recovered sidecar steps or predictor delta mismatch')
        print('RECOVERED Phase B sidecar with NO TRAINING',side,flush=True)
    else:
        before=evaluate_views(model,safe,dataset,args.holdout_indices,device,args.seed,args.strength)
        optimizer=torch.optim.AdamW([p for _,p in selected],lr=args.lr,weight_decay=0.)
        trace=[]
        for step,index in enumerate(args.train_indices[:args.steps],start=1):
            xi,yi=load_source_example(dataset,index,device,args.seed)
            spec=sample_appearance(args.seed,index,strength=args.strength)
            xa=apply_appearance(xi,spec)
            item=paired_meta_step(model,safe,xi,xa,yi,optimizer,
                        objective_mode=args.objective_mode,
                        nonregression_weight=args.nonregression_weight,
                        clean_weight=args.clean_weight,teacher_weight=args.teacher_weight)
            check_frozen(safe,snap)
            item.update({'step':step,'index':index,'appearance':spec_dict(spec),
                         'train_clean_sha256':image_fingerprint(xi),
                         'train_shifted_sha256':image_fingerprint(xa)})
            trace.append(item)
            print('STEP',step,'IDX',index,'GT',int((yi.sum(-1)>0).sum()),
                  'PRED_META_GRAD',f'{item["predictor_gradient_l2"]:.5g}',
                  'CLEAN_DELTA',('disabled' if item['clean'] is None else f"{item['clean']['adapted_minus_native']:+.6g}"),
                  'SHIFT_DELTA',f'{item["adapted_minus_native"]:+.6g}',flush=True)
        if record_model_hash(model)!=model_hash:raise RuntimeError('Native YOLOX mutated during Phase B')
        movement=predictor_delta(safe,initial_pred)
        if not movement>0:raise RuntimeError('No predictor learning detected')
        _atomic_save_sidecar(side,safe,config,trace,before,movement)
        print('SAVED_SIDECAR_EARLY',side,flush=True)
    after=evaluate_views(model,safe,dataset,args.holdout_indices,device,args.seed,args.strength)
    check_frozen(safe,snap)
    if record_model_hash(model)!=model_hash:raise RuntimeError('Native YOLOX mutated during evaluation')
    summary=summarize_views(before,after)
    report={'protocol':config,'source_epoch':ck.get('start_epoch'),
        'steps':trace,'initial_holdout':before,'trained_holdout':after,
        'summary':summary,'predictor_parameter_delta_l2':movement,
        'frozen_state_changed_keys':[],'native_state_changed_keys':[],
        'limitations':['ONLY source-train indices, no clean validation AP/AR',
                       'small fixed-matching losses may have FP32 quantization effects',
                       'synthetic appearance changes do not prove adverse-DG performance',
                       'no learned safe abstention',
                       'teacher option remains unvalidated and is off by default']}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,allow_nan=False,indent=2),encoding='utf8')
    for v in ('clean','shifted'):
        s=summary[v]
        print(v.upper(),'mean initial',f'{s["mean_initial_delta"]:+.6g}',
              'trained',f'{s["mean_trained_delta"]:+.6g}',
              'paired',f'{s["mean_paired_delta"]:+.6g}',
              'BETTER/WORSE',s['better_than_initial'],s['worse_than_initial'],
              'STRICT',s['trained_strict'],'/',len(s['per_image']),flush=True)
    print('SAVED',out)
    print('NO CLAIM: clean-source pilot cannot prove AP/AR, domain generalization or safety')


if __name__=='__main__':main()
