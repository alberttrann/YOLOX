"""HARD-CAPPED, CLEAN-SOURCE vNext.3 PG-SMA pilot; intentionally bypasses 80-epoch trainer.

Single GPU, single process, at most 2048 optimizer updates. This is *not* a
reimplementation of the full YOLOX scheduler/curriculum/EMA/late-epoch training.
Its clean-val COCO scores are subset diagnostics, never official full-set AP.

Run as: python -m tools.pgsma_bounded_pilot --variant native_scpa --steps 200 ...
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import os
import json
import math
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from ._pgsma_pilot_core import (validate_budget,configure_variant,
  model_parameter_split,finite_scalar,atomic_json,file_digest,group_scale,
  native_recipe,expected_warmstart_source)
from ._pgsma_pilot_eval import run_eval

VARIANTS=('native_simota','native_scpa','native_scpa_instance','memory','pgsma',
          'native_simota_continue','memory_simota','pgsma_simota')


def safe_checkpoint_state(path):
    # Trust only user-generated checkpoint files. No pickle fallback.
    try:
        checkpoint=torch.load(str(path),map_location='cpu',weights_only=True)
    except Exception as e:
        raise RuntimeError('Checkpoint safe loading failed; do not use an untrusted pickle: '+str(e)) from e
    if not isinstance(checkpoint,dict):
        raise RuntimeError('Checkpoint object is not a dictionary')
    if isinstance(checkpoint.get('model'),dict):
        state=checkpoint['model']
    elif isinstance(checkpoint.get('ema_model'),dict):
        state=checkpoint['ema_model']
    elif checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state=checkpoint
    else: raise ValueError('No model state_dict found in checkpoint')
    clean={}
    for k,value in state.items():
        k=str(k)
        while k.startswith('module.') or k.startswith('_orig_mod.'):
            k=k[7:] if k.startswith('module.') else k[10:]
        if k in clean:raise RuntimeError('State key collision: '+k)
        clean[k]=value
    return checkpoint,clean


def load_weights(model,path,pilot_checkpoint=False):
    checkpoint,weights=safe_checkpoint_state(path)
    target=model.state_dict()
    good={k:v for k,v in weights.items() if k in target and tuple(v.shape)==tuple(target[k].shape)}
    mismatches=[k for k in weights if k in target and tuple(weights[k].shape)!=tuple(target[k].shape)]
    if pilot_checkpoint:
        metadata=checkpoint.get('pgsma_pilot')
        if not isinstance(metadata,dict) or metadata.get('variant') not in ('native_simota','native_scpa','native_scpa_instance'):
            raise RuntimeError('Semantic pilot requires a checkpoint from a successful bounded NATIVE pilot')
        if int(metadata.get('successful_native_updates',0))<=0:
            raise RuntimeError('Native warm-start checkpoint has no successful training updates')
        # A native control checkpoint has NOT trained a semantic gate. A new
        # gate architecture is expected to be initialized independently, while
        # EVERY native tensor must still load with exactly matching geometry.
        # Never silently accept an incomplete or incompatible native warmstart.
        semantic_prefix='head.pgsma_engine.'
        native_keys={k for k in target if not k.startswith(semantic_prefix)}
        missing_native=native_keys-set(good)
        mismatch_native=[k for k in mismatches if not k.startswith(semantic_prefix)]
        unexpected_native=[k for k in weights if not k.startswith(semantic_prefix) and k not in target]
        if missing_native or mismatch_native or unexpected_native:
            raise RuntimeError('Native warm-start mismatch: missing='+repr(sorted(missing_native)[:8])+
                ' size='+repr(mismatch_native[:8])+' unexpected='+repr(unexpected_native[:8]))
        # Do NOT initialize the new gate/predictor/prototypes from a historical
        # native pilot's random, never-trained semantic state.
        good={k:v for k,v in good.items() if not k.startswith(semantic_prefix)}
    else:
        # Original COCO checkpoint contains 80-class heads and additional new vNext3 weights.
        # Check coverage of SOURCE pretrained tensors, not new target tensors.
        for prefix in ('backbone.backbone.stem.','backbone.backbone.dark2.',
                       'backbone.backbone.dark3.','backbone.backbone.dark4.',
                       'backbone.backbone.dark5.'):
            orig={k:v for k,v in weights.items() if k.startswith(prefix)}
            if not orig:
                raise RuntimeError('No pretrained source tensors for '+prefix)
            elements=sum(t.numel() for t in orig.values())
            matched=sum(t.numel() for k,t in orig.items() if k in good)
            if matched / max(elements,1)<0.99:
                raise RuntimeError(f'Pretrained source compatibility too low for {prefix}: {matched}/{elements}')
        bad=[k for k in mismatches if not k.startswith('head.cls_preds.')]
        if bad:raise RuntimeError('Unexpected pretrained tensor size mismatches: '+repr(bad[:12]))
    model.load_state_dict(good,strict=False)
    return {'checkpoint':str(path),'sha256':file_digest(path),'matched_tensors':len(good),
            'shape_mismatches':mismatches,'is_pilot_warmstart':bool(pilot_checkpoint)}


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False

def capture_rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
    if state[3] is not None:torch.cuda.set_rng_state_all(state[3])


def state_digest(model, semantic_prefix='head.pgsma_engine.'):
    """Bitwise digests for native model weights AND persistent buffers."""
    groups={'backbone':hashlib.sha256(), 'head':hashlib.sha256(),
            'other':hashlib.sha256(), 'all':hashlib.sha256()}
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith(semantic_prefix):
            continue
        part='backbone' if name.startswith('backbone.') else 'head' if name.startswith('head.') else 'other'
        value=tensor.detach().cpu().contiguous()
        raw=name.encode('utf8')+b'\x00'+str(value.dtype).encode('ascii')+b'\x00'+str(tuple(value.shape)).encode('ascii')+b'\x00'+value.numpy().tobytes()
        for key in (part,'all'):
            groups[key].update(raw)
    return {key:v.hexdigest() for key,v in groups.items()}


def grad_digest(model, native_params):
    """Fingerprint native gradients; call before and after clipping."""
    names={id(p):name for name,p in model.named_parameters()}
    digest=hashlib.sha256()
    for param in sorted(native_params,key=lambda p:names[id(p)]):
        name=names[id(param)]
        digest.update(name.encode('utf8')+b'\x00')
        if param.grad is None:
            digest.update(b'NO_GRAD')
        else:
            grad=param.grad.detach().cpu().contiguous()
            digest.update(grad.numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(model, optimizer):
    """Fingerprint momentum/state to distinguish optimizer drift from model drift."""
    names={id(p):name for name,p in model.named_parameters()}
    digest=hashlib.sha256()
    for group in optimizer.param_groups:
        for param in group['params']:
            digest.update(names[id(param)].encode('utf8')+b'\x00')
            for key,value in sorted(optimizer.state[param].items()):
                digest.update(str(key).encode('utf8')+b'\x00')
                if torch.is_tensor(value):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode('utf8'))
    return digest.hexdigest()


def rng_digest(state):
    h=hashlib.sha256()
    h.update(repr(state[0]).encode('utf8'))
    h.update(repr(state[1]).encode('utf8'))
    h.update(state[2].cpu().numpy().tobytes())
    if state[3] is not None:
        for item in state[3]:
            h.update(item.cpu().numpy().tobytes())
    return h.hexdigest()


def append_parity_row(path, payload):
    with open(path,'a',encoding='utf8') as output:
        output.write(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n')


def parity_capture_steps(steps):
    return {step for step in (0,1,2,4,8,16,32,64,128,256,384,512,768,1024,1536,2048)
            if step<=steps}


def make_scaler(enabled):
    # torch 2.0.1 has no torch.amp.GradScaler; use stable CUDA legacy path.
    return torch.cuda.amp.GradScaler(enabled=enabled,init_scale=1024.0,growth_interval=2000)


def semantic_utility_probe(model,seed,mode):
    """Same perturbed feature snapshot; candidate/gate FG and BG utility separately."""
    if mode not in ('memory','pgsma'):return None
    cache=model.head.pgsma_cache
    if cache is None or cache.get('match') is None:return None
    sem=model.semantic
    with torch.no_grad():
        feats=cache['features'];matches=cache['match']
        pert=sem._perturb(feats,seed)
        baselines=[];corrections=[];gates=[]
        for i,feat in enumerate(pert):
            layer=model.head.cls_preds[i]
            base=F.conv2d(feat.float(),layer.weight.detach().float(),
                           layer.bias.detach().float() if layer.bias is not None else None)
            d=sem._candidate(i,feat,base,cache['native_obj'][i],mode=mode,freeze_reference=True)
            baselines.append(base);corrections.append(d['delta']);gates.append(d['gate'])
        flat=sem._flatten(baselines)
        delta=sem._flatten(corrections)
        gate=sem._flatten(gates)
        fg=matches['fg_ids'];lab=matches['fg_labels']
        quality=matches['fg_quality'];uid=matches['gt_uid'];bg=matches['bg_ids']
        lp=sem._sample_losses
        nf,nb=lp(flat,fg,lab,quality,uid,bg,False)
        cf,cb=lp(flat+delta,fg,lab,quality,uid,bg,False)
        gf,gb=lp(flat+delta*gate,fg,lab,quality,uid,bg,False)
        return {k:finite_scalar(v,k) for k,v in {
            'candidate_fg_improvement':nf-cf,
            'candidate_bg_improvement':nb-cb,
            'candidate_total_improvement':(nf+nb)-(cf+cb),
            'gate_fg_improvement':nf-gf,
            'gate_bg_improvement':nb-gb,
            'gate_total_improvement':(nf+nb)-(gf+gb),
            'mean_abs_candidate_delta':delta.abs().mean(),
            'mean_gate_acceptance':gate.mean(),
            'max_abs_candidate_delta':delta.abs().max(),
            'mean_baseline_foreground_loss':nf,
            'mean_baseline_background_loss':nb
        }.items()}


def level_positive_counts(model):
    cache=model.head.pgsma_cache
    if cache is None or cache.get('match') is None:return {}
    ids=cache['match']['fg_ids']
    # The native flatten layout is [B,concat(levels)], not [concat(B*P3,B*P4,B*P5)].
    # Use per-image strides to calculate correct level offsets.
    spatial=[int(f.shape[-2]*f.shape[-1]) for f in cache['features']]
    stride_offsets=[0,spatial[0],spatial[0]+spatial[1],sum(spatial)]
    total=sum(spatial)
    ids_mod=ids%total
    return {f'P{i+3}':int(((ids_mod>=stride_offsets[i]) & (ids_mod<stride_offsets[i+1])).sum())
            for i in range(3)}


def verify_batch(images,targets,exp):
    if images.ndim!=4 or images.shape[1]!=4:
        raise RuntimeError('Expected RGB+validity channel training images')
    if targets.ndim!=3 or targets.shape[-1]!=5:
        raise RuntimeError('Expected YOLOX [B,L,5] training targets')
    if images.shape[0]!=targets.shape[0]:
        raise RuntimeError('Batch and target count differ')
    valid=torch.isfinite(targets).all(-1)&(targets[...,3]>0)&(targets[...,4]>0)
    cls=targets[...,0][valid]
    if cls.numel() and ((cls<0).any() or (cls>=exp.num_classes).any() or (cls!=cls.round()).any()):
        raise RuntimeError('Invalid classes from training loader')
    if not torch.isfinite(images).all():raise FloatingPointError('Nonfinite training input')
    return int(valid.sum())


def save_pilot_checkpoint(path,model,native_optimizer,semantic_optimizer,scaler,meta):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':model.state_dict(),'native_optimizer':native_optimizer.state_dict(),
             'semantic_optimizer':semantic_optimizer.state_dict() if semantic_optimizer else None,
             'scaler':scaler.state_dict(),'pgsma_pilot':meta}
    tmp=path.with_name(path.name+'.writing')
    torch.save(payload,tmp);tmp.replace(path)


def run(args):
    validate_budget(args.steps,args.eval_images,args.batch_size,args.eval_every)
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or int(os.environ.get('RANK','0'))!=0:
        raise RuntimeError('Bounded pilot is strictly single-process; DDP must not be active')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.amp and not args.device.startswith('cuda'):
        raise ValueError('AMP allowed only on CUDA in this pilot')
    if args.variant in ('memory','pgsma','memory_simota','pgsma_simota','native_simota_continue') and not args.warmstart:
        raise RuntimeError('Semantic pilot requires --warmstart <native pilot checkpoint>')
    if args.variant not in ('memory','pgsma','native_scpa','native_simota_continue','memory_simota','pgsma_simota') and args.warmstart:
        raise RuntimeError('Warm-start unsupported for this pilot variant')
    if args.seed<0 or args.seed>2147483647:raise ValueError('Seed must be in [0,2^31-1]')
    if args.strict_determinism:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in PowerShell BEFORE launching Python')
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'Refusing nonempty output directory {out}; use a new run name')
    out.mkdir(parents=True,exist_ok=True)
    seed_everything(args.seed)
    from exps.default.tde_yolox_s_pgsma import Exp
    from yolox.core.pgsma_step import semantic_optimizer_step
    exp=Exp()
    if exp.allow_full_training:
        raise RuntimeError('Expected full training guard to remain False; pilot refuses modified experiment')
    exp.data_num_workers=0
    exp.ema=False # This bounded pilot deliberately does not use full-trainer EMA.
    exp.scpa=(native_recipe(args.variant)!='simota_per_positive')
    exp.per_gt_normalization=(args.variant=='native_scpa_instance')
    exp.pgsma_variant='memory' if args.variant in ('memory','memory_simota') else 'pgsma' if args.variant in ('pgsma','pgsma_simota') else 'native'
    model=exp.get_model()
    if args.candidate_objective != 'v2' and args.variant != 'pgsma_simota':
        raise RuntimeError('Experimental candidate objectives apply only to pgsma_simota bounded pilots')
    model.semantic.candidate_objective = args.candidate_objective
    inference_mode=configure_variant(model,args.variant)
    checkpoint_info=load_weights(model,args.warmstart or args.checkpoint,
                                 pilot_checkpoint=bool(args.warmstart))
    if args.warmstart:
        checkpoint_meta,_=safe_checkpoint_state(args.warmstart)
        required=expected_warmstart_source(args.variant)
        actual=checkpoint_meta['pgsma_pilot']['variant']
        if required is None or actual!=required:
            raise RuntimeError('Wrong warm-start matcher: requested '+args.variant+
                               ' requires native '+str(required)+', got '+str(actual))
    device=torch.device(args.device)
    model=model.to(device)
    model.train()
    native_optimizer=exp.get_optimizer(args.batch_size)
    native_lr=float(args.native_lr) if args.native_lr else float(exp.basic_lr_per_img*args.batch_size)
    if native_lr<=0 or native_lr>1e-2:raise ValueError('native-lr outside pilot safety envelope (0,1e-2]')
    for g in native_optimizer.param_groups:g['lr']=native_lr
    semantic_optimizer=None
    if inference_mode!='native':
        # EMA teacher projectors are registered semantic parameters but frozen.
        # Only trainable semantic parameters belong in the optimizer: the
        # exact ownership contract checked by model_parameter_split below.
        trainable_semantic = [p for p in model.semantic_parameters() if p.requires_grad]
        if not trainable_semantic:
            raise RuntimeError('Semantic variant has no trainable semantic parameters')
        semantic_optimizer=torch.optim.AdamW(trainable_semantic,
                                             lr=float(args.semantic_lr),weight_decay=0.0)
    native_params=model_parameter_split(model,native_optimizer,semantic_optimizer)
    scaler=make_scaler(args.amp)
    # Seed AFTER model/checkpoint/optimizer construction for matched data-stream RNG.
    seed_everything(args.seed+101)
    train_loader=exp.get_data_loader(args.batch_size,False,no_aug=args.no_mosaic,cache_img=False)
    batches=iter(train_loader)
    clean_loader=exp.get_eval_loader(batch_size=1,is_distributed=False,testdev=False)
    nval=len(clean_loader.dataset)
    rng=np.random.RandomState(args.seed+444)
    eval_indices=sorted(rng.choice(nval,size=min(args.eval_images,nval),replace=False).tolist())
    labels_file_digest=file_digest(exp.val_ann) if Path(exp.val_ann).is_file() else None
    run_info={'variant':args.variant,'steps_requested':args.steps,'seed':args.seed,
              'input_size':list(exp.input_size),'test_size':list(exp.test_size),
              'class_count':exp.num_classes,'native_lr':native_lr,
              'semantic_lr':args.semantic_lr if semantic_optimizer else None,
              'checkpoint':checkpoint_info,'warmstart':bool(args.warmstart),
              'native_recipe':native_recipe(args.variant),
              'semantic_recipe':'source_dualview_fg_guard_halo_utility_gate_v2',
              'candidate_objective':args.candidate_objective,
              'mosaic_enabled':not args.no_mosaic,'mosaic_prob':exp.mosaic_prob if not args.no_mosaic else 0.0,
              'eval_images':len(eval_indices),'eval_indices':eval_indices,
              'clean_val_json_sha256':labels_file_digest,
              'limitations':['single GPU, single-process only','no full 80-epoch schedule',
                 'fixed training input shape','no full-trainer EMA','pilot COCO metrics use subset images',
                 'semantic utility uses source-feature perturbations','no adverse-set evaluation']}
    atomic_json(out/'manifest.json',run_info)
    # Validation loader creation and Subset iteration may draw RNG state even
    # with num_workers=0. Preserve training random stream exactly.
    state=capture_rng()
    try:
        initial={}
        modes=('native',inference_mode) if inference_mode!='native' else ('native',)
        for m in dict.fromkeys(modes):
            initial[m]=run_eval(model,exp,eval_indices,device,m,args.conf,args.nms)
        atomic_json(out/'clean_val_step_00000.json',{'step':0,'metrics':initial,'variant':args.variant})
    except Exception as exc:
        atomic_json(out/'result.json',{**run_info,'status':'BLOCKED',
            'stage':'initial_clean_validation',
            'error':f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        restore_rng(state)
    logs=[];valid_updates=0;semantic_updates=0;overflow_count=0;first_digest=None;last_metrics={}
    trace_path=out/'native_parity_trace.jsonl'
    traced_steps=parity_capture_steps(args.steps) if args.trace_parity else set()
    if args.trace_parity:
        append_parity_row(trace_path,{'step':0,'phase':'post_initial_validation',
                         'model':state_digest(model),
                         'optimizer':optimizer_digest(model,native_optimizer),
                         'rng':rng_digest(capture_rng()),
                         'strict_determinism':bool(args.strict_determinism)})
    stream_digest=hashlib.sha256()
    import time
    started=time.monotonic()
    try:
        for step in range(1,args.steps+1):
            try: batch=next(batches)
            except StopIteration:
                batches=iter(train_loader);batch=next(batches)
            images,targets=batch[:2]
            gt_count=verify_batch(images,targets,exp)
            # Full-input fingerprint only on parity-traced steps. The ordinary
            # stream fingerprint samples RGB pixels and is NOT a proof that
            # every training input pixel is equal.
            full_input_digest=None
            if step in traced_steps:
                h=hashlib.sha256()
                for t in (images,targets):
                    v=t.detach().cpu().contiguous()
                    h.update(str(tuple(v.shape)).encode('ascii'))
                    h.update(str(v.dtype).encode('ascii'))
                    h.update(v.numpy().tobytes())
                full_input_digest=h.hexdigest()
            # Cheap reproducibility fingerprint from GTs and sparse RGB pixels.
            # Not a cryptographic proof of bitwise equality across full images.
            small=hashlib.sha256()
            small.update(images[:, :3, ::32, ::32].detach().cpu().numpy().tobytes())
            small.update(targets.detach().cpu().numpy().tobytes())
            stream_digest.update(small.digest())
            if first_digest is None:
                h=hashlib.sha256()
                for t in (images,targets):h.update(t.detach().cpu().numpy().tobytes())
                first_digest=h.hexdigest()
            images=images.float().to(device,non_blocking=False)
            targets=targets.float().to(device,non_blocking=False)
            images,targets=exp.preprocess(images,targets,exp.input_size)
            valid_mask=images[:,3:4]>0.5
            rgb=images[:,:3].contiguous()
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs=model(rgb,targets,valid_image_mask=valid_mask)
            loss=outputs['total_loss']
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Native forward produced nonfinite loss')
            correction_mode='memory' if args.variant in ('memory','memory_simota') else 'pgsma'
            sem_metrics=None;utility=None
            trace_row = {'step':step,'phase':'native_update',
                         'full_input_sha256':full_input_digest,
                         'native_loss':finite_scalar(loss.detach(),'native_loss'),
                         'iou_loss':finite_scalar(outputs['iou_loss'].detach(),'iou_loss'),
                         'cls_loss':finite_scalar(outputs['cls_loss'].detach(),'cls_loss'),
                         'conf_loss':finite_scalar(outputs['conf_loss'].detach(),'conf_loss'),
                         'native_pre_step':state_digest(model)} if step in traced_steps else None
            semantic_rng_before=None
            if semantic_optimizer is not None:
                # Diagnostic isolation: semantic work must not consume global
                # Python / NumPy / CPU Torch / CUDA Torch random streams.
                semantic_rng_before=capture_rng() if (args.isolate_semantic_rng or args.trace_parity) else None
                try:
                    with torch.cuda.amp.autocast(enabled=False):
                        success,sem_metrics=semantic_optimizer_step(model,semantic_optimizer,
                            seed=args.seed+1000003*step,correction_mode=correction_mode)
                    if not success:
                        raise FloatingPointError(f'Semantic optimizer failed at update {step}: {sem_metrics}')
                    semantic_updates+=1
                    if step%args.log_every==0 or step in (1,args.steps):
                        utility=semantic_utility_probe(model,args.seed+1000003*step,correction_mode)
                finally:
                    if semantic_rng_before is not None:
                        semantic_rng_after=capture_rng()
                        if trace_row is not None:
                            trace_row['semantic_rng_before']=rng_digest(semantic_rng_before)
                            trace_row['semantic_rng_after']=rng_digest(semantic_rng_after)
                            trace_row['semantic_rng_changed']=(
                                trace_row['semantic_rng_before']!=trace_row['semantic_rng_after'])
                        if args.isolate_semantic_rng:
                            restore_rng(semantic_rng_before)
                semantic_optimizer.zero_grad(set_to_none=True)
            native_optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(native_optimizer)
            if trace_row is not None:
                trace_row['native_grad_before_clip']=grad_digest(model,native_params)
            bad_native=[p for p in native_params if p.grad is not None and not torch.isfinite(p.grad).all()]
            grad_before=None
            if not bad_native:
                grad_before=finite_scalar(torch.nn.utils.clip_grad_norm_(native_params,10.0,error_if_nonfinite=True),'grad_norm')
            if trace_row is not None:
                trace_row['native_grad_after_clip']=grad_digest(model,native_params)
            before=scaler.get_scale()
            scaler.step(native_optimizer);scaler.update()
            if trace_row is not None:
                trace_row['native_post_step']=state_digest(model)
                trace_row['native_optimizer_post_step']=optimizer_digest(model,native_optimizer)
                trace_row['rng_after_step']=rng_digest(capture_rng())
                append_parity_row(trace_path,trace_row)
            after=scaler.get_scale()
            did_step=(not args.amp) or (after>=before)
            if bad_native and did_step:
                raise RuntimeError('Nonfinite native gradient but AMP scaler failed to skip update')
            if did_step:valid_updates+=1
            else:
                overflow_count+=1
                if overflow_count>=3:
                    raise FloatingPointError('Repeated AMP skip; pilot aborts instead of hiding instability')
            if step%args.log_every==0 or step in (1,args.steps):
                stats=model.head.last_assignment_stats.copy()
                matched_ids=model.head.pgsma_cache['match']['gt_uid']
                real_covered=int(torch.unique(matched_ids).numel())
                real_total=int(stats['num_gt_total'])
                if real_covered>real_total:
                    raise RuntimeError('Assignment GT identifiers exceed GT count')
                real_uncovered=real_total-real_covered
                if model.head.assignment=='scpa' and real_uncovered != int(stats['uncovered_gt']):
                    raise RuntimeError('SCPA coverage telemetry is internally inconsistent')
                stats['covered_gt']=real_covered
                stats['uncovered_gt']=real_uncovered
                stats['coverage_source']='derived_from_unique_gt_uid'
                quality=model.head.pgsma_cache['match']['fg_quality'].detach().float()
                stats['positive_iou_mean']=float(quality.mean()) if quality.numel() else None
                stats['positive_iou_lt_0p1']=int((quality<0.1).sum())
                stats['positive_iou_lt_0p3']=int((quality<0.3).sum())
                rec={'step':step,'native_loss':finite_scalar(loss.detach(),'native_loss'),
                     'native_grad_norm_before_clip':grad_before,'native_step_success':did_step,
                     'scaler_scale':float(after),'valid_gt':gt_count,'assignments':stats,
                     'pos_per_level':level_positive_counts(model),'semantic_step':sem_metrics,
                     'semantic_utility':utility}
                logs.append(rec)
                with open(out/'train_metrics.jsonl','a',encoding='utf8') as f:
                    f.write(json.dumps(rec,allow_nan=False,default=str)+'\n')
                last_metrics=rec
                print(f'[{args.variant}] step {step}/{args.steps} native={rec["native_loss"]:.4f} '
                      f'GT={stats.get("num_gt_total")} uncovered={stats.get("uncovered_gt")} '
                      f'native_step={did_step}',flush=True)
            del outputs, loss, rgb, images, targets,valid_mask
            if step%args.eval_every==0 or step==args.steps:
                val={}
                modes=('native',inference_mode) if inference_mode!='native' else ('native',)
                state=capture_rng()
                try:
                    for m in dict.fromkeys(modes):
                        val[m]=run_eval(model,exp,eval_indices,device,m,args.conf,args.nms)
                finally:
                    restore_rng(state)
                atomic_json(out/f'clean_val_step_{step:05d}.json',{'step':step,'variant':args.variant,
                                                   'metrics':val})
                model.train()
                print(f'[{args.variant}] clean-val subset evaluated at step {step}',flush=True)
        native_hash=hashlib.sha256()
        for name,value in sorted(model.state_dict().items()):
            if name.startswith('head.pgsma_engine.'):continue
            native_hash.update(name.encode('utf8'))
            native_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        run_info.update({'status':'CHECKS_PASSED','native_state_sha256':native_hash.hexdigest(),
                   'training_batch_stream_sha256':stream_digest.hexdigest(),
                   'successful_native_updates':valid_updates,
                   'successful_semantic_updates':semantic_updates,'amp_skips':overflow_count,
                   'first_training_batch_sha256':first_digest,'seconds_elapsed':round(time.monotonic()-started,2),
                   'last_telemetry':last_metrics,'clean_validation':'subset_only'})
        save_pilot_checkpoint(out/'pilot_last.pth',model,native_optimizer,semantic_optimizer,scaler,
            {'variant':args.variant,'steps':args.steps,'successful_native_updates':valid_updates,
             'successful_semantic_updates':semantic_updates,'seed':args.seed,
             'semantic_recipe':'source_dualview_fg_guard_halo_utility_gate_v2_'+args.candidate_objective,
             'candidate_objective':args.candidate_objective,
             'clean_val_sha256':labels_file_digest,'source_sha256':checkpoint_info['sha256']})
        atomic_json(out/'result.json',run_info)
        return run_info
    except Exception as e:
        run_info.update({'status':'BLOCKED','error':f'{type(e).__name__}: {e}',
                   'steps_logged':len(logs),'successful_native_updates':valid_updates,
                   'successful_semantic_updates':semantic_updates})
        atomic_json(out/'result.json',run_info)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--candidate-objective',
                   choices=('v2','margin_probe','bounded_probe','margin_bg_probe','bounded_bg_probe'),
                   default='v2',
                   help='Opt-in causal objective ablation; native training/inference unchanged')
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--checkpoint',default='weights/yolox_s.pth')
    p.add_argument('--warmstart',default=None,help='Required for memory/pgsma; native pilot checkpoint')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--device',default='cuda')
    p.add_argument('--amp',action='store_true')
    p.add_argument('--trace-parity',action='store_true',
                   help='Write per-stage native model/gradient/optimizer/RNG fingerprints at diagnostic steps')
    p.add_argument('--strict-determinism',action='store_true',
                   help='Require deterministic PyTorch/CUDA algorithms; may reveal unsupported ops')
    p.add_argument('--isolate-semantic-rng',action='store_true',
                   help='Capture/restore global RNG around semantic optimizer/probe')
    p.add_argument('--no-mosaic',action='store_true')
    p.add_argument('--native-lr',type=float,default=None)
    p.add_argument('--semantic-lr',type=float,default=0.001)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--eval-images',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--conf',type=float,default=0.01)
    p.add_argument('--nms',type=float,default=0.60)
    p.add_argument('--out-dir',required=True)
    args=p.parse_args()
    if args.log_every<1 or args.log_every>2048:p.error('--log-every must be 1..2048')
    if args.semantic_lr<=0 or args.semantic_lr>0.01:p.error('semantic LR outside safety envelope')
    if not (0<args.conf<1 and 0<args.nms<1):p.error('Confidence/NMS thresholds must be inside (0,1)')
    try:
        result=run(args)
        print(json.dumps({'status':result['status'],'variant':result['variant'],
             'native_updates':result['successful_native_updates'],
             'semantic_updates':result['successful_semantic_updates']},indent=2))
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'},indent=2))
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
