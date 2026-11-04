"""Real YOLOX minibatch CUDA/AMP and native optimizer-step equivalence audit.

DRY RUN: optimizers update only local copies in memory; nothing is saved or
written to the user's real checkpoint. Use --checkpoint for meaningful features.
Run from the actual YOLOX checkout. Uses no synthetic images/labels.
"""
from __future__ import annotations
import argparse
import copy
import json
import random
import sys
import torch
from ._pgsma_preflight_common import (
    load_exp,one_real_train_batch,load_torch_checkpoint,unwrap_state_dict,
    normalize_state_keys,checkpoint_coverage,image_gt_sizes,dump_json)


def make_cuda_grad_scaler(enabled):
    """Support early PyTorch (torch.cuda.amp) and modern torch.amp API."""
    namespace = getattr(torch, 'amp', None)
    modern = getattr(namespace, 'GradScaler', None) if namespace is not None else None
    if modern is not None:
        try:
            return modern('cuda', enabled=enabled)
        except TypeError:
            pass  # Transitional signature: use legacy CUDA API.
    legacy = getattr(getattr(torch, 'cuda', None), 'amp', None)
    ctor = getattr(legacy, 'GradScaler', None)
    if ctor is None:
        raise RuntimeError('PyTorch exposes no supported CUDA GradScaler')
    return ctor(enabled=enabled)

def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def native_optimizer(exp, model, batch_size):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if not p.requires_grad or name.startswith('head.pgsma_engine.'):
            continue
        if ('bias' in name or p.ndim<=1 or 'norm' in name or 'gn.' in name):
            no_decay.append(p)
        else:decay.append(p)
    return torch.optim.SGD([
        {'params':decay,'weight_decay':exp.weight_decay},
        {'params':no_decay,'weight_decay':0.0}],
        lr=exp.basic_lr_per_img*batch_size,momentum=exp.momentum,nesterov=True)


def native_named(model):
    return {k:v for k,v in model.named_parameters() if 'pgsma_engine.' not in k}


def state_diff(a,b):
    maxabs=0.;names=[]
    for k,x in a.items():
        y=b[k]
        if x.shape!=y.shape:raise ValueError(f'mismatched tensor shape {k}')
        err=float((x.detach().float()-y.detach().float()).abs().max()) if x.numel() else 0.
        if err>maxabs:maxabs=err
        if err>1e-5:names.append(k)
    return {'max_abs':maxabs,'more_than_1e-5':names[:12], 'count_over_1e-5':len(names)}


def tensors_named(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()
            if 'pgsma_engine.' not in k}


def read_exp(module_path=None):
    if module_path is None:return load_exp()
    import importlib
    mod=importlib.import_module(module_path)
    return mod.Exp()


def evaluate_one(model, images, labels, use_amp, semantic_optimizer=None,
                 semantic_seed=20261103, mode='pgsma', valid_image_mask=None):
    from yolox.core.pgsma_step import semantic_optimizer_step
    device=images.device
    amp=bool(use_amp and device.type=='cuda')
    model.train()
    reset_seed(8803)
    with torch.autocast(device_type=device.type,enabled=amp):
        out=model(images,labels,valid_image_mask=valid_image_mask)
    if not bool(torch.isfinite(out['total_loss'])):
        raise FloatingPointError('Nonfinite native loss on a real minibatch')
    sem_report=None
    if semantic_optimizer is not None:
        with torch.autocast(device_type=device.type,enabled=False):
            success, sem_report=semantic_optimizer_step(model,semantic_optimizer,
                                                        seed=semantic_seed,
                                                        correction_mode=mode)
        if not success:
            raise FloatingPointError(f'Semantic optimizer failed: {sem_report}')
        contamination=[name for name,p in native_named(model).items() if p.grad is not None]
        if contamination:
            raise AssertionError(f'Semantic step wrote native gradients: {contamination[:5]}')
    return out,sem_report


def audit(args):
    device=torch.device(args.device)
    if args.amp and device.type!='cuda':raise ValueError('--amp requires --device cuda')
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('torch.cuda.is_available() is false on this machine')
    torch.backends.cudnn.benchmark=False
    if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.deterministic=True
    reset_seed(331)
    exp=read_exp(args.exp_module)
    if args.force_mosaic and not args.mosaic:
        raise ValueError('--force-mosaic requires --mosaic')
    if args.force_mosaic:
        # Force the *actual* MosaicDetection branch before sampling.
        # Merely passing mosaic=True leaves mosaic_prob at its configured 0.5.
        exp.data_num_workers=0
        loader=exp.get_data_loader(int(args.batch_size),is_distributed=False,
                                   no_aug=False,cache_img=False)
        ds=loader.dataset
        if type(ds).__name__!='MosaicDetection' or not bool(getattr(ds,'enable_mosaic',False)):
            raise RuntimeError('Cannot force mosaic: MosaicDetection branch not active')
        ds.mosaic_prob=1.0
        imgs,tars,*_=next(iter(loader))
    else:
        imgs,tars,_,loader=one_real_train_batch(exp,args.batch_size,mosaic=args.mosaic)
    import hashlib
    input_report={'image_shape':list(imgs.shape),'target_shape':list(tars.shape),
                  'gt':image_gt_sizes(tars),'dataset_type':type(loader.dataset).__name__,
                  'mosaic_enabled':bool(getattr(loader.dataset,'enable_mosaic',False)),
                  'mosaic_probability':float(getattr(loader.dataset,'mosaic_prob',0.0)),
                  'mosaic_forced':bool(args.force_mosaic),
                  'image_sha256_24':hashlib.sha256(imgs.contiguous().numpy().tobytes()).hexdigest()[:24],
                  'labels_sha256_24':hashlib.sha256(tars.contiguous().numpy().tobytes()).hexdigest()[:24]}
    imgs=imgs.float().to(device);tars=tars.float().to(device)
    valid_image_mask=None
    if imgs.shape[1]==4:
        valid_image_mask=imgs[:,3:4] > 0.5
        imgs=imgs[:,:3].contiguous()
        input_report['valid_mask_ratio'] = float(valid_image_mask.float().mean())
        input_report['has_explicit_validity_mask'] = True
    else:
        input_report['has_explicit_validity_mask'] = False
    if imgs.shape[2]%32 or imgs.shape[3]%32:
        raise ValueError('Real loader image dimensions must be divisible by 32')
    if args.max_pixels and imgs.shape[-2]*imgs.shape[-1] > args.max_pixels:
        raise RuntimeError('Real batch exceeds --max-pixels; do not silently resize annotations')
    base=exp.get_model().cpu()
    ckpt_summary=None
    if args.checkpoint:
        ckpt=load_torch_checkpoint(args.checkpoint)
        raw,subkey=unwrap_state_dict(ckpt,prefer_ema=args.prefer_ema)
        cov=checkpoint_coverage(base,normalize_state_keys(raw))
        base.load_state_dict(cov['matched'],strict=False)
        ckpt_summary={'source':args.checkpoint,'subkey':subkey,'matched_tensors':len(cov['matched']),
                      'dark2_ratio':cov['groups']['backbone.backbone.dark2']['element_ratio']}
        if (ckpt_summary['dark2_ratio'] or 0)<.85:
            raise RuntimeError('Checkpoint dark2 coverage <85%: inspect checkpoint before real AMP audit')
    base.train()
    base.to(device)
    dual=copy.deepcopy(base).to(device)
    opt_a=native_optimizer(exp,base,int(imgs.shape[0]))
    opt_b=native_optimizer(exp,dual,int(imgs.shape[0]))
    sem_opt=torch.optim.AdamW(dual.semantic_parameters(),lr=exp.semantic_lr,weight_decay=0)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    out_a,_=evaluate_one(base,imgs,tars,args.amp,valid_image_mask=valid_image_mask)
    stats_a=dict(base.head.last_assignment_stats)
    out_b,sem=evaluate_one(dual,imgs,tars,args.amp,sem_opt,mode=args.mode,
                           valid_image_mask=valid_image_mask)
    stats_b=dict(dual.head.last_assignment_stats)
    rawlossdiff=float((out_a['total_loss'].detach()-out_b['total_loss'].detach()).abs())
    if stats_a!=stats_b:raise AssertionError(f'Native assignments differ: {stats_a} vs {stats_b}')
    scaler_a=make_cuda_grad_scaler(enabled=args.amp)
    scaler_b=make_cuda_grad_scaler(enabled=args.amp)
    opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
    scaler_a.scale(out_a['total_loss']).backward()
    scaler_b.scale(out_b['total_loss']).backward()
    grads_a={n:p.grad.detach().clone() for n,p in native_named(base).items() if p.grad is not None}
    grads_b={n:p.grad.detach().clone() for n,p in native_named(dual).items() if p.grad is not None}
    if grads_a.keys()!=grads_b.keys():raise AssertionError('Native gradient structure mismatch')
    grad_diff=state_diff(grads_a,grads_b)
    native_semantic_grads=sum(p.grad is not None for p in base.semantic_parameters())
    if native_semantic_grads:raise AssertionError('Native-only backward touched semantic parameters')
    scaler_a.step(opt_a); scaler_a.update()
    scaler_b.step(opt_b); scaler_b.update()
    state_a=tensors_named(base);state_b=tensors_named(dual)
    param_diff=state_diff(state_a,state_b)
    # Compare momentum buffers, not just weight values.
    opt_state_a=[opt_a.state[p].get('momentum_buffer') for group in opt_a.param_groups for p in group['params']]
    opt_state_b=[opt_b.state[p].get('momentum_buffer') for group in opt_b.param_groups for p in group['params']]
    moment_err=max((float((x-y).abs().max()) for x,y in zip(opt_state_a,opt_state_b)
                    if x is not None and y is not None),default=0.)
    cache=dual.head.pgsma_cache
    feature_counts=[int(x.shape[-2]*x.shape[-1]) for x in cache['features']]
    boundaries=[];c=0
    for n in feature_counts:
        c+=n;boundaries.append(c)
    fg=cache['match']['fg_ids']
    total_per_image=c
    local_fg=(fg%total_per_image).cpu().tolist()
    pos_per_level={f'P{i+3}':sum((j<(boundaries[i])) and (i==0 or j>=boundaries[i-1]) for j in local_fg)
                   for i in range(len(boundaries))}
    results={'device':str(device),'amp':args.amp,'torch_version':torch.__version__,
             'scaler_api':type(scaler_a).__module__+'.'+type(scaler_a).__name__,
             'real_batch':input_report,
             'checkpoint':ckpt_summary,'native_loss_control':float(out_a['total_loss'].detach()),
             'native_loss_with_semantic':float(out_b['total_loss'].detach()),
             'native_loss_absdiff':rawlossdiff,'native_assignments':stats_b,
             'pos_per_level':pos_per_level,
             'teacher_P5_note':('P5 lacks positives in this minibatch; zero P5 teacher gradient is expected'
                                if pos_per_level.get('P5',0)==0 else 'P5 has matched positives'),
             'semantic_step':sem,'native_grad_diff':grad_diff,
             'native_parameter_and_buffer_diff':param_diff,
             'momentum_buffer_max_absdiff':moment_err,
             'native_scaler_scale_control':float(scaler_a.get_scale()),
             'native_scaler_scale_with_semantic':float(scaler_b.get_scale()),
             'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()) if device.type=='cuda' else None,
             'limitations':['one real batch is not a full-dataset fidelity proof',
                            'no DDP (bundle is single-GPU only)',
                            'actual gradient parity is subject to CUDA deterministic tolerance'],
             'status':'CHECKS_PASSED'}
    atol=args.tolerance
    if (rawlossdiff>atol or grad_diff['max_abs']>atol or param_diff['max_abs']>atol or moment_err>atol):
        results['status']='BLOCKED'
        results['blocker']='Native optimizer-step equivalence exceeded tolerance'
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--prefer-ema',action='store_true')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--mosaic',action='store_true')
    p.add_argument('--force-mosaic',action='store_true',help='Set MosaicDetection.mosaic_prob=1.0 BEFORE sampling; requires --mosaic')
    p.add_argument('--mode',choices=['pgsma','memory'],default='pgsma')
    p.add_argument('--exp-module',default=None,help='e.g. exps.default.tde_yolox_s_pgsma_mosaic_audit')
    p.add_argument('--tolerance',type=float,default=5e-5)
    p.add_argument('--max-pixels',type=int,default=0)
    p.add_argument('--json',default=None)
    a=p.parse_args()
    try:res=audit(a)
    except Exception as exc:res={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
    dump_json(res,a.json)
    sys.exit(2 if res['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
