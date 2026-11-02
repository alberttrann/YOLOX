#!/usr/bin/env python3
"""Persistent-GradScaler forensics for tde-vNext / YOLOX.

Use from repository root. No checkpoint files are modified or saved.
Default --mode no-update runs forward/backward with real consecutive batches,
while SGD lr=0 to isolate AMP scale recovery. --mode update resumes the real
SGD optimizer and changes weights IN MEMORY ONLY; it cannot faithfully recreate
an earlier training trajectory, but can expose update-dependent instability.

Examples:
 python tdevnext_amp_continuity_probe.py --checkpoint YOLOX_outputs/tde_yolox_s/last_epoch_ckpt.pth --max-batches 64 --batch-size 8
 python tdevnext_amp_continuity_probe.py --checkpoint YOLOX_outputs/tde_yolox_s/last_epoch_ckpt.pth --mode update --max-batches 64 --batch-size 8
 python tdevnext_amp_continuity_probe.py --checkpoint YOLOX_outputs/tde_yolox_s/last_epoch_ckpt.pth --dtype fp32 --no-amp --max-batches 16
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch


def classify_group(name):
    if name.startswith('backbone.backbone.stem'): return 'stem'
    if 'dark2' in name: return 'dark2'
    if 'dark3' in name: return 'dark3'
    if 'dark4' in name: return 'dark4'
    if 'dark5' in name: return 'dark5'
    if 'spi_highway' in name: return 'spi'
    if 'C3_' in name: return 'neck'
    if 'head.reg_' in name: return 'head_reg'
    if 'head.obj_' in name: return 'head_obj'
    if 'head.cls_' in name: return 'head_cls'
    if 'ttt_lrs' in name: return 'ttt_meta_lr'
    if 'projector' in name: return 'projector'
    if 'memory_banks' in name: return 'memory'
    if 'affordance_head' in name: return 'affordance'
    return 'other'


def grad_diagnostics(model, example_limit=6):
    bad_elements = 0
    bad_params = 0
    inf_params = []
    by_group = Counter()
    finite_norm_sq = 0.0
    max_finite_grad = 0.0
    for name,p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach()
        good = torch.isfinite(g)
        num_bad = int((~good).sum().item())
        if num_bad:
            bad_params += 1
            bad_elements += num_bad
            by_group[classify_group(name)] += num_bad
            if len(inf_params) < example_limit:
                inf_params.append({'name':name, 'bad':num_bad,
                                   'nan':int(torch.isnan(g).sum().item()),
                                   'inf':int(torch.isinf(g).sum().item())})
        # Analyze only finite gradient entries. FP64 for reliable norm accumulation.
        finite = g[good].double()
        if finite.numel():
            finite_norm_sq += float(finite.square().sum().item())
            max_finite_grad = max(max_finite_grad, float(finite.abs().max().item()))
    return dict(bad_elements=bad_elements, bad_parameters=bad_params,
                bad_groups=dict(by_group), first_bad_params=inf_params,
                finite_element_norm_fp64=math.sqrt(finite_norm_sq),
                max_finite_abs_grad=max_finite_grad)


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--exp', default='exps/default/tde_yolox_s.py')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-batches', type=int, default=64)
    ap.add_argument('--dtype', choices=['fp16','fp32'], default='fp16')
    ap.add_argument('--no-amp', action='store_true', help='True FP32 forward if used with --dtype fp32')
    ap.add_argument('--init-scale', type=float, default=65536)
    ap.add_argument('--mode', choices=['no-update','update'], default='no-update')
    ap.add_argument('--clip', type=float, default=10.0)
    ap.add_argument('--size', default=None, help='H,W e.g. 640,1152')
    ap.add_argument('--epoch', type=int, default=None)
    ap.add_argument('--seed', type=int, default=703)
    ap.add_argument('--output', default='tdevnext_amp_continuity.json')
    args=ap.parse_args()
    if not torch.cuda.is_available(): ap.error('CUDA required')
    if args.batch_size < 2 or args.max_batches < 1: ap.error('batch-size >= 2, max-batches >= 1')
    if args.no_amp and args.dtype!='fp32': ap.error('--no-amp requires --dtype fp32')
    if args.clip <= 0: ap.error('--clip must be positive')

    from yolox.exp import get_exp
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    exp=get_exp(args.exp, None)
    exp.data_num_workers=0
    model=exp.get_model().cuda().train()
    ckpt=torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model',ckpt),strict=True)
    epoch=args.epoch if args.epoch is not None else int(ckpt.get('start_epoch',1))
    if hasattr(model,'set_meta_training_state'):
        model.set_meta_training_state(epoch, getattr(exp,'max_epoch',80))
    optimizer=exp.get_optimizer(args.batch_size)
    if args.mode=='update':
        if 'optimizer' not in ckpt: ap.error('Checkpoint has no optimizer state')
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        for pg in optimizer.param_groups: pg['lr']=0.0

    # Counting a genuine call to optimizer.step is more reliable than comparing scale alone.
    step_calls=[0]
    original_step=optimizer.step
    def counted_step(*a,**kw):
        step_calls[0] += 1
        return original_step(*a,**kw)
    optimizer.step=counted_step
    use_amp=not args.no_amp
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp,init_scale=args.init_scale)
    loader=exp.get_data_loader(args.batch_size,is_distributed=False,no_aug=False,cache_img=False)
    shape=tuple(map(int,args.size.split(','))) if args.size else tuple(exp.input_size)
    dtype=torch.float16 if args.dtype=='fp16' else torch.float32

    records=[]
    print('Persistent scale probe: mode=',args.mode,'dtype=',args.dtype,'AMP=',use_amp,
          'epoch=',epoch,'batch=',args.batch_size,'initial_scale=',args.init_scale,flush=True)
    for index,batch in enumerate(loader):
        if index>=args.max_batches: break
        images,targets=batch[:2]
        images=images.cuda().to(dtype); targets=targets.cuda().to(dtype)
        targets.requires_grad=False
        images,targets=exp.preprocess(images,targets,shape)
        optimizer.zero_grad(set_to_none=True)
        rec={'batch':index,'input_shape':list(images.shape),
             'gt_count':int((targets[...,3]>0).sum().item())}
        try:
            scale_before=float(scaler.get_scale()) if use_amp else 1.0
            with torch.cuda.amp.autocast(enabled=use_amp):
                result=model(images,targets)
            loss=result['total_loss']
            rec['loss']=float(loss.detach().cpu()); rec['scale_before']=scale_before
            rec['loss_finite']=bool(torch.isfinite(loss).item())
            before=step_calls[0]
            if rec['loss_finite']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                rec.update(grad_diagnostics(model))
                rec['grad_clip_applied']=False
                if rec['bad_elements']==0:
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=args.clip,
                                                       error_if_nonfinite=True)
                    rec['unclipped_global_norm']=float(norm.detach().cpu())
                    rec['clip_factor']=min(1.0,args.clip/(rec['unclipped_global_norm']+1e-6))
                    rec['grad_clip_applied']=True
                # If AMP found overflow, its cached inf flag suppresses the step.
                # Do not naively scrub nan/inf values after clipping.
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif rec['bad_elements']==0:
                    optimizer.step()
                rec['step_called']=step_calls[0]>before
                if rec['step_called'] and args.mode=='update':
                    if hasattr(model,'project_prototypes_to_unit_sphere'):
                        model.project_prototypes_to_unit_sphere()
            else:
                rec.update({'step_called':False,'bad_elements':None})
            rec['scale_after']=float(scaler.get_scale()) if use_amp else 1.0
            records.append(rec)
            print('batch=%03d gt=%3s scale=%7g -> %-7g bad=%5s step=%s clip=%s' % (
                index,rec['gt_count'],rec['scale_before'],rec['scale_after'],
                rec.get('bad_elements'),rec['step_called'],
                ('%.4f'%rec['clip_factor'] if 'clip_factor' in rec else '-')),flush=True)
        except Exception as exc:
            rec['error']=repr(exc); records.append(rec)
            print('ERROR at batch',index,repr(exc),flush=True)
            break
        finally:
            del images,targets

    good=sum(r.get('step_called',False) for r in records)
    bad=len(records)-good
    summary={'observed_batches':len(records), 'optimizer_step_calls':good,
             'skipped_or_failed_batches':bad, 'skip_fraction':bad/max(1,len(records)),
             'scale_at_end':float(scaler.get_scale()) if use_amp else 1.0,
             'recorded_nonfinite_batches':sum((r.get('bad_elements') or 0)>0 for r in records),
             'important_note':'Default no-update mode uses zero learning rate: the loss-scale trajectory is real, but weight-dependent instabilities cannot be reproduced.'}
    Path(args.output).write_text(json.dumps({'config':vars(args),'summary':summary,
                                               'records':records},indent=2,allow_nan=False),encoding='utf-8')
    print('SUMMARY:',json.dumps(summary,indent=2)); print('Saved',args.output)
    print('No checkpoint file was modified or saved.')

if __name__=='__main__': main()
