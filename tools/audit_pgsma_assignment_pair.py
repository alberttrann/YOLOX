"""Matched, NO-OPTIMIZER assignment-quality audit on one trained native checkpoint.

Both SimOTA and SCPA are evaluated on the SAME native predictions, images,
weights, and RNG snapshot. This is a diagnostic, not a training or AP audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch

from ._pgsma_pilot_core import atomic_json, group_scale
from .pgsma_bounded_pilot import (seed_everything, capture_rng, restore_rng,
                                   load_weights, verify_batch)

LEVELS=('P3','P4','P5')
BINS=('lt4','4to8','8to16','16to32','ge32')


def _empty_bucket():
    return {name:{'gt':0,'covered':0,'positives':0,'iou_sum':0.,'low_iou':0}
            for name in BINS}


def _measure(model, labels):
    cache=model.head.pgsma_cache
    assigned=cache['match']
    ids=assigned['fg_ids'].long().detach().cpu()
    uid=assigned['gt_uid'].long().detach().cpu()
    quality=assigned['fg_quality'].float().detach().cpu()
    bsz=int(labels.shape[0]);level_shapes=[int(x.shape[-2]*x.shape[-1]) for x in cache['features']]
    n_per_image=sum(level_shapes)
    expected_offsets=[0,level_shapes[0],level_shapes[0]+level_shapes[1],n_per_image]
    gt_sizes=[]
    for b in range(bsz):
        lab=labels[b].detach().cpu()
        valid=(lab[:,3]>0)&(lab[:,4]>0)&torch.isfinite(lab).all(dim=1)
        gt_sizes.extend(lab[valid,3:5].min(dim=1).values.tolist())
    if len(gt_sizes)!=int(model.head.last_assignment_stats['num_gt_total']):
        raise RuntimeError('GT count mismatch between batch and matcher')
    if ids.numel()!=uid.numel() or ids.numel()!=quality.numel():
        raise RuntimeError('Inconsistent positive identity/quality lengths')
    if uid.numel() and (int(uid.max())>=len(gt_sizes) or int(uid.min())<0):
        raise RuntimeError('Invalid matched GT identity')
    counts=torch.bincount(uid,minlength=len(gt_sizes)) if gt_sizes else torch.zeros(0,dtype=torch.long)
    result={'gt':len(gt_sizes),'covered':int((counts>0).sum()),'positives':len(ids),
            'iou_sum':float(quality.sum()),'low_iou_0p1':int((quality<.1).sum()),
            'low_iou_0p3':int((quality<.3).sum()),
            'levels':{},'sizes':_empty_bucket(),
            'core_positives':int(model.head.last_assignment_stats.get('core_positives',0))}
    anchor_mod=ids%n_per_image
    for i,level in enumerate(LEVELS):
        subset=(anchor_mod>=expected_offsets[i])&(anchor_mod<expected_offsets[i+1])
        q=quality[subset]
        result['levels'][level]={'positives':int(q.numel()),'iou_sum':float(q.sum()),
                                 'low_iou_0p1':int((q<.1).sum()),'low_iou_0p3':int((q<.3).sum())}
    for g,size in enumerate(gt_sizes):
        bin_name=group_scale(size)
        item=result['sizes'][bin_name]
        item['gt']+=1
        item['covered']+=int(counts[g]>0)
    for i in range(len(uid)):
        bin_name=group_scale(gt_sizes[int(uid[i])]);q=float(quality[i]);item=result['sizes'][bin_name]
        item['positives']+=1;item['iou_sum']+=q;item['low_iou']+=int(q<.1)
    return result


def _aggregate(samples):
    n=len(samples)
    if not n:raise RuntimeError('No batches audited')
    ag={'batches':n,'total_gt':0,'covered_gt':0,'total_positives':0,'mean_positive_iou':None,
        'fraction_positive_iou_lt_0p1':None,'fraction_positive_iou_lt_0p3':None,
        'mean_positives_per_gt':None,'core_positives':0,'levels':{},'sizes':{}}
    accum_sum=0.;low01=0;low03=0
    for x in samples:
        ag['total_gt']+=x['gt'];ag['covered_gt']+=x['covered']
        ag['total_positives']+=x['positives'];ag['core_positives']+=x['core_positives']
        accum_sum+=x['iou_sum'];low01+=x['low_iou_0p1'];low03+=x['low_iou_0p3']
    if ag['total_positives']:
        ag['mean_positive_iou']=accum_sum/ag['total_positives']
        ag['fraction_positive_iou_lt_0p1']=low01/ag['total_positives']
        ag['fraction_positive_iou_lt_0p3']=low03/ag['total_positives']
    if ag['total_gt']:
        ag['mean_positives_per_gt']=ag['total_positives']/ag['total_gt']
    ag['coverage_fraction']=(ag['covered_gt']/ag['total_gt'] if ag['total_gt'] else None)
    ag['uncovered_gt']=ag['total_gt']-ag['covered_gt']
    for level in LEVELS:
        pos=sum(x['levels'][level]['positives'] for x in samples)
        s=sum(x['levels'][level]['iou_sum'] for x in samples)
        ag['levels'][level]={'positives':pos,'mean_positive_iou':s/pos if pos else None,
            'fraction_iou_lt_0p1':sum(x['levels'][level]['low_iou_0p1'] for x in samples)/pos if pos else None}
    for bin_name in BINS:
        grp={key:sum(x['sizes'][bin_name][key] for x in samples)
             for key in ('gt','covered','positives','iou_sum','low_iou')}
        ag['sizes'][bin_name]={'gt':grp['gt'],'covered_gt':grp['covered'],
            'coverage_fraction':grp['covered']/grp['gt'] if grp['gt'] else None,
            'mean_positives_per_gt':grp['positives']/grp['gt'] if grp['gt'] else None,
            'mean_positive_iou':grp['iou_sum']/grp['positives'] if grp['positives'] else None,
            'fraction_iou_lt_0p1':grp['low_iou']/grp['positives'] if grp['positives'] else None}
    return ag


def run(args):
    if not (1<=args.batches<=64 and 1<=args.batch_size<=4):
        raise ValueError('Audit bounded to 1..64 batches and batch_size 1..4')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    seed_everything(args.seed)
    from exps.default.tde_yolox_s_pgsma import Exp
    exp=Exp();exp.data_num_workers=0
    if exp.allow_full_training:raise RuntimeError('Full training guard must remain False')
    model=exp.get_model()
    source=load_weights(model,args.checkpoint,pilot_checkpoint=True)
    model=model.to(args.device)
    model.train()
    # Prevent BN running state changing across the two matcher forwards;
    # BOTH algorithms see identical frozen native predictions.
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
    train_loader=exp.get_data_loader(args.batch_size,False,no_aug=args.no_mosaic,cache_img=False)
    batches=iter(train_loader)
    samples={'simota':[],'scpa':[]}
    native_output_max_delta=0.
    for batch_id in range(args.batches):
        try:batch=next(batches)
        except StopIteration:batches=iter(train_loader);batch=next(batches)
        images,targets=batch[:2]
        verify_batch(images,targets,exp)
        images=images.float().to(args.device)
        targets=targets.float().to(args.device)
        images,targets=exp.preprocess(images,targets,exp.input_size)
        mask=images[:,3:4]>.5
        rgb=images[:,:3].contiguous()
        rng=capture_rng()
        baseline=None
        with torch.no_grad():
            for name,mode in (('simota','baseline'),('scpa','scpa')):
                restore_rng(rng)
                model.head.assignment=mode
                model.head.use_baseline_assignments=(mode=='baseline')
                model(rgb,targets,valid_image_mask=mask)
                prediction=[x.clone().detach() for x in
                            model.head.pgsma_cache['native_cls']+model.head.pgsma_cache['native_obj']]
                if baseline is None:baseline=prediction
                else:
                    for a,b in zip(baseline,prediction):
                        if a.shape!=b.shape:raise RuntimeError('Shape mismatch between frozen predictions')
                        native_output_max_delta=max(native_output_max_delta,float((a-b).abs().max()))
                samples[name].append(_measure(model,targets))
        print(f'pair {batch_id+1}/{args.batches} GT='+str(samples['simota'][-1]['gt'])+
              f" simota_fg={samples['simota'][-1]['positives']} scpa_fg={samples['scpa'][-1]['positives']}",flush=True)
    if native_output_max_delta>1e-6:
        raise RuntimeError('Native prediction changed with matching algorithm: '+str(native_output_max_delta))
    report={'status':'CHECKS_PASSED','training_performed':False,
      'checkpoint':source,'seed':args.seed,'batches':args.batches,
      'input_size':list(exp.input_size),
      'same_native_logits_max_abs_difference':native_output_max_delta,
      'simota':_aggregate(samples['simota']),'scpa':_aggregate(samples['scpa']),
      'limitations':['Predictions frozen at ONE trained native checkpoint',
        'Metrics are assignment IoU and coverage, NOT detection AP/AR',
        'Not a full-source annotation provenance audit','No training, gradient or optimizer updates']}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--no-mosaic',action='store_true')
    p.add_argument('--json',required=True)
    a=p.parse_args()
    try:
        report=run(a)
        atomic_json(a.json,report)
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0
    except Exception as exc:
        report={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
        atomic_json(a.json,report)
        print(json.dumps(report,indent=2))
        return 2
if __name__=='__main__':sys.exit(main())
