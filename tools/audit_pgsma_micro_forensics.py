"""Explain GT drops at the final TrainTransform boundary without changing training.

Install alongside tools.audit_pgsma_annotation_survival from the fidelity kit.
The earlier audit intentionally disables final HSV/flip to allow geometry matching.
This tool uses that same controlled setting, and reports the actual dimensions
of every unmatched GT and whether the standard YOLOX >1 px minimum-side filter
explains it. The filter is diagnosed, NEVER changed.

This does NOT establish source-annotation retention through Mosaic crop/affine.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

from tools.audit_pgsma_annotation_survival import TracePreproc, size_bin

MIN_SIDE_THRESHOLD_DIAGNOSTIC = 1.0  # upstream YOLOX TrainTransform: strictly greater than 1 pixel


def classify_trace(trace, *, threshold=MIN_SIDE_THRESHOLD_DIAGNOSTIC):
    """Explain unmatched valid boxes and compare against a geometric filter oracle.

    Neither a match nor a filter predicate certifies the presence of visual
    evidence. Duplicate GTs and fallback behavior can complicate attribution.
    """
    before=np.asarray(trace['before_boxes'], dtype=np.float64).reshape(-1,5)
    after=np.asarray(trace['after_labels'], dtype=np.float64).reshape(-1,5)
    missing=set(trace['unmatched_pre_indices'])
    scale=float(trace['scale'])
    valid=(np.isfinite(before).all(axis=1) &
           ((before[:,2]-before[:,0])>0) &
           ((before[:,3]-before[:,1])>0))
    w=(before[:,2]-before[:,0])*scale
    h=(before[:,3]-before[:,1])*scale
    predicts_keep=valid & (np.minimum(w,h)>threshold)
    real_post=int(np.count_nonzero(np.isfinite(after).all(axis=1) & (after[:,3]>0) & (after[:,4]>0)))
    expected_keep=int(predicts_keep.sum())
    fallback_possible=bool(valid.any() and expected_keep==0 and real_post>0)
    missing_details=[]
    for idx in sorted(missing):
        if not valid[idx]:continue
        side=float(min(w[idx],h[idx]))
        filt=bool(side<=threshold)
        # This is a *prediction* from a source-code predicate. Not proof.
        reason=('consistent_with_min_side_filter' if filt else
                'not_explained_by_min_side_filter')
        missing_details.append({
            'gt_pre_index':int(idx), 'class_id':int(before[idx,4]),
            'size_bin':size_bin(w[idx],h[idx]),
            'width_px':round(float(w[idx]),6),
            'height_px':round(float(h[idx]),6),
            'min_side_px':round(side,6),
            'reason':reason,
            'before_bbox_xyxy':[round(float(x),5) for x in before[idx,:4]],
        })
    good_unmatched=sum(x['reason']=='consistent_with_min_side_filter' for x in missing_details)
    unknown_unmatched=len(missing_details)-good_unmatched
    invalid=int((~valid).sum())
    return {
        'valid_pre_boxes':int(valid.sum()),
        'invalid_pre_boxes':invalid,
        'post_valid_boxes':real_post,
        'predicted_keep_by_min_side_filter':expected_keep,
        'predicted_filter_by_min_side':int((valid&~predicts_keep).sum()),
        'unmatched_valid_boxes':len(missing_details),
        'unmatched_explained_by_min_side_filter':good_unmatched,
        'unmatched_not_explained_by_min_side_filter':unknown_unmatched,
        'post_count_agrees_with_filter_prediction':bool(real_post==expected_keep),
        'traintransform_all_filtered_fallback_possible':fallback_possible,
        'missing':missing_details,
    }


def source_filter_inspection():
    """Report whether installed YOLOX source visibly uses the >1 side filter."""
    try:
        from yolox.data import TrainTransform
        source=inspect.getsource(TrainTransform.__call__)
        source_compact=''.join(source.split())
        # Only accept exact, known expression format; report unknown otherwise.
        matches=('np.minimum(boxes[:,2],boxes[:,3])>1' in source_compact or
                 'np.minimum(boxes[:,2],boxes[:,3])>1.0' in source_compact)
        return {'available':True,
                'traintransform_call_sha256':hashlib.sha256(source.encode('utf8')).hexdigest(),
                'standard_strict_gt_1px_filter_detected':bool(matches),
                'diagnostic_threshold_px':MIN_SIDE_THRESHOLD_DIAGNOSTIC,
                'warning':None if matches else 'Installed TrainTransform differs; do not attribute loss to >1px filter without reviewing its source'}
    except Exception as e:
        return {'available':False,
                'diagnostic_threshold_px':MIN_SIDE_THRESHOLD_DIAGNOSTIC,
                'warning':f'Could not inspect installed TrainTransform: {type(e).__name__}: {e}'}


def audit(args):
    import torch
    from exps.default.tde_yolox_s_pgsma import Exp
    out={'status':'CHECKS_PASSED', 'purpose':'audit_only_no_training',
         'seed':args.seed,'batches':args.batches,'batch_size':args.batch_size,
         'threshold_code_check':source_filter_inspection(), 'variants':{},'blockers':[],
         'scope':['Final TrainTransform only, after upstream Mosaic crop/affine',
                  'Final flip and HSV disabled for geometric identity audit',
                  'Does not modify box filtering or training parameters',
                  'The 1px rule is a diagnostic oracle, not an endorsement of that threshold']}
    for variant in args.variants:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        exp=Exp(); exp.data_num_workers=0
        loader=exp.get_data_loader(args.batch_size,is_distributed=False,
                                   no_aug=(variant=='no_aug'), cache_img=False)
        ds=loader.dataset
        if type(ds).__name__!='MosaicDetection':
            raise RuntimeError(f'{variant}: expected MosaicDetection; got {type(ds).__name__}')
        if variant=='forced_mosaic':
            ds.mosaic_prob=1.0
            if not ds.enable_mosaic:raise RuntimeError('Cannot force Mosaic; enable_mosaic=False')
        tracer=TracePreproc(ds.preproc, exp.input_size)
        ds.preproc=tracer
        it=iter(loader)
        for _ in range(args.batches):
            next(it)
        if len(tracer.traces)!=args.batch_size*args.batches:
            raise AssertionError(f'{variant}: expected {args.batch_size*args.batches} trace samples, got {len(tracer.traces)}')
        counts=Counter(); lost=[]; exceptions=[]
        for sample_index,trace in enumerate(tracer.traces):
            report=classify_trace(trace)
            for key in ['valid_pre_boxes','invalid_pre_boxes','post_valid_boxes',
                        'predicted_keep_by_min_side_filter','predicted_filter_by_min_side',
                        'unmatched_valid_boxes','unmatched_explained_by_min_side_filter',
                        'unmatched_not_explained_by_min_side_filter']:
                counts[key]+=report[key]
            if not report['post_count_agrees_with_filter_prediction']:
                counts['samples_count_disagree']+=1
                exceptions.append({'sample_index':sample_index,
                                   'reason':'observed_post_count_not_equal_to_filter_prediction',
                                   'post_count':report['post_valid_boxes'],
                                   'predicted_count':report['predicted_keep_by_min_side_filter'],
                                   'possible_all_filtered_fallback':report['traintransform_all_filtered_fallback_possible']})
            if report['unmatched_not_explained_by_min_side_filter']:
                counts['samples_with_unexplained_unmatched']+=1
            for row in report['missing']:
                row['sample_index']=sample_index
                lost.append(row)
        stats=dict(counts)
        stats.update({'variant':variant,'samples_traced':len(tracer.traces),
                      'mosaic_probability':float(getattr(ds,'mosaic_prob',0.0)),
                      'mosaic_enabled':bool(getattr(ds,'enable_mosaic',False)),
                      'unmatched_details':lost[:args.max_details],
                      'num_unmatched_details_omitted':max(0,len(lost)-args.max_details),
                      'exceptions':exceptions[:args.max_details]})
        out['variants'][variant]=stats
        if stats.get('unmatched_not_explained_by_min_side_filter',0) or stats.get('samples_count_disagree',0):
            out['blockers'].append(f'{variant}: some final-transform losses not explained by strict >1px size filtering; investigate recorded exceptions')
    if out['threshold_code_check'].get('standard_strict_gt_1px_filter_detected') is False:
        out['blockers'].append('Installed upstream TrainTransform does not match the known strict >1px predicate')
    if out['blockers']:out['status']='NEEDS_INVESTIGATION'
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batches',type=int,default=40)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--variants',nargs='+',default=['no_aug','forced_mosaic','mixed_mosaic'],
                    choices=['no_aug','forced_mosaic','mixed_mosaic'])
    ap.add_argument('--max-details',type=int,default=100)
    ap.add_argument('--json',default='audits/micro_forensics.json')
    args=ap.parse_args()
    try:
        if args.batches<1 or args.batch_size<1 or args.max_details<0:raise ValueError('Invalid audit counts')
        res=audit(args)
    except Exception as e:
        res={'status':'BLOCKED','error':type(e).__name__+': '+str(e)}
    txt=json.dumps(res,indent=2,allow_nan=False)
    print(txt)
    path=Path(args.json)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(txt+'\n',encoding='utf8')
    sys.exit(0 if res['status']=='CHECKS_PASSED' else 2)

if __name__=='__main__':main()
