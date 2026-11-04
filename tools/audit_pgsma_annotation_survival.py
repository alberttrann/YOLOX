"""Non-training audit of final TrainTransform GT retention and letterbox anchor masks.

This audit runs the *actual* YOLOX data loader with num_workers=0, wraps its
preprocessing hook to see boxes before and after TrainTransform, and disables
only final flip/HSV so boxes can be matched geometrically. It distinguishes
losses *within the final TrainTransform* from earlier Mosaic crop/affine losses,
which it cannot measure. No checkpoints or optimizer steps are used.

Run: python -m tools.audit_pgsma_annotation_survival --batches 24 --batch-size 2
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np


def size_bin(width, height):
    """Fine scale bins defined by minimum effective input-space box dimension."""
    d = min(float(width), float(height))
    if d < 4: return '<4px'
    if d < 8: return '4-8px'
    if d < 16: return '8-16px'
    if d < 32: return '16-32px'
    return '>=32px'


def box_matches_pre_to_post(before, after, input_shape, original_shape, tol=1.5):
    """Greedy one-to-one xyxy match for final transform with flip disabled.

    before: [N,5] (xyxy,class), after: [L,5] (class,cx,cy,w,h).
    The upstream YOLOX TrainTransform should not alter box geometry beyond
    top-left letterbox scaling when final flip is disabled. Returns indices,
    not a claim about objects cropped out *before* this hook.
    """
    before=np.asarray(before,dtype=np.float64).reshape(-1,5)
    after=np.asarray(after,dtype=np.float64).reshape(-1,5)
    ih,iw=map(int,input_shape)
    oh,ow=map(int,original_shape)
    scale=min(ih/oh,iw/ow)
    candidates=[]
    for j,row in enumerate(after):
        c,cx,cy,w,h=row
        if not np.isfinite(row).all() or w<=0 or h<=0: continue
        candidates.append((j,c,np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])))
    # Match the most constrained inputs first, avoiding duplicate positives.
    match_pairs=[];available=set(j for j,_,_ in candidates)
    ranked=[]
    for i,row in enumerate(before):
        if not np.isfinite(row).all() or row[2]<=row[0] or row[3]<=row[1]: continue
        box=row[:4]*scale
        poss=sorted((float(np.max(np.abs(box-xyxy))),j)
                    for j,c,xyxy in candidates if abs(c-row[4])<1e-4)
        ranked.append((len([1 for err,j in poss if err<=tol]),i,poss))
    for _,i,poss in sorted(ranked):
        for err,j in poss:
            if err>tol: break
            if j in available:
                available.remove(j);match_pairs.append((i,j,err));break
    matched_pre=set(i for i,_,_ in match_pairs)
    return {
        'pre_valid_count':len(ranked),'post_valid_count':len(candidates),
        'matched_count':len(match_pairs),
        'unmatched_pre_indices':[i for _,i,_ in ranked if i not in matched_pre],
        'unmatched_post_indices':sorted(available),
        'match_error_max':max((e for _,_,e in match_pairs),default=0.0),
        'scale':scale,
    }


class TracePreproc:
    def __init__(self, original, input_size):
        self.original=original
        self.expected_size=tuple(input_size)
        self.traces=[]
        # Turn off *final* geometric flip for forensic identity matching.
        # The Mosaic crop/affine/scale stage remains unchanged.
        base=getattr(original,'base',None)
        if base is None or not hasattr(base,'flip_prob'):
            raise RuntimeError('Unknown TrainTransform wrapper: cannot disable flip for geometric audit')
        self.original_flip_prob=float(base.flip_prob)
        base.flip_prob=0.0
        if hasattr(base,'hsv_prob'):
            self.original_hsv_prob=float(base.hsv_prob)
            base.hsv_prob=0.0

    def __call__(self,image,targets,input_dim):
        before=np.asarray(targets).copy().reshape(-1,5)
        out_img,out_labels=self.original(image,targets,input_dim)
        result=box_matches_pre_to_post(before,out_labels,input_dim,image.shape[:2])
        result['source_shape']=list(image.shape[:2]);result['target_canvas']=list(input_dim)
        result['before_boxes']=before.tolist()  # audit-only, not serialized in report
        result['after_labels']=np.asarray(out_labels).tolist()
        self.traces.append(result)
        return out_img,out_labels


def summarize(traces, out_masks, feat_counts, batch_digest, ds, variant):
    counts=Counter();retained=Counter();by_class=defaultdict(lambda:Counter());class_keep=defaultdict(lambda:Counter())
    nofinal=0;limit_hit=0;input_boxes=0;output_boxes=0;match_errors=[]
    max_input=0;max_output=0
    for tr in traces:
        pre=np.asarray(tr['before_boxes'],dtype=np.float64).reshape(-1,5)
        input_boxes+=tr['pre_valid_count'];output_boxes+=tr['post_valid_count']
        max_input=max(max_input,tr['pre_valid_count']);max_output=max(max_output,tr['post_valid_count'])
        matched=set(range(len(pre)))-set(tr['unmatched_pre_indices'])
        # traced pre labels should have valid geometry; invalids are reported separately
        invalid=len(pre)-tr['pre_valid_count']
        nofinal+=invalid
        for i,row in enumerate(pre):
            if row[2]<=row[0] or row[3]<=row[1] or not np.isfinite(row).all():continue
            k=size_bin((row[2]-row[0])*tr['scale'],(row[3]-row[1])*tr['scale'])
            c=str(int(row[4]))
            counts[k]+=1;by_class[c][k]+=1
            if i in matched:
                retained[k]+=1;class_keep[c][k]+=1
        match_errors.append(float(tr['match_error_max']))
        if tr['post_valid_count']>=len(tr['after_labels']): limit_hit+=1
    return {
      'variant':variant,'samples_traced':len(traces),
      'mosaic_enabled':bool(getattr(ds,'enable_mosaic',False)),
      'mosaic_probability':float(getattr(ds,'mosaic_prob',0.0)),
      'final_transform_flip_disabled':True,
      'mosaic_affine_left_enabled':True,
      'pre_transform_valid_boxes':input_boxes,
      'post_transform_valid_boxes':output_boxes,
      'pre_invalid_boxes':nofinal,
      'max_pre_boxes_per_sample':max_input,
      'max_post_boxes_per_sample':max_output,
      'samples_exactly_filled_annotation_capacity':limit_hit,
      'retention_by_min_dimension':{
        k:{'pre':int(counts[k]),'matched':int(retained[k]),
           'rate':round(retained[k]/counts[k],6) if counts[k] else None}
        for k in ('<4px','4-8px','8-16px','16-32px','>=32px')},
      'retention_by_class':{
        c:{k:{'pre':int(n),'matched':int(class_keep[c][k])}
           for k,n in ctr.items()}
        for c,ctr in by_class.items()},
      'max_geometric_match_error_px':max(match_errors,default=0.0),
      'valid_pixel_fractions_min_max': [round(min(out_masks),5),round(max(out_masks),5)] if out_masks else None,
      'valid_anchor_counts_min_max':[min(feat_counts),max(feat_counts)] if feat_counts else None,
      'batch_image_digest_first':batch_digest,
    }


def run(args):
    import torch
    from exps.default.tde_yolox_s_pgsma import Exp
    from yolox.models.pgsma_validity import valid_anchors_from_image_mask
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    exp=Exp();exp.data_num_workers=0
    results={'status':'CHECKS_PASSED','seed':args.seed,'batches':args.batches,
             'batch_size':args.batch_size,'variants':{},'blockers':[],
             'warnings':[], 'scope':[
                 'Forensic final-TrainTransform survival, not source-to-final Mosaic GT survival',
                 'Final TrainTransform flip/HSV disabled for geometric matching; Mosaic affine retained',
                 'Mosaic synthetic background considered valid; masks are not pixel provenance',
                 'No performance, TensorRT, or 80-epoch claims']}
    for variant in args.variants:
        no_aug=variant=='no_aug'
        loader=exp.get_data_loader(args.batch_size,is_distributed=False,no_aug=no_aug,cache_img=False)
        ds=loader.dataset
        if type(ds).__name__!='MosaicDetection':
            raise RuntimeError('Expected MosaicDetection dataset wrapper')
        if variant=='forced_mosaic':
            if not hasattr(ds,'mosaic_prob'):raise RuntimeError('Cannot force mosaic on unknown wrapper')
            ds.mosaic_prob=1.0
            if not bool(getattr(ds,'enable_mosaic',False)):
                raise RuntimeError('Mosaic wrapper has enable_mosaic=False; cannot claim forced mosaic')
        if variant=='no_aug' and bool(getattr(ds,'enable_mosaic',False)):
            results['warnings'].append('No-Aug variant has enable_mosaic=True in dataset; sampler may disable it')
        trace=TracePreproc(ds.preproc,exp.input_size)
        ds.preproc=trace
        fractions=[];counts=[];fingerprint=None
        itr=iter(loader)
        for batch_index in range(args.batches):
            ims,targets,*_=next(itr)
            if ims.shape[1]!=4: raise AssertionError('Expected RGB+validity training batch')
            if fingerprint is None: fingerprint=hashlib.sha256(ims[:1].contiguous().numpy().tobytes()).hexdigest()[:24]
            masks=ims[:,3:4].float()
            fractions.extend(masks.mean((1,2,3)).tolist())
            b,_,h,w=ims.shape
            fmap=[torch.empty((b,1,h//s,w//s)) for s in (8,16,32)]
            counts.extend(valid_anchors_from_image_mask(masks,fmap).sum(-1).tolist())
            if not torch.isfinite(ims).all() or not torch.isfinite(targets).all():
                results['blockers'].append(f'{variant}: nonfinite image or GT label')
        if len(trace.traces)!=args.batches*args.batch_size:
            results['blockers'].append(f'{variant}: trace calls {len(trace.traces)} != expected {args.batches*args.batch_size}; inspect loader')
        report=summarize(trace.traces,fractions,counts,fingerprint,ds,variant)
        results['variants'][variant]=report
        if report['samples_exactly_filled_annotation_capacity']:
            results['blockers'].append(f'{variant}: annotation capacity reached')
        if report['max_geometric_match_error_px']>args.tolerance:
            results['blockers'].append(f'{variant}: geometric matched error beyond tolerance')
        for key,v in report['retention_by_min_dimension'].items():
            if v['pre'] and v['rate']<1:
                results['warnings'].append(f'{variant}: final-transform {key} retention {v["matched"]}/{v["pre"]}; inspect GT filters/matches')
        if variant=='forced_mosaic' and (not report['mosaic_enabled'] or report['mosaic_probability']!=1.0):
            results['blockers'].append('Forced mosaic not actually enabled')
    if results['blockers']:results['status']='BLOCKED'
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batches',type=int,default=24)
    ap.add_argument('--batch-size',type=int,default=2)
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--tolerance',type=float,default=1.5)
    ap.add_argument('--variants',nargs='+',choices=['no_aug','forced_mosaic','mixed_mosaic'],
                    default=['no_aug','forced_mosaic','mixed_mosaic'])
    ap.add_argument('--json',default='audits/annotation_survival.json')
    args=ap.parse_args()
    try:
        if args.batches<1 or args.batch_size<1:raise ValueError('Batch count must be positive')
        res=run(args)
    except Exception as e:
        res={'status':'BLOCKED','error':type(e).__name__+': '+str(e)}
    txt=json.dumps(res,indent=2,allow_nan=False)
    print(txt)
    p=Path(args.json);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(txt+'\n',encoding='utf8')
    sys.exit(0 if res['status']=='CHECKS_PASSED' else 2)


if __name__=='__main__':main()
