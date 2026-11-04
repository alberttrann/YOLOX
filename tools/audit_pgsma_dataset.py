"""Real COCO metadata + actual YOLOX loader preflight, no training.

Run from D:\\YOLOX-3rd as `python -m tools.audit_pgsma_dataset ...`.
Reports rather than hides augmentation, crowd and padding integration gaps.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

from ._pgsma_preflight_common import (dump_json, load_exp, image_gt_sizes,
                                       one_real_train_batch)


def inspect_coco_file(path, expected_classes=None, inspect_files_root=None, file_sample=16):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"COCO JSON missing: {path}")
    with path.open('r',encoding='utf-8') as f:
        obj = json.load(f)
    imgs = obj.get('images',[]); anns = obj.get('annotations',[]); cats = obj.get('categories',[])
    image_map = {im['id']:im for im in imgs}
    catids = {cat['id'] for cat in cats}
    cls = {cat['id']:cat.get('name','') for cat in cats}
    counts=Counter(); classes=Counter(); min_width=None; bad=[]
    small=[0,0]; crowd=0; zero_or_negative=0; outside=0; unknown=0
    for i, ann in enumerate(anns):
        if ann.get('iscrowd',0): crowd += 1
        g=ann.get('image_id'); c=ann.get('category_id')
        if g not in image_map:
            bad.append({'index':i,'error':'unknown_image','image_id':g})
            continue
        if c not in catids: unknown+=1
        xywh=ann.get('bbox',[])
        if len(xywh)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in xywh):
            bad.append({'index':i,'error':'invalid_bbox'});continue
        x,y,w,h=xywh
        if w<=0 or h<=0: zero_or_negative+=1;continue
        if x < -1e-4 or y < -1e-4 or x+w > image_map[g].get('width',0)+1e-4 or y+h > image_map[g].get('height',0)+1e-4:
            outside += 1
        if w<4 or h<4: small[0]+=1
        if w<8 or h<8: small[1]+=1
        if not ann.get('iscrowd',0):
            counts[g]+=1;classes[c]+=1
            min_width=w if min_width is None else min(min_width,w)
    top=counts.most_common(6)
    absent=[]
    if inspect_files_root:
        root=Path(inspect_files_root)
        for im in imgs[:file_sample]:
            name=str(im.get('file_name',''))
            if not name: continue
            possibilities=[root/name,root/'images'/name,root/'train2017'/name,root/'val2017'/name]
            if not any(p.is_file() for p in possibilities):
                absent.append(name)
    issues=[]
    if len(image_map)!=len(imgs):issues.append('duplicate image IDs in COCO JSON')
    if len(catids)!=len(cats):issues.append('duplicate category IDs')
    if expected_classes is not None and len(cats)!=expected_classes:
        issues.append(f'COCO taxonomy has {len(cats)} categories but Exp.num_classes={expected_classes}')
    if unknown:issues.append(f'{unknown} annotations have category IDs absent from categories')
    if bad:issues.append(f'{len(bad)} malformed or orphaned annotations')
    if zero_or_negative:issues.append(f'{zero_or_negative} nonpositive GT boxes')
    if outside:issues.append(f'{outside} out-of-image GT boxes')
    if absent:issues.append(f'{len(absent)} of sampled images unresolved beneath data_dir')
    if crowd:issues.append(f'{crowd} crowd annotations: current head does not propagate ignore masks')
    return {'path':str(path),'images':len(imgs),'annotations':len(anns),
            'categories':{str(i):name for i,name in sorted(cls.items())},
            'category_ids':sorted(catids),'crowd':crowd,
            'invalid_ann':len(bad)+zero_or_negative,'outside_boxes':outside,
            'under_4px_original':small[0],'under_8px_original':small[1],
            'max_non_crowd_annotations_per_image':max(counts.values(), default=0),
            'top_image_counts':top,'mean_non_crowd_annotations_per_image':round(sum(counts.values())/max(len(imgs),1),4),
            'sample_missing_image_files':absent[:10], 'issues':issues}


def audit(args):
    import torch
    if args.exp_module:
        import importlib
        exp=importlib.import_module(args.exp_module).Exp()
    else:
        exp=load_exp()
    exp.data_num_workers=0
    stats={'experiment':exp.exp_name,'input_size':list(exp.input_size),'configured_num_classes':int(exp.num_classes),
           'configured_mosaic_prob':float(exp.mosaic_prob),
           'configured_mixup_prob':float(exp.mixup_prob),'configured_allow_full_training':bool(exp.allow_full_training),
           'static':{},'loader':{},'blockers':[],'warnings':[]}
    cats_by_split={}
    for label,key in [('train','train_ann'),('clean_val','val_ann'),('adverse_test','test_ann')]:
        p=getattr(exp,key,None)
        if not p or not Path(p).is_file():
            stats['warnings'].append(f'{label} annotation file unavailable at {p!r}')
            continue
        res=inspect_coco_file(p,exp.num_classes if label=='train' else None,
                              exp.data_dir if label=='train' else None)
        stats['static'][label]=res
        cats_by_split[label]=res['categories']
        if label=='train' and res['invalid_ann']:
            stats['blockers'].append('Training annotations contain invalid/nonpositive boxes')
        if label=='train' and res['crowd']:
            stats['blockers'].append('Crowd labels present; missing ignore-region mask propagation must be resolved')
        if label=='train' and res['sample_missing_image_files']:
            stats['blockers'].append('Some source image paths do not resolve')
        for issue in res['issues']:
            stats['warnings'].append(f'{label}: {issue}')
    for name,cat in cats_by_split.items():
        if name!='train' and 'train' in cats_by_split and cat != cats_by_split['train']:
            stats['warnings'].append(f'{name}: taxonomy ID/name mapping differs from train; COCO AP comparison requires explicit mapping')
    # Report the raw path used by get_data_loader, including whether a MosaicDetection wrapper exists.
    for name, use_mosaic in [('no_aug',False),('mosaic_requested',True)]:
        if name=='mosaic_requested' and args.skip_mosaic:
            continue
        try:
            # Be explicit: a YoloBatchSampler(mosaic=True) alone is NOT a MosaicDetection wrapper.
            images,labels,full,loader=one_real_train_batch(exp,args.batch_size,mosaic=use_mosaic)
            ds=loader.dataset
            if images.shape[1]==4:
                validity=images[:,3:4].float()
                if not torch.isfinite(validity).all():
                    stats['blockers'].append(f'{name}: nonfinite validity mask')
                if bool(((validity<0)|(validity>1)).any()):
                    stats['blockers'].append(f'{name}: invalid mask outside [0,1]')
                if not bool((validity.sum(dim=(1,2,3))>0).all()):
                    stats['blockers'].append(f'{name}: sample has no valid pixels')
                valid_ratio=validity.mean(dim=(1,2,3)).tolist()
            else:
                valid_ratio=None
                if args.require_valid_masks:
                    stats['blockers'].append(f'{name}: missing 4th-channel validity mask')
            wrapper_type=type(ds).__name__
            has_wrapper=any(type(o).__name__=='MosaicDetection' for o in [ds,getattr(ds,'_dataset',None)])
            if use_mosaic and exp.mosaic_prob>0 and not has_wrapper:
                stats['blockers'].append('Mosaic configured but training loader is not using MosaicDetection; sampler flag alone is insufficient')
            if not torch.isfinite(images).all():stats['blockers'].append(f'{name}: nonfinite image batch')
            if not torch.isfinite(labels).all():stats['blockers'].append(f'{name}: nonfinite label batch')
            valid=(labels[...,3]>0)&(labels[...,4]>0)&torch.isfinite(labels).all(-1)
            c=labels[...,0][valid]
            if len(c) and (bool((c<0).any()) or bool((c>=exp.num_classes).any()) or bool((c!=c.round()).any())):
                stats['blockers'].append(f'{name}: label class index out of configured contiguous [0,K) range')
            if labels.shape[1]==0:stats['blockers'].append(f'{name}: zero padded annotation capacity')
            filled=valid.sum(-1)
            if bool((filled==labels.shape[1]).any()):
                stats['warnings'].append(f'{name}: at least one image exactly fills padded annotation capacity—possible silent truncation')
            h,w=images.shape[2:]
            gt_xy=labels[valid]
            outbounds=0
            if gt_xy.numel():
                outbounds=int(((gt_xy[:,1]-gt_xy[:,3]/2<-1e-2)|
                    (gt_xy[:,2]-gt_xy[:,4]/2<-1e-2)|
                    (gt_xy[:,1]+gt_xy[:,3]/2>w+1e-2)|
                    (gt_xy[:,2]+gt_xy[:,4]/2>h+1e-2)).sum())
            if outbounds:stats['warnings'].append(f'{name}: {outbounds} box(es) extend beyond batch canvas; inspect clipping policy')
            stats['loader'][name]={
                'dataset_type':wrapper_type,'contains_mosaic_wrapper':has_wrapper,
                'batch_shape':list(images.shape),'label_shape':list(labels.shape),
                'valid_annotations':int(valid.sum()),
                'per_image':image_gt_sizes(labels),
                'out_of_canvas_boxes':outbounds,
                'has_validity_mask':valid_ratio is not None,
                'valid_pixel_fractions':valid_ratio,
                'mosaic_flag':bool(use_mosaic)
            }
            # Contrast no_aug vs mosaic may be identical legitimately by chance; wrapper presence is decisive.
        except Exception as exc:
            stats['loader'][name]={'error':f'{type(exc).__name__}: {exc}'}
            stats['blockers'].append(f'{name} loader failed: {type(exc).__name__}: {exc}')
    if args.require_valid_masks:
        # The archived head has optional valid_anchor_mask but the top-level
        # PGSMAYOLOX.forward does not feed it. This is a structural gate, not
        # something images alone can certify.
        import inspect
        from yolox.models.pgsma_model import PGSMAYOLOX
        src=inspect.getsource(PGSMAYOLOX.forward)
        if 'valid_anchor_mask' not in src:
            stats['blockers'].append('Valid-content masks do not propagate from image/loader into native head')
    stats['status']='BLOCKED' if stats['blockers'] else 'CHECKS_PASSED'
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--exp-module',default=None,help='Alternate experiment module for opt-in Mosaic wrapper')
    p.add_argument('--skip-mosaic',action='store_true')
    p.add_argument('--require-valid-masks',action='store_true',default=False)
    p.add_argument('--json',default=None)
    args=p.parse_args()
    try:result=audit(args)
    except Exception as exc:
        result={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
    dump_json(result,args.json)
    sys.exit(2 if result['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
