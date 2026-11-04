"""Frozen PG-SMA inference sweep with disjoint calibration/confirmation splits.

No optimizer, backward, target-domain inputs, or learned-parameter changes.
Runs one PyTorch inference per image, reusing EXACT native logits and candidate/gate
outputs before decoding/NMS for a paired score-only intervention.

python -m tools.audit_pgsma_correction_sweep --help
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from ._pgsma_pilot_core import atomic_json, file_digest, group_scale, greedy_recall_pairs
from .pgsma_bounded_pilot import safe_checkpoint_state

SIZES = ('lt4', '4to8', '8to16', '16to32', 'ge32')
METRICS = ('AP50_95', 'AP50', 'AP75', 'AR100', 'AP_small_COCO', 'AP_medium_COCO', 'AP_large_COCO')


def parse_alphas(text):
    result = sorted(set(float(s.strip()) for s in text.split(',')))
    if not result or result[0] <= 0.0 or result[-1] > 1.5 or any(not math.isfinite(v) for v in result):
        raise ValueError('Alphas must be in (0,1.5], comma separated')
    if len(result) > 8:
        raise ValueError('At most 8 nonzero alpha strengths')
    return result


def config_key(alpha, gate):
    if alpha == 0:
        return 'native'
    # Stable, lossless form for multiples of .01, and no decimal ambiguity.
    if abs(round(alpha * 1000) - alpha * 1000) > 1e-8:
        raise ValueError('Alpha must be a multiple of 0.001')
    return ('gated' if gate else 'ungated') + '_' + ('%04d' % int(round(alpha * 1000)))


def configurations(alphas):
    specs = [('native', 0.0, True)]
    for gate in (True, False):
        specs += [(config_key(a,gate),a,gate) for a in alphas]
    return specs


def select_splits(total, excluded, calibration_images, confirmation_images, seed):
    excluded = set(int(i) for i in excluded)
    if excluded and (min(excluded) < 0 or max(excluded) >= total):
        raise ValueError('Excluded manifest indices outside validation dataset')
    available = [i for i in range(total) if i not in excluded]
    if len(available) < calibration_images + confirmation_images:
        raise ValueError('Not enough clean validation images after excluding prior pilot images')
    rng = np.random.RandomState(seed)
    chosen = rng.permutation(np.asarray(available, dtype=np.int64))
    a = sorted(int(i) for i in chosen[:calibration_images])
    b = sorted(int(i) for i in chosen[calibration_images:calibration_images+confirmation_images])
    if set(a)&set(b) or set(a)&excluded or set(b)&excluded:
        raise AssertionError('Split leakage')
    return a,b


def state_hash(model):
    h=hashlib.sha256()
    for name, val in sorted(model.state_dict().items()):
        h.update(name.encode('utf8'))
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def capture_native_and_semantics(model, images):
    """Exactly one model forward; native decoded boxes/objectness, original cls logits,
    raw semantic correction and learned gate are captured before NMS.

    Temporary method replacement is restored on errors and never persisted.
    """
    engine=model.head.pgsma_engine
    original=engine.forward_levels
    previous_mode=getattr(model.head,'inference_semantic_mode','native')
    capture={}

    def intercept(features, native_logits, objectness, mode='pgsma', accept=True, collect=False):
        if mode!='pgsma' or len(features)!=3:
            raise RuntimeError('Unexpected PG-SMA feature flow during frozen inference')
        deltas=[]; gates=[]
        for i,(feature,cls,obj) in enumerate(zip(features,native_logits,objectness)):
            values=engine._candidate(i,feature.detach(),cls.detach(),obj.detach(),mode='pgsma')
            deltas.append(values['delta'].detach())
            gates.append(values['gate'].detach())
        capture['native_logits']=engine._flatten(native_logits).detach()
        capture['delta']=engine._flatten(deltas).detach()
        capture['gate']=engine._flatten(gates).detach()
        return native_logits  # preserve native logits for raw decoding

    engine.forward_levels=intercept
    model.set_inference_mode('pgsma')
    try:
        with torch.inference_mode():
            native=model(images)
    finally:
        engine.forward_levels=original
        model.set_inference_mode(previous_mode)
    if set(capture) != {'native_logits','delta','gate'}:
        raise RuntimeError('Did not capture semantic logits, delta and gate')
    if native.shape[-1]!=5+capture['native_logits'].shape[-1] or native.shape[:2]!=capture['delta'].shape[:2]:
        raise RuntimeError('Anchor/class logit layout mismatch')
    if capture['native_logits'].shape!=capture['delta'].shape or capture['gate'].shape[-1]!=1:
        raise RuntimeError('Semantic correction shape mismatch')
    if not all(torch.isfinite(x).all() for x in (native,*capture.values())):
        raise FloatingPointError('Nonfinite frozen inference output')
    if not torch.allclose(native[...,5:], capture['native_logits'].sigmoid(),atol=1e-6,rtol=1e-6):
        raise RuntimeError('Native class-probability reconstruction is not exact')
    return native,capture


def apply_correction(native, captured, alpha, gated):
    result=native.clone()
    if alpha!=0:
        shift=captured['delta']
        if gated:shift=shift*captured['gate']
        corrected=(captured['native_logits']+float(alpha)*shift).sigmoid()
        result[...,5:]=corrected.to(result.dtype)
    return result


def model_parity_check(model, imgs, native, capture, atol=2e-6):
    """Verify the frozen sweep reproduces the unmodified installed forward path."""
    previous=getattr(model.head,'inference_semantic_mode','native')
    try:
        model.set_inference_mode('native')
        with torch.inference_mode(): direct_native=model(imgs)
        model.set_inference_mode('pgsma')
        with torch.inference_mode(): direct_pgsma=model(imgs)
    finally:
        model.set_inference_mode(previous)
    reconstructed=apply_correction(native,capture,1.0,True)
    err_native=float((direct_native-native).abs().max())
    err_pgsma=float((direct_pgsma-reconstructed).abs().max())
    if err_native > atol or err_pgsma>atol:
        raise RuntimeError(f'Sweep changed native/PG-SMA inference behavior: {err_native}, {err_pgsma}')
    return {'native_max_abs':err_native, 'pgsma_gated_1_max_abs':err_pgsma,'tolerance':atol}


def scaled_gt(dataset, image_id, scale, cats):
    ann_ids=dataset.coco.getAnnIds(imgIds=[int(image_id)],iscrowd=False)
    anns=dataset.coco.loadAnns(ann_ids)
    out=[]
    for ann in anns:
        cat=int(ann['category_id'])
        if cat not in cats:continue
        x,y,w,h=map(float,ann['bbox'])
        if w<=0 or h<=0:continue
        out.append({'category_id':cat, 'box':[x,y,x+w,y+h],
                    'scaled_box':[x*scale,y*scale,(x+w)*scale,(y+h)*scale],
                    'size':group_scale(min(w,h)*scale),'ann_id':int(ann['id'])})
    return out


def bbox_iou_tensor(pred_xywh, gt_xyxy):
    """[N,4] xywh center -> [N,G] exact box IoU, on the same image canvas."""
    n=pred_xywh.shape[0]
    if not gt_xyxy:
        return pred_xywh.new_zeros((n,0))
    p=torch.cat([pred_xywh[:,:2]-pred_xywh[:,2:4]/2,
                 pred_xywh[:,:2]+pred_xywh[:,2:4]/2],dim=-1)
    g=pred_xywh.new_tensor(gt_xyxy)
    lu=torch.maximum(p[:,None,:2],g[None,:,:2])
    rd=torch.minimum(p[:,None,2:],g[None,:,2:])
    inter=(rd-lu).clamp_min(0).prod(-1)
    a=(p[:,2:]-p[:,:2]).clamp_min(0).prod(-1)
    b=(g[:,2:]-g[:,:2]).clamp_min(0).prod(-1)
    return inter/(a[:,None]+b[None,:]-inter).clamp_min(1e-9)


def pre_nms_gt_support(raw, ious, gt, cats, conf):
    """Class-aware score/IoU support before NMS (not necessarily final GT matching)."""
    if not gt:return []
    class_max,class_ix=raw[0,:,5:].max(-1)
    combined=raw[0,:,4]*class_max
    scores_ok=combined > conf
    top_class=class_ix.long()
    output=[]
    for j,ann in enumerate(gt):
        c=cats.index(ann['category_id'])
        output.append(bool(((ious[:,j]>=0.5)&scores_ok&(top_class==c)).any()))
    return output


def gate_groups(native,capture,ious,gt,cats):
    """Anchor-conditioned gate observations, NOT a causal loss attribution."""
    gate=capture['gate'][0,:,0]
    delta=capture['delta'][0]
    obj=native[0,:,4]
    stats={}
    def add(key, mask, class_index=None):
        if not mask.any():return
        vals=gate[mask]
        val={'anchors':int(mask.sum()), 'gate_mean':float(vals.mean()),
             'gate_min':float(vals.min()), 'gate_ge_0p95':float((vals>=.95).float().mean())}
        if class_index is not None:
            corr=delta[mask,class_index[mask]]
            val['matched_gt_class_delta_mean']=float(corr.mean())
            val['matched_gt_class_delta_positive_fraction']=float((corr>0).float().mean())
        stats[key]=val
    add('all',torch.ones_like(gate,dtype=torch.bool))
    if ious.shape[1]:
        best_iou, best_gt=ious.max(-1)
        close=best_iou>=.5
        cls_ids=torch.tensor([cats.index(g['category_id']) for g in gt],device=gate.device)
        best_cls=cls_ids[best_gt]
        add('gt_overlap_iou_ge_0p5',close,best_cls)
        for size in SIZES:
            gt_mask=torch.tensor([g['size']==size for g in gt],device=gate.device)
            add('gt_overlap_'+size,close & gt_mask[best_gt],best_cls)
        background=best_iou<.1
    else:
        background=torch.ones_like(gate,dtype=torch.bool)
    add('background_iou_lt_0p1',background)
    if background.any():
        ids=torch.nonzero(background,as_tuple=False).flatten()
        chosen=ids[obj[ids].topk(min(256,len(ids))).indices]
        mask=torch.zeros_like(gate,dtype=torch.bool);mask[chosen]=True
        add('hard_background_top_objectness',mask)
    return stats


def aggregate_groups(group_dicts):
    """Anchor-count weighted, not image-averaged gate telemetry."""
    accumulator={}
    for groups in group_dicts:
        for key,d in groups.items():
            current=accumulator.setdefault(key,{'anchors':0})
            n=d['anchors'];current['anchors']+=n
            for fld,val in d.items():
                if fld!='anchors':current[fld]=current.get(fld,0)+n*val
    for key,d in accumulator.items():
        n=d['anchors']
        for fld in list(d):
            if fld!='anchors':d[fld]=d[fld]/n if n else None
    return accumulator


def evaluate_coco(dataset,detections,image_ids,cats):
    from pycocotools.cocoeval import COCOeval
    if not detections:return None
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        result=dataset.coco.loadRes(detections)
        coco=COCOeval(dataset.coco,result,'bbox')
        coco.params.imgIds=list(image_ids)
        coco.params.catIds=list(cats)
        coco.params.maxDets=[1,10,100]
        coco.evaluate(); coco.accumulate();coco.summarize()
    s=coco.stats
    return {
        'AP50_95':float(s[0]),'AP50':float(s[1]),'AP75':float(s[2]),
        'AR100':float(s[8]),'AP_small_COCO':float(s[3]),
        'AP_medium_COCO':float(s[4]),'AP_large_COCO':float(s[5])}


def postprocess_one(raw, gt, image_id, original_h, original_w, scale, cats, conf, nms):
    from yolox.utils import postprocess
    output=postprocess(raw.clone(),len(cats),float(conf),float(nms),class_agnostic=False)[0]
    detections=[];pairs=[]
    if output is not None:
        for row in output.detach().cpu().tolist():
            x1,y1,x2,y2,obj,cls_conf,c=row[:7]
            c=int(c)
            if not 0<=c<len(cats):raise RuntimeError('Invalid class id after NMS')
            x1=max(0.0,min(original_w,x1/scale)); x2=max(0.0,min(original_w,x2/scale))
            y1=max(0.0,min(original_h,y1/scale));y2=max(0.0,min(original_h,y2/scale))
            if x2<=x1 or y2<=y1:continue
            score=float(obj*cls_conf);category=int(cats[c])
            detections.append({'image_id':int(image_id),'category_id':category,
                               'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
            pairs.append((category,(x1,y1,x2,y2),score))
    gt_pairs=[(g['category_id'],g['box'],0.0) for g in gt]
    return detections,greedy_recall_pairs(gt_pairs,pairs,threshold=.5)


def summarize_split(dataset,mode,records,cats,threshold=.01,bootstrap=1000,seed=42):
    image_ids=[im['image_id'] for im in records]
    outcomes={}
    totals={s:sum(sum(gt['size']==s for gt in im['gt']) for im in records) for s in SIZES}
    native='native'
    keys=list(records[0]['predictions'])
    for key in keys:
        data=[]; gt_counts={s:[] for s in SIZES};hits={s:[] for s in SIZES}
        gained={s:0 for s in SIZES};lost={s:0 for s in SIZES}
        lost_examples=[]
        for im in records:
            pred=im['predictions'][key]
            data.extend(pred['detections'])
            found=set(pred['hits']);base=set(im['predictions']['native']['hits'])
            for size in SIZES:
                gt_counts[size].append(sum(g['size']==size for g in im['gt']))
                hits[size].append(sum(im['gt'][i]['size']==size for i in found))
            for i,g in enumerate(im['gt']):
                if i in found and i not in base:gained[g['size']]+=1
                if i not in found and i in base:
                    lost[g['size']]+=1
                    if len(lost_examples)<60:
                        before=im['predictions'][key]['pre_nms_support'][i]
                        lost_examples.append({'image_id':im['image_id'],'annotation_id':g['ann_id'],
                            'category_id':g['category_id'],'effective_size':g['size'],
                            'likely_stage':'nms_or_score_ordering' if before else 'class_or_confidence_before_nms'})
        metrics=evaluate_coco(dataset,data,image_ids,cats)
        matched={s:int(sum(hits[s])) for s in SIZES}
        outcomes[key]={'coco_metrics':metrics,'detections':len(data),
            'effective_scale_recall_iou50':{s:{'total_gt':totals[s],'matched':matched[s],
               'recall':matched[s]/totals[s] if totals[s] else None,
               'gained_vs_native':gained[s],'lost_vs_native':lost[s]} for s in SIZES},
            'lost_examples_vs_native':lost_examples}
        outcomes[key]['_arrays']={'gt_counts':gt_counts,'hits':hits}
    if 'native' not in outcomes:raise AssertionError('Reference native prediction is required')
    base=outcomes['native']
    for key,val in outcomes.items():
        if val['coco_metrics'] and base['coco_metrics']:
            val['delta_coco_vs_native']={m:val['coco_metrics'][m]-base['coco_metrics'][m] for m in METRICS}
        else:val['delta_coco_vs_native']=None
        if key!='native':
            val['recall_paired_bootstrap95']=paired_bootstrap_ci(
                base['_arrays']['hits'],val['_arrays']['hits'],base['_arrays']['gt_counts'],bootstrap,seed)
    for val in outcomes.values():
        val.pop('_arrays')
    return {'split':mode,'images':len(records),'image_ids':image_ids,'gt_by_effective_size':totals,
            'gate_diagnostics':aggregate_groups([im['gate_telemetry'] for im in records]),
            'variants':outcomes}


def paired_bootstrap_ci(native_hits,trial_hits,gt_counts,n_boot,seed):
    rng=np.random.RandomState(seed)
    result={}
    for size in SIZES:
        n=np.asarray(native_hits[size],dtype=np.float64)
        t=np.asarray(trial_hits[size],dtype=np.float64)
        denom=np.asarray(gt_counts[size],dtype=np.float64)
        if not denom.sum():
            result[size]=None;continue
        draws=rng.randint(0,len(n),size=(n_boot,len(n)))
        d=denom[draws].sum(-1)
        dif=(t[draws]-n[draws]).sum(-1)
        with np.errstate(divide='ignore',invalid='ignore'):
            vals=dif/np.maximum(d,1)
        vals=vals[d>0]
        result[size]={'difference_point':float((t-n).sum()/denom.sum()),
                      'ci95':[float(x) for x in np.percentile(vals,[2.5,97.5])] if len(vals) else None,
                      'n_boot':int(n_boot)}
    return result


def eligible_candidate(result,baseline,ap_tolerance=.005,recall_tolerance=.005,min_gt_for_recall=30):
    """Pre-register *all adequately sampled scales*, not just the failing 8-16 bin.

    The <4px group usually has too few GTs for a useful nonregression test;
    that must remain a stated evidence gap, never a reported pass.
    """
    trial=result['coco_metrics'];base=baseline['coco_metrics']
    if not trial or not base:return False,['missing_coco_metrics']
    if any(base[k] < 0 for k in ('AP_small_COCO','AP_medium_COCO','AP_large_COCO')):
        return False,['undefined_size_AP_in_calibration']
    reasons=[]
    for metric,tolerance in (
         ('AP50_95',0.0),('AR100',0.0),
         ('AP_small_COCO',ap_tolerance),('AP_medium_COCO',ap_tolerance),
         ('AP_large_COCO',ap_tolerance)):
        if trial[metric] < base[metric]-tolerance: reasons.append(metric+'_regression')
    for size in SIZES:
        base_size=baseline['effective_scale_recall_iou50'].get(size)
        new_size=result['effective_scale_recall_iou50'].get(size)
        if not base_size or not new_size:
            reasons.append(size+'_missing')
            continue
        n=int(base_size.get('total_gt',0))
        if n >= min_gt_for_recall:
            r0=base_size.get('recall');r1=new_size.get('recall')
            if r0 is None or r1 is None or r1 < r0-recall_tolerance:
                reasons.append(size+'_R0p5_regression')
    return len(reasons)==0,reasons


def select_candidate(cal_report):
    trials=cal_report['variants'];base=trials['native'];rows=[]
    for key,item in trials.items():
        if key=='native':continue
        eligible,reasons=eligible_candidate(item,base)
        metrics=item['delta_coco_vs_native']
        score=(metrics['AP50_95']+.5*metrics['AR100']+.1*metrics['AP75']) if metrics else -float('inf')
        rows.append({'key':key,'eligible':eligible,'reasons':reasons,'objective_delta':score})
    winners=[r for r in rows if r['eligible']]
    winners.sort(key=lambda r:(-r['objective_delta'],r['key']))
    selected=winners[0]['key'] if winners else 'native'
    return {'selected':selected,'rules':'AP and AR nonregression; COCO small/medium/large AP >= native-.005; '
                 'effective-bin R@0.5 >= native-.005 for each bin with >=30 GTs; '
                 'maximize dAP+.5*dAR+.1*dAP75',
            'rankings':rows,'fallback_to_native':not bool(winners)}


def run_split(model,exp,dataset,indices,device,specs,conf,nms,split_name):
    from torch.utils.data import DataLoader,Subset
    base_loader=exp.get_eval_loader(batch_size=1,is_distributed=False,testdev=False)
    loader=DataLoader(Subset(dataset,indices),batch_size=1,shuffle=False,num_workers=0,
                      pin_memory=False,collate_fn=base_loader.collate_fn)
    cats=list(dataset.class_ids)
    records=[];first_parity=None
    with torch.inference_mode():
        for batch_num,batch in enumerate(loader):
            imgs,_,info,ids=batch[:4]
            imgs=imgs.to(device).float()
            if imgs.shape[1]!=3:raise RuntimeError('Clean-val inference expects 3-channel RGB')
            image_id=int(ids[0].item() if hasattr(ids[0],'item') else ids[0])
            h=float(info[0][0]);w=float(info[1][0])
            scale=min(exp.test_size[0]/h,exp.test_size[1]/w)
            gt=scaled_gt(dataset,image_id,scale,cats)
            native,capture=capture_native_and_semantics(model,imgs)
            if first_parity is None:
                first_parity=model_parity_check(model,imgs,native,capture)
            ious=bbox_iou_tensor(native[0,:,:4],[g['scaled_box'] for g in gt])
            rec={'image_id':image_id,'gt':gt,'predictions':{},
                 'gate_telemetry':gate_groups(native,capture,ious,gt,cats)}
            for key,alpha,gate in specs:
                adjusted=apply_correction(native,capture,alpha,gate)
                detections,hits=postprocess_one(adjusted,gt,image_id,h,w,scale,cats,conf,nms)
                rec['predictions'][key]={'hits':sorted(hits),'detections':detections,
                    'pre_nms_support':pre_nms_gt_support(adjusted,ious,gt,cats,conf)}
            records.append(rec)
            if (batch_num+1)%32==0:
                print(f'[{split_name}] images {batch_num+1}/{len(indices)}',flush=True)
    if len(records)!=len(indices):raise RuntimeError('Missing validation image')
    return summarize_split(dataset,split_name,records,cats), first_parity


def run(args):
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA not available; choose --device cpu for a very slow audit')
    alphas=parse_alphas(args.alphas)
    if 1.0 not in alphas:raise ValueError('Alpha list must contain 1.0 for original checkpoint parity')
    if args.calibration_images<16 or args.confirmation_images<16 or args.calibration_images+args.confirmation_images>512:
        raise ValueError('Calibration/confirmation both >=16; total at most 512')
    if not (100<=args.bootstrap<=5000):raise ValueError('Bootstrap samples must be 100..5000')
    if not(0<args.conf<1 and 0<args.nms<1):raise ValueError('conf/nms must be between 0 and 1')
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing nonempty results directory')
    excluded=set()
    excluded_manifests=[]
    for path in args.exclude_manifest:
        source=Path(path)
        if not source.is_file():
            raise FileNotFoundError('Required prior pilot manifest: '+str(source))
        info=json.loads(source.read_text(encoding='utf8'))
        idx=info.get('eval_indices')
        if not isinstance(idx,list) or not idx:
            raise RuntimeError('Exclude manifest has no eval_indices: '+str(source))
        excluded.update(int(i) for i in idx)
        excluded_manifests.append({'path':str(source),'sha256':file_digest(source),
                                   'indices_count':len(idx)})
    from exps.default.tde_yolox_s_pgsma import Exp
    exp=Exp()
    if exp.allow_full_training:raise RuntimeError('Full training guard was unexpectedly unlocked')
    exp.ema=False;exp.data_num_workers=0
    model=exp.get_model()
    checkpoint,weights=safe_checkpoint_state(args.checkpoint)
    meta=checkpoint.get('pgsma_pilot')
    if not isinstance(meta,dict) or meta.get('variant') not in ('pgsma_simota','pgsma') or int(meta.get('successful_semantic_updates',0))<1:
        raise RuntimeError('Expected trained PG-SMA bounded-pilot checkpoint')
    model.load_state_dict(weights,strict=True)
    device=torch.device(args.device)
    model.to(device).eval()
    source_loader=exp.get_eval_loader(batch_size=1,is_distributed=False,testdev=False)
    dataset=source_loader.dataset
    if not hasattr(dataset,'coco') or not hasattr(dataset,'class_ids'):
        raise RuntimeError('Expected clean-source COCO validation dataset')
    cal_indices,confirm_indices=select_splits(len(dataset),excluded,args.calibration_images,args.confirmation_images,args.seed)
    out.mkdir(parents=True,exist_ok=True)
    before=state_hash(model)
    all_specs=configurations(alphas)
    manifest={'status':'RUNNING','checkpoint_path':args.checkpoint,'checkpoint_sha256':file_digest(args.checkpoint),
        'checkpoint_pilot_metadata':meta,'torch_version':torch.__version__,
        'eval_ann_path':str(exp.val_ann),
        'eval_ann_sha256':file_digest(exp.val_ann) if Path(exp.val_ann).is_file() else None,
        'exclude_manifests':excluded_manifests,
        'excluded_prior_eval_indices':len(excluded),
        'calibration_indices':cal_indices,'confirmation_indices':confirm_indices,
        'seed':args.seed,'alphas':alphas,'conf':args.conf,'nms':args.nms,
        'pre_registered_selection':('AP+AR nonregression; <=0.005 COCO small/medium/large AP declines; '
          '<=0.005 R@.5 decline for each effective-size bin with >=30 GTs; objective dAP+.5*dAR+.1*dAP75'),
        'scope':'clean source only; no training; inference-only; no adverse-set tuning'}
    atomic_json(out/'manifest.json',manifest)
    cal,check=run_split(model,exp,dataset,cal_indices,device,all_specs,args.conf,args.nms,'calibration')
    cal['first_image_direct_forward_parity']=check
    atomic_json(out/'calibration.json',cal)
    selected=select_candidate(cal)
    atomic_json(out/'selection.json',selected)
    confirm_specs=[('native',0.0,True)]
    for key,alpha,gated in all_specs:
        if key in (selected['selected'],'gated_1000') and key!='native':
            confirm_specs.append((key,alpha,gated))
    confirm_specs=list({s[0]:s for s in confirm_specs}.values())
    confirm,parity=run_split(model,exp,dataset,confirm_indices,device,confirm_specs,args.conf,args.nms,'confirmation')
    confirm['first_image_direct_forward_parity']=parity
    atomic_json(out/'confirmation.json',confirm)
    after=state_hash(model)
    if before!=after:raise RuntimeError('Frozen model state changed during inference sweep!')
    result={'status':'CHECKS_PASSED','checkpoint_sha256':manifest['checkpoint_sha256'],
       'model_state_unchanged':True,'model_state_sha256':before,
       'calibration_images':len(cal_indices),'confirmation_images':len(confirm_indices),
       'selection':selected,'confirmation_native':confirm['variants']['native']['coco_metrics'],
       'confirmation_selected':confirm['variants'][selected['selected']]['coco_metrics'],
       'confirmation_original_gated_1':confirm['variants']['gated_1000']['coco_metrics'],
       'limitations':['One checkpoint and one deterministic clean-source split',
         'No adverse-domain evidence; do not tune on the adverse test set',
         '96 earlier pilot images excluded, but clean validation remains source-domain',
         'Per-image paired bootstrap covers scale recall only, not COCO AP confidence interval',
         'COCO small AP is area-based and differs from effective min-side scale recall',
         'Pre-NMS support classification is diagnostic, not proof of a unique NMS cause',
         'No training, no parameter updates, no full 80-epoch permission']}
    atomic_json(out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--exclude-manifest',required=True,action='append',
                   help='Repeat for each previous pilot eval manifest; excluded indices are unioned')
    p.add_argument('--calibration-images',type=int,default=128)
    p.add_argument('--confirmation-images',type=int,default=128)
    p.add_argument('--alphas',default='0.25,0.5,0.75,1.0')
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--conf',type=float,default=0.01)
    p.add_argument('--nms',type=float,default=0.60)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--out-dir',required=True)
    args=p.parse_args()
    try:
        result=run(args)
        print(json.dumps({'status':result['status'],'model_state_unchanged':True,
             'selected':result['selection']['selected'],
             'confirmation_native':result['confirmation_native'],
             'confirmation_selected':result['confirmation_selected']},indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','error':type(exc).__name__+': '+str(exc)},indent=2))
        return 2

if __name__=='__main__':sys.exit(main())
