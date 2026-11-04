"""Deterministic CLEAN-source COCO subset evaluation. No adverse-set evaluation here.

AP/AR metrics are genuine COCOeval for the sampled image IDs, not full-set metrics.
Post-NMS per-min-side R@IoU=.5 is a separate diagnostic, *not* official COCO AR.
"""
from __future__ import annotations
import contextlib
import io
import math


def run_eval(model, exp, image_indices, device, mode, conf, nms, eval_batch_size=1):
    import torch
    from torch.utils.data import DataLoader, Subset
    from pycocotools.cocoeval import COCOeval
    from yolox.utils import postprocess
    from ._pgsma_pilot_core import group_scale, greedy_recall_pairs

    source_loader=exp.get_eval_loader(batch_size=1, is_distributed=False, testdev=False)
    dataset=source_loader.dataset
    if not hasattr(dataset, 'coco') or not hasattr(dataset, 'class_ids'):
        raise RuntimeError('COCODataset must expose coco metadata and class_ids')
    indices=list(image_indices)
    if not indices or max(indices)>=len(dataset):
        raise ValueError('Bad deterministic clean-val subset indices')
    loader=DataLoader(Subset(dataset,indices),batch_size=eval_batch_size,
                      shuffle=False,num_workers=0,pin_memory=False,
                      collate_fn=source_loader.collate_fn)
    category_ids=list(dataset.class_ids)
    was_training=model.training
    prior_mode=getattr(model.head,'inference_semantic_mode','native')
    model.eval();model.set_inference_mode(mode)
    detections=[];selected_ids=[]
    scale_totals={k:0 for k in ('lt4','4to8','8to16','16to32','ge32')}
    scale_hits={k:0 for k in scale_totals}
    det_count=0;image_count=0
    try:
        with torch.inference_mode():
            for batch in loader:
                imgs, _, info, ids=batch[:4]
                imgs=imgs.to(device).float()
                if imgs.shape[1]!=3:
                    raise ValueError('Clean-validation loader expected RGB')
                raw=model(imgs)
                if not torch.isfinite(raw).all():
                    raise FloatingPointError('Nonfinite clean-validation inference')
                decoded=postprocess(raw.clone(),int(exp.num_classes),float(conf),float(nms),class_agnostic=False)
                for j,pred in enumerate(decoded):
                    image_id=int(ids[j].item() if hasattr(ids[j],'item') else ids[j])
                    original_h=float(info[0][j]);original_w=float(info[1][j])
                    scale=min(float(exp.test_size[0])/original_h,float(exp.test_size[1])/original_w)
                    image_count+=1;selected_ids.append(image_id)
                    gt_anns=dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[image_id],iscrowd=False))
                    gt=[]
                    for ann in gt_anns:
                        cat_id=int(ann['category_id'])
                        if cat_id not in category_ids:continue
                        x,y,w,h=map(float,ann['bbox'])
                        if w<=0 or h<=0:continue
                        gt.append((cat_id,(x,y,x+w,y+h),0.0))
                        scale_totals[group_scale(min(w,h)*scale)]+=1
                    per_img=[]
                    if pred is not None:
                        for row in pred.detach().cpu().tolist():
                            x1,y1,x2,y2,obj,cls_conf,class_ix=row[:7]
                            class_ix=int(class_ix)
                            if not (0<=class_ix<len(category_ids)):
                                raise RuntimeError('Class category map mismatch')
                            cat_id=int(category_ids[class_ix]);confidence=float(obj*cls_conf)
                            x1=max(0.0,min(original_w,float(x1)/scale));y1=max(0.0,min(original_h,float(y1)/scale))
                            x2=max(0.0,min(original_w,float(x2)/scale));y2=max(0.0,min(original_h,float(y2)/scale))
                            if x2<=x1 or y2<=y1:continue
                            detections.append({'image_id':image_id,'category_id':cat_id,
                                    'bbox':[x1,y1,x2-x1,y2-y1],'score':confidence})
                            per_img.append((cat_id,(x1,y1,x2,y2),confidence))
                    det_count+=len(per_img)
                    hits=greedy_recall_pairs(gt,per_img,threshold=.5)
                    for i in hits:
                        bbox=gt[i][1]
                        scale_hits[group_scale(min(bbox[2]-bbox[0],bbox[3]-bbox[1])*scale)]+=1
    finally:
        model.set_inference_mode(prior_mode)
        model.train(was_training)
    if image_count!=len(indices):
        raise RuntimeError('Clean validation subset did not yield expected number of images')
    if len(set(selected_ids))!=len(selected_ids):
        raise RuntimeError('Repeated image ID in clean-validation subset')
    scale_recall={k: {'matched':scale_hits[k],'total_gt':scale_totals[k],
                      'recall_iou50':scale_hits[k]/scale_totals[k] if scale_totals[k] else None}
                  for k in scale_totals}
    result={'images':image_count,'image_ids':selected_ids,'detections':det_count,
            'post_nms_recall_iou50_by_effective_min_side':scale_recall,
            'note':'Source-clean COCO image SUBSET only; per-min-side R@.5 is NOT COCO AR.'}
    if not detections:
        result['coco_metrics']=None
        result['warning']='No detections passed NMS; COCO AP/AR undefined for this audit.'
        return result
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        coco_dt=dataset.coco.loadRes(detections)
        evaluator=COCOeval(dataset.coco,coco_dt,'bbox')
        evaluator.params.imgIds=selected_ids
        evaluator.params.catIds=category_ids
        evaluator.params.maxDets=[1,10,100]
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    stats=evaluator.stats
    result['coco_metrics']={'AP50_95':float(stats[0]),'AP50':float(stats[1]),
                            'AP75':float(stats[2]),'AR100':float(stats[8]),
                            'AP_small_COCO':float(stats[3]),
                            'AP_medium_COCO':float(stats[4]),
                            'AP_large_COCO':float(stats[5])}
    return result
