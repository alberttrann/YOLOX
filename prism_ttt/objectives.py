"""GT-supervised OUTER loss only; NEVER called during TTT online inference.

Native matches are frozen via `NativeMatchReplay`. The original TDE losses are
reconstructed with exactly the native foreground normalizer. Additional ranking
and coverage proxies are source-label-only and do not alter the head itself.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from .fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity

SCALES=('small','medium','large')


def pair_iou_xywh(boxes:Tensor,gt:Tensor)->Tensor:
    # boxes [N,4], gt [4], cxcywh, IoU per candidate
    a0=boxes[:,:2]-boxes[:,2:].clamp_min(0)/2
    a1=boxes[:,:2]+boxes[:,2:].clamp_min(0)/2
    b0=gt[:2]-gt[2:].clamp_min(0)/2
    b1=gt[:2]+gt[2:].clamp_min(0)/2
    inter=(torch.minimum(a1,b1)-torch.maximum(a0,b0)).clamp_min(0).prod(-1)
    area_a=(a1-a0).clamp_min(0).prod(-1)
    area_b=(b1-b0).clamp_min(0).prod(-1)
    return inter/(area_a+area_b-inter+1e-8)


def _raw_centers(raw):
    shifts_x=raw['shifts_x'];shifts_y=raw['shifts_y'];strides=raw['strides']
    if not(len(shifts_x)==len(shifts_y)==len(strides)):
        raise RuntimeError('Unknown raw head grid signature')
    centers=[]
    for sx,sy,st in zip(shifts_x,shifts_y,strides):
        xx=sx.reshape(-1).float();yy=sy.reshape(-1).float();ss=st.reshape(-1).float()
        if xx.numel()!=yy.numel() or xx.numel()!=ss.numel():raise RuntimeError('Anchor-grid mismatch')
        centers.append(torch.stack(((xx+.5)*ss,(yy+.5)*ss),-1))
    return torch.cat(centers,0)


def ranking_and_coverage(raw,head,capture:NativeMatchReplay,*,topk:int=24,hardneg:int=64,fixed_hard_ids=None):
    """Per-GT recall proxy includes ALL GTs, even GTs with no SimOTA positive.

    Hard negatives are selected from detached native predictions for stable
    native/adapted comparison; call with `native_score_mask` precomputed if
    extending to a real batch with changing predicted hard-negative selection.
    """
    pred=raw['outputs'].float();labels=raw['labels'].float()
    centers=_raw_centers(raw).to(pred.device)
    if centers.shape[0]!=pred.shape[1]:raise RuntimeError('Prediction/grid count mismatch')
    hard_choices=[];per_gt=[];supports={s:0 for s in SCALES};gt_losses={s:[] for s in SCALES}
    for bi in range(pred.shape[0]):
        p=pred[bi];gts=labels[bi];gts=gts[(gts.sum(-1)>0)]
        matched=next((m for m in capture.stored if m.image_index==bi),None)
        fg=(matched.match[1].to(device=p.device,dtype=torch.bool) if matched is not None
            else torch.zeros(len(p),device=p.device,dtype=torch.bool))
        # Negatives selected using score DETACHED, preserving fixed hard-negative
        # semantics across native/adapted passes when provided precomputed indices.
        score=(p[:,4].sigmoid()*p[:,5:].sigmoid().amax(-1))
        neg_ids=torch.where(~fg)[0]
        if neg_ids.numel():
            if fixed_hard_ids is None:
                k=min(hardneg,int(neg_ids.numel()))
                hard_ids=neg_ids[torch.topk(score[neg_ids].detach(),k).indices]
            else:
                hard_ids=fixed_hard_ids[bi].to(p.device)
                if hard_ids.numel() and bool(fg[hard_ids].any()):raise RuntimeError('Fixed hard-negative set overlaps GT matches')
            neg_score=score[hard_ids]
        else:
            hard_ids=torch.empty((0,),device=p.device,dtype=torch.long)
            neg_score=p.new_zeros(0)
        hard_choices.append(hard_ids.detach())
        positive_ids=torch.where(fg)[0]
        gt_ids=(matched.match[3].to(p.device).long() if matched is not None else
                torch.empty((0,),device=p.device,dtype=torch.long))
        native_ious=(matched.match[2].to(p.device).float() if matched is not None else
                torch.empty((0,),device=p.device))
        lam=min(1.,max(0.,(int(head.current_epoch)-10)/40.))
        for gi,gt in enumerate(gts):
            cls=int(gt[0].item())
            if cls<0 or cls>=head.num_classes:raise RuntimeError('GT class invalid')
            box=gt[1:].float();scale=('small' if box[2]*box[3]<32**2 else
                                      'medium' if box[2]*box[3]<96**2 else 'large')
            supports[scale]+=1
            # FIXED geometry shortlist including unmatched GT, avoids solely
            # supervising the already-discovered native SimOTA positives.
            distance=((centers-box[:2]).square().sum(1))
            ids=distance.topk(min(topk,len(p)),largest=False).indices
            iou=pair_iou_xywh(p[ids,:4],box)
            correct=(p[ids,4].sigmoid()*p[ids,5+cls].sigmoid())
            quality=correct*iou
            # Smooth best-candidate quality in [0,1], stable under weak logits.
            soft=torch.softmax(quality/.1,dim=0)
            cover=1-(soft*quality).sum()
            # Positive ranking using the best GT-associated shortlist score.
            positive=(soft*correct).sum()
            if neg_score.numel():
                # Ranking promotes precision, not only positive-anchor BCE.
                rank=F.softplus((neg_score-positive+.05)*8).mean()/8
            else:rank=positive*0
            assigned=(gt_ids==gi)
            if bool(assigned.any()):
                ix=positive_ids[assigned]
                native_iou=native_ious[assigned]
                expected_cls=F.one_hot(torch.full((len(ix),),cls,dtype=torch.long,device=p.device),head.num_classes).float()*native_iou[:,None]
                obj_y=((1-lam)+lam*native_iou.clamp(.1,1))[:,None]
                gt_ref=box[None,:].expand(len(ix),-1)
                matched_box=2.5*head.iou_loss(p[ix,:4],gt_ref).mean()
                matched_cls=F.binary_cross_entropy_with_logits(p[ix,5:],expected_cls,reduction='none').sum(-1).mean()
                matched_obj=F.binary_cross_entropy_with_logits(p[ix,4:5],obj_y,reduction='none').mean()
                gt_task=matched_box+matched_cls+matched_obj
            else:
                gt_task=p.sum()*0.0
            gt_losses[scale].append((cover,rank,gt_task))
            per_gt.append((cover,rank,gt_task))
    zero=pred.sum()*0.0
    if per_gt:
        coverage=torch.stack([x[0] for x in per_gt]).mean()
        ranking=torch.stack([x[1] for x in per_gt]).mean()
    else:coverage=ranking=zero
    scale_proxy={s:(torch.stack([x[0]+x[1] for x in gt_losses[s]]).mean()
                     if supports[s] else None) for s in SCALES}
    scale_gt={s:(torch.stack([x[2] for x in gt_losses[s]]).mean()
                    if supports[s] else None) for s in SCALES}
    return {'coverage':coverage,'ranking':ranking,'scale_proxy':scale_proxy,'gt_support':supports,'hard_negative_ids':hard_choices,'scale_gt_detection':scale_gt}


def native_relative_loss(native:dict,adapted:dict,*,clean:bool=False,
                         background_weight:float=.5,tail_weight:float=.5,clean_weight:float=1.,
                         ranking_weight:float=.1,coverage_weight:float=.1,
                         source_margin:float=0.0)->tuple[Tensor,dict]:
    """Task-aligned constraint objective, same normalized TDE loss units."""
    a=adapted['replay'];n=native['replay']
    bg=a['components']['background_objectness']-n['components']['background_objectness'].detach()
    base=a['total']
    proxy=ranking_weight*adapted['proxy']['ranking']+coverage_weight*adapted['proxy']['coverage']
    native_proxy=ranking_weight*native['proxy']['ranking'].detach()+coverage_weight*native['proxy']['coverage'].detach()
    harm=[];cover={}
    for s in SCALES:
        if native['proxy']['gt_support'][s]<1:continue
        # Existing TDE sums per matched foreground / all positive anchors.
        # GT-scale guard use per-GT coverage/ranking + matched loss normalized
        # by matched positives in the group. Empty native matches still retain
        # the GT-coverage proxy.
        # GT-balanced supervised detection loss: average over GT instances,
        # not over all matched SimOTA anchor replicas.
        ad=adapted['proxy']['scale_gt_detection'][s]+adapted['proxy']['scale_proxy'][s]
        na=native['proxy']['scale_gt_detection'][s].detach()+native['proxy']['scale_proxy'][s].detach()
        harm.append(ad-na)
        cover[s]=int(native['proxy']['gt_support'][s])
    if harm:
        h=torch.stack(harm)
        # Smooth CVaR/top-tail surrogate over *present* GT scale groups.
        k=max(1,math.ceil(.5*len(h)))
        tail=torch.topk(h,k).values.clamp_min(source_margin).mean()
    else:tail=base*0
    clean_reg=(a['total']-n['total'].detach()).clamp_min(source_margin) if clean else base*0
    obj=base+proxy+background_weight*bg.clamp_min(source_margin)+tail_weight*tail+clean_weight*clean_reg
    return obj,{'total_delta':a['total']-n['total'].detach(),'background_delta':bg,
                'scale_tail':tail,'scale_support':cover,'clean_penalty':clean_reg,
                'ranking_delta':adapted['proxy']['ranking']-native['proxy']['ranking'].detach(),
                'coverage_delta':adapted['proxy']['coverage']-native['proxy']['coverage'].detach()}
