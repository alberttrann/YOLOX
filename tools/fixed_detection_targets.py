"""Strict native-match capture/replay and TDE_Head component accounting.

Works against the vNext.2 TDE_Head signature in the supplied source dump.
Refuses changed signatures, changed GT, changed head loss semantics, or different
anchor counts. No trainable weights are installed or mutated by this module.

This is a *local-source supervised diagnostic*: it uses BDD CLEAN TRAIN labels.
Never call on ACDC as a tuning signal. OOD does not provide target labels at TTT.
"""
from __future__ import annotations

import contextlib
import inspect
import math
import re
from types import MethodType
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class NativeMatch:
    image_index: int
    gtboxes: torch.Tensor
    gtclasses: torch.Tensor
    match: tuple[Any,...]


def _clone_matched(out):
    if len(out)!=5:
        raise RuntimeError(f'Unsupported get_assignments return signature, expected 5, got {len(out)}')
    return tuple(x.detach().clone() if isinstance(x,torch.Tensor) else x for x in out)


class NativeMatchReplay:
    """Decorates instance-local get_assignments; restores it exactly on exit.

    Capture and replay requires both forward calls to have the SAME image batch
    ordering and matching layout. The num_foreground, matching, native IoU class
    weights and native soft objectness targets are all frozen, not only fg_mask.
    """
    def __init__(self, head):
        self.head=head
        if not hasattr(head,'get_assignments') or not hasattr(head,'get_losses_with_targets'):
            raise RuntimeError('Requires TDE_Head get_assignments and get_losses_with_targets; refuse generic YOLOXHead.')
        self.prev=None
        self.had_instance_override=False
        self.mode='capture'
        self.stored=[]
        self.cursor=0
        self.head_inputs=None
        self.loss_original=None
        self.loss_had_instance=False

    def __enter__(self):
        self.had_instance_override='get_assignments' in self.head.__dict__
        self.prev=self.head.__dict__.get('get_assignments')
        self.loss_had_instance='get_losses_with_targets' in self.head.__dict__
        self.loss_original=self.head.__dict__.get('get_losses_with_targets')
        self.native_get_assignments=self.head.get_assignments
        self.native_get_losses=self.head.get_losses_with_targets
        recorder=self
        def captured(_head,*args,**kwargs):
            if len(args)<4:
                raise RuntimeError('Unknown get_assignments argument layout')
            idx,ngt,boxes,classes=args[:4]
            idx=int(idx);ngt=int(ngt)
            if recorder.mode=='capture':
                if idx!=len(recorder.stored):
                    # Empty-GT images skip matching; order may have gaps.
                    if idx<len(recorder.stored):
                        raise RuntimeError('Duplicate get_assignments image index')
                result=recorder.native_get_assignments(*args,**kwargs)
                recorder.stored.append(NativeMatch(idx,boxes.detach().clone(),classes.detach().clone(),_clone_matched(result)))
                return result
            if recorder.mode!='replay':
                raise RuntimeError('Unknown replay mode')
            if recorder.cursor>=len(recorder.stored):
                raise RuntimeError('Adapted pass requested more matches than native pass')
            record=recorder.stored[recorder.cursor]
            recorder.cursor+=1
            if idx!=record.image_index or ngt!=record.gtboxes.shape[0]:
                raise RuntimeError('Changed GT order/count during replay')
            if not torch.equal(boxes.detach(),record.gtboxes) or not torch.equal(classes.detach(),record.gtclasses):
                raise RuntimeError('GT coordinates/classes changed during replay')
            if (args[4].shape[0]!=record.match[1].shape[0]):
                raise RuntimeError('Changed candidate anchor geometry during replay')
            return _clone_matched(record.match)
        def capture_inputs(_head,imgs,x_shifts,y_shifts,expanded_strides,labels,outputs,origin_preds,dtype):
            recorder.head_inputs={'images':imgs, 'shifts_x':x_shifts, 'shifts_y':y_shifts,
              'strides':expanded_strides, 'labels':labels, 'outputs':outputs,
              'origin_preds':origin_preds, 'dtype':dtype}
            return recorder.native_get_losses(imgs,x_shifts,y_shifts,expanded_strides,labels,outputs,origin_preds,dtype)
        self.head.get_assignments=MethodType(captured,self.head)
        self.head.get_losses_with_targets=MethodType(capture_inputs,self.head)
        return self

    def replay(self):
        if not self.stored:
            # Legitimate if all training examples contain zero GT.
            pass
        self.mode='replay';self.cursor=0;self.head_inputs=None

    def finalize(self):
        if self.mode=='replay' and self.cursor!=len(self.stored):
            raise RuntimeError(f'Replay used {self.cursor}/{len(self.stored)} native matches')
        if self.head_inputs is None:
            raise RuntimeError('TDE_Head did not expose get_losses_with_targets')

    def __exit__(self, *_):
        if self.had_instance_override:
            self.head.__dict__['get_assignments']=self.prev
        else:
            self.head.__dict__.pop('get_assignments',None)
        if self.loss_had_instance:
            self.head.__dict__['get_losses_with_targets']=self.loss_original
        else:
            self.head.__dict__.pop('get_losses_with_targets',None)


def _scale_of_area(area):
    # COCO thresholds in *input/augmented detector pixels*, not raw sensor pixels.
    return 'small' if area < 32.0**2 else ('medium' if area < 96.0**2 else 'large')


def reconstruct_tde_loss(head, capture: NativeMatchReplay):
    """Compute loss components using native SimOTA assignments and native IoU targets.

    Key feature: each positive foreground anchor is assigned the scale of its
    matched GT, not an inferred FPN level. Background objectness remains its
    own component; it cannot legitimately be attributed to one GT size.

    Output is differentiable w.r.t. adapted head logits, but native targets are
    detached. Match-head equality is a mandatory hard gate.
    """
    raw=capture.head_inputs
    if raw is None: raise RuntimeError('get_losses_with_targets was not called')
    if bool(head.use_l1):
        raise RuntimeError('L1-on TDE mode not supported by this exact audit; do not silently omit it')
    pred=raw['outputs'].float()
    labels=raw['labels'].float()
    if pred.ndim!=3 or pred.shape[-1]!=5+head.num_classes:
        raise RuntimeError('Unknown TDE output tensor layout')
    if labels.ndim!=3 or labels.shape[0]!=pred.shape[0] or labels.shape[-1]!=5:
        raise RuntimeError('Expected B x max_gt x 5 class/cx/cy/w/h labels')
    if len(raw['strides'])!=len(head.strides):
        raise RuntimeError('Unknown pyramid level list')
    lengths=[int(x.shape[1]) for x in raw['strides']]
    if sum(lengths)!=pred.shape[1]:
        raise RuntimeError('Anchor count mismatch vs expanded strides')
    device=pred.device
    zero=pred.sum()*0.0
    components={key:zero for key in ('box','classification','positive_objectness','background_objectness')}
    scale_comp={name:{'box':zero,'classification':zero,'positive_objectness':zero}
                for name in ('small','medium','large')}
    level_comp={int(s):{'box':zero,'classification':zero,'positive_objectness':zero}
                for s in head.strides}
    totals={'small':0,'medium':0,'large':0,'empty_images':0,'foreground':0,'background':0}
    num_gts=(labels.sum(-1)>0).sum(-1)
    matches={x.image_index:x for x in capture.stored}
    nfg=sum(int(m.match[4]) for m in capture.stored)
    denominator=max(nfg,1)
    lambda_soft=min(1.0,max(0.0,(int(head.current_epoch)-10)/40.0))
    reg_weight=2.5  # validated against the current TDE_Head source *and* runtime head loss
    offsets=[0]
    for n in lengths: offsets.append(offsets[-1]+n)
    for b in range(pred.shape[0]):
        y=labels[b]
        ngt=int(num_gts[b]); total_anchors=int(pred.shape[1])
        bbox=pred[b,:,:4]
        logit_obj=pred[b,:,4:5]
        logit_cls=pred[b,:,5:]
        obj_target=torch.zeros_like(logit_obj)
        mask=torch.zeros(total_anchors,device=device,dtype=torch.bool)
        if ngt:
            if b not in matches:
                raise RuntimeError(f'Missing native match for GT image {b}')
            m=matches[b]
            matched_classes, fg_mask, pred_iou, gtids, npos=m.match
            fg_mask=fg_mask.to(device=device,dtype=torch.bool)
            npos=int(npos)
            if fg_mask.numel()!=total_anchors or int(fg_mask.sum())!=npos:
                raise RuntimeError('Native foreground mask mismatch')
            gtids=gtids.to(device=device,dtype=torch.int64)
            pred_iou=pred_iou.to(device=device,dtype=torch.float32)
            matched_classes=matched_classes.to(device=device,dtype=torch.long)
            if not (len(gtids)==len(matched_classes)==len(pred_iou)==npos):
                raise RuntimeError('Native GT index/IoU class-target lengths mismatch')
            mask=fg_mask
            obj_target[mask]=(1-lambda_soft)+lambda_soft*pred_iou[:,None].clamp(0.1,1)
            if npos:
                positives=bbox[mask]
                gtboxes=y[:ngt,1:5][gtids].float()
                true_classes=y[:ngt,0][gtids].long()
                if not torch.equal(true_classes,matched_classes):
                    raise RuntimeError('Matching native class labels disagree with GT')
                bce_target=F.one_hot(matched_classes,head.num_classes).float()*pred_iou[:,None]
                boxloss=head.iou_loss(positives,gtboxes).reshape(-1)*reg_weight
                clsloss=F.binary_cross_entropy_with_logits(logit_cls[mask],bce_target,reduction='none').sum(-1)
                obloss=F.binary_cross_entropy_with_logits(logit_obj[mask],obj_target[mask],reduction='none').reshape(-1)
                components['box']=components['box']+boxloss.sum()/denominator
                components['classification']=components['classification']+clsloss.sum()/denominator
                components['positive_objectness']=components['positive_objectness']+obloss.sum()/denominator
                flattened=torch.where(mask)[0]
                areas=gtboxes[:,2]*gtboxes[:,3]
                scale_id=torch.where(areas<32.0**2,0,torch.where(areas<96.0**2,1,2))
                for sid,name in enumerate(('small','medium','large')):
                    selected=scale_id==sid
                    totals[name]+=int(selected.sum())
                    for key,losses in (('box',boxloss),('classification',clsloss),('positive_objectness',obloss)):
                        scale_comp[name][key]=scale_comp[name][key]+losses[selected].sum()/denominator
                for li, stride in enumerate(head.strides):
                    selected=(flattened>=offsets[li]) & (flattened<offsets[li+1])
                    for key,losses in (('box',boxloss),('classification',clsloss),('positive_objectness',obloss)):
                        level_comp[int(stride)][key]=level_comp[int(stride)][key]+losses[selected].sum()/denominator
        else:
            totals['empty_images']+=1
        bg=F.binary_cross_entropy_with_logits(logit_obj[~mask],obj_target[~mask],reduction='none').sum()/denominator
        components['background_objectness']=components['background_objectness']+bg
        totals['background']+=int((~mask).sum()); totals['foreground']+=int(mask.sum())
    comp_total=sum(components.values())
    return {
      'total':comp_total,
      'components':components,'scales':scale_comp,'levels':level_comp,
      'counts':totals,'frozen_gt_matches':len(matches),'num_foreground_normalizer':denominator,
      'objectness_soft_target_fraction':lambda_soft,
    }


def assert_head_loss_parity(head_outputs, recon, *, atol=2e-4, rtol=2e-4):
    if not isinstance(head_outputs,(tuple,list)) or len(head_outputs)<9:
        raise RuntimeError('Expected enriched TDE_Head output with >=9 fields')
    direct={'total':head_outputs[0], 'box':head_outputs[1],
            'positive_plus_background_objectness':head_outputs[2],
            'classification':head_outputs[3]}
    merged=recon['components']
    comparisons={
       'total':(direct['total'],recon['total']),
       'box':(direct['box'],merged['box']),
       'objectness':(direct['positive_plus_background_objectness'],merged['positive_objectness']+merged['background_objectness']),
       'classification':(direct['classification'],merged['classification']),
    }
    errors={}
    for name,(a,b) in comparisons.items():
        diff=float((a.float()-b.float()).detach().abs())
        errors[name]=diff
        if not torch.allclose(a.float().detach(),b.float().detach(),atol=atol,rtol=rtol):
            raise RuntimeError(f'TDE_Head loss-parity mismatch [{name}]: {float(a.detach()):.7g} vs {float(b.detach()):.7g}. ' 
                               'Source/head semantics have changed; do NOT treat results as fixed-target detector gradients.')
    return errors
