"""Source-only supervised meta-trainer with frozen SimOTA and GT coverage.

GT is used only in outer losses. The online `PRISMTTT.propose` is label-free.
Train the meta-objective with forced bounded updates; calibrate gate afterwards
on DISJOINT source scenes. No unavailable OOD labels are consulted.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor
from torch.nn import functional as F
from .fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity
from .objectives import ranking_and_coverage,native_relative_loss,SCALES


def assert_frozen_native(model):
    if model.native.training or model.native.backbone.training:
        raise RuntimeError('Frozen YOLOX backbone must be in eval mode')
    if any(p.requires_grad for p in model.native.parameters()):
        raise RuntimeError('Native detector parameters require gradients')
    if any(m.training for m in model.native.head.modules() if m is not model.native.head):
        raise RuntimeError('Native head child module has mutable training state')


def source_view_result(model,images:Tensor,labels:Tensor,*,meta_train:bool=True,
                       clean:bool=False,mode:str='on',weight_ranking:float=.10,weight_coverage:float=.10,
                       tail_weight:float=.5,bg_weight:float=.5,clean_weight:float=1.0):
    if images.shape[0]!=1 or labels.shape[0]!=1:
        raise ValueError('Use B=1 source episodes; accumulate multiple scenes in optimizer')
    assert_frozen_native(model)
    with torch.no_grad():native_feat=model.native_features(images)
    with NativeMatchReplay(model.native.head) as cap:
        with torch.no_grad():
            native_head=model.head_predictions(native_feat,images,labels)
        cap.finalize()
        native_rec=reconstruct_tde_loss(model.native.head,cap)
        parity_native=assert_head_loss_parity(native_head,native_rec)
        native_proxy=ranking_and_coverage(cap.head_inputs,model.native.head,cap)
        hard_ids=native_proxy['hard_negative_ids']
        # Same native matches, IoU classification targets and negative anchors.
        cap.replay()
        proposal=model.sidecar.propose(native_feat,meta_train=meta_train)
        adapted,gate_applied=model.sidecar.apply(native_feat,proposal,mode,require_calibration=(mode=='auto'))
        with torch.enable_grad():
            adapted_head=model.head_predictions(adapted,images,labels)
            cap.finalize()
            adapted_rec=reconstruct_tde_loss(model.native.head,cap)
            parity_adapted=assert_head_loss_parity(adapted_head,adapted_rec)
            adapted_proxy=ranking_and_coverage(cap.head_inputs,model.native.head,cap,
                                                 fixed_hard_ids=hard_ids)
            loss,parts=native_relative_loss({'replay':native_rec,'proxy':native_proxy},
                         {'replay':adapted_rec,'proxy':adapted_proxy},clean=clean,
                         background_weight=bg_weight,tail_weight=tail_weight,clean_weight=clean_weight,
                         ranking_weight=weight_ranking,coverage_weight=weight_coverage)
        return {'objective':loss,'native':native_rec,'adapted':adapted_rec,
                'native_proxy':native_proxy,'adapted_proxy':adapted_proxy,
                'parts':parts,'proposal':proposal,'gate_applied':gate_applied,
                'parity':(parity_native,parity_adapted)}


def source_risk_gate_loss(model,report,*,benefit_tol:float=2e-5):
    """GT-derived gate labels used ONLY in source training/calibration.

    For a represented scale, accept if no background harm and scale-local
    adapted detection+coverage loss improves beyond the tolerance.
    Unrepresented scales have zero weight, NOT synthetic safety passes.
    """
    nat=report['native']; ada=report['adapted']
    np=report['native_proxy'];ap=report['adapted_proxy']
    targets=[];mask=[]
    bg_delta=(ada['components']['background_objectness']-nat['components']['background_objectness']).detach()
    for s in SCALES:
        present=np['gt_support'][s]>0
        mask.append(float(present))
        if not present:
            targets.append(0.)
            continue
        all_delta=(report['adapted_proxy']['scale_gt_detection'][s]-
                   report['native_proxy']['scale_gt_detection'][s]+
                   ap['scale_proxy'][s]-np['scale_proxy'][s]).detach()
        targets.append(float((all_delta < -benefit_tol) and (bg_delta <= benefit_tol)))
    scores=model.sidecar.gate(report['proposal'].signals.detach())
    t=scores.new_tensor(targets)[None,:]
    weights=scores.new_tensor(mask)[None,:]
    loss=(F.binary_cross_entropy(scores,t,reduction='none')*weights).sum()/weights.sum().clamp_min(1)
    return loss,{'labels':targets,'present':mask,'gate_scores':scores.detach().flatten().tolist()}


def source_pair_step(model,clean,shifted,labels,*,clean_weight=1.,gate_weight=.1,
                     accum=1,tail_weight=.5,bg_weight=.5):
    """Accumulate gradients for paired clean-source images; caller owns step."""
    assert clean.shape==shifted.shape and clean.shape[0]==1
    if torch.equal(clean,shifted):raise RuntimeError('Appearance-shifted view unchanged')
    results=[]
    for view,images,weight in (('clean',clean,clean_weight),('shifted',shifted,1.)):
        report=source_view_result(model,images,labels,meta_train=True,clean=(view=='clean'),
                                  tail_weight=tail_weight,bg_weight=bg_weight)
        gate_loss,gate_meta=source_risk_gate_loss(model,report)
        total=(weight*report['objective']+gate_weight*gate_loss)/((1.+clean_weight)*accum)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite source meta loss')
        total.backward()
        results.append({'view':view,'native':float(report['native']['total'].detach()),
          'adapted':float(report['adapted']['total'].detach()),
          'delta':float(report['parts']['total_delta'].detach()),
          'bg_delta':float(report['parts']['background_delta'].detach()),
          'gt_support':report['native_proxy']['gt_support'],
          'gate':gate_meta})
    return results


def parameter_groups(sidecar,lr:float):
    groups=[]
    for name,mod,factor in [('basis',sidecar.bases,.2),('ssl_weight',sidecar.ssl_weight_net,1.),
                            ('step',sidecar.step_net,.5),('gate',sidecar.gate.net,.5)]:
        ps=[p for p in mod.parameters() if p.requires_grad]
        if ps:groups.append({'params':ps,'lr':lr*factor,'tag':name})
    return groups


def optimizer_step(sidecar,optim,max_norm=.5):
    params=[p for p in sidecar.parameters() if p.requires_grad]
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
        optim.zero_grad(set_to_none=True);raise FloatingPointError('Nonfinite meta-gradients')
    grad_norm=float(torch.nn.utils.clip_grad_norm_(params,max_norm,error_if_nonfinite=True))
    if not math.isfinite(grad_norm) or grad_norm<=0:
        optim.zero_grad(set_to_none=True);raise RuntimeError('No detector-to-SSL meta-gradient')
    optim.step();optim.zero_grad(set_to_none=True)
    return grad_norm
