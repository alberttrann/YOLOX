"""Read-only Pilot G3: identify amplitude, geometry, and selective gating bottlenecks.

Run inside the existing patched tdevNext.3 checkout with
    python -m tools.audit_pgsma_gate_geometry --checkpoint PATH ...
No train/eval weights are changed. Reports are training-source mosaic mechanistic
observations only, NOT held-out AP and NOT a model-selection criterion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def _safe_mean(x):
    return float(x.float().mean().item()) if x.numel() else None


def _quantiles(x):
    if not x.numel():
        return None
    v=x.detach().float().flatten().cpu()
    q=torch.quantile(v,torch.tensor([0.,.1,.5,.9,.99,1.]))
    return {'min':float(q[0]),'p10':float(q[1]),'median':float(q[2]),
            'p90':float(q[3]),'p99':float(q[4]),'max':float(q[5]),'mean':float(v.mean())}


def _auc_gate(g, useful, harmful):
    """AUC for gate vs. utility labels. Returns None when a class is missing."""
    if not useful.any() or not harmful.any():
        return None
    pos=g[useful].detach().float().cpu().numpy().astype(np.float64)
    neg=np.sort(g[harmful].detach().float().cpu().numpy().astype(np.float64))
    lower=np.searchsorted(neg,pos,side='left')
    upper=np.searchsorted(neg,pos,side='right')
    return float((0.5*(lower+upper)).mean()/len(neg))


def _class_margin(logit, labels):
    """Ground-truth class minus highest competing class, same anchor."""
    if not len(labels):
        return logit.new_empty(0)
    row=torch.arange(len(labels),device=labels.device)
    true=logit[row,labels]
    false=logit.masked_fill(F.one_hot(labels, logit.shape[-1]).bool(),float('-inf'))
    return true-false.amax(-1)


def _all_stats(semantic,features,logits,objs,match,utility_margin=1e-3):
    """Entirely read-only; assumes B x C x H x W per pyramid level."""
    candidates=[]; npos=[]; movement=[]
    for level,(feat,cls,obj) in enumerate(zip(features,logits,objs)):
        d=semantic._candidate(level,feat.detach(),cls.detach(),obj.detach(),
                              mode='pgsma',freeze_reference=True)
        candidates.append(d)
        movement.append((d['r']-d['z']).float().square().sum(dim=1).sqrt())
        npos.append(int(feat.shape[-2]*feat.shape[-1]))
    baseline=semantic._flatten(logits).detach()
    delta=semantic._flatten([v['delta'] for v in candidates]).detach()
    gate=semantic._flatten([v['gate'] for v in candidates]).detach()
    fg=match['fg_ids'].long();lab=match['fg_labels'].long()
    quality=match['fg_quality'].float().detach()
    bg=match['bg_ids'].long()
    uid=match['gt_uid'].long();sizes=match['fg_sizes'].float().detach()
    native=baseline.reshape(-1,baseline.shape[-1])
    corr=native+delta.reshape(-1,delta.shape[-1])
    final=native+(gate*delta).reshape(-1,delta.shape[-1])
    all_g=gate.reshape(-1); all_d=delta.reshape(-1,delta.shape[-1])
    result={'counts':{'fg_anchors':int(fg.numel()),'semantic_bg_anchors':int(bg.numel()),
                      'total_anchors':int(native.shape[0])},
            'all_abs_delta':_quantiles(all_d.abs()),
            'all_gate':_quantiles(all_g),
            'embedding_r_minus_z_norm_by_level':[_quantiles(v) for v in movement],
            'logit_gain_by_level':[float(F.softplus(v.detach())) for v in semantic.log_gain],
            'foreground':{},'background':{}}
    if bg.numel():
        bg_gain=(F.softplus(native[bg]).sum(-1)-F.softplus(final[bg]).sum(-1))
        bg_cand=(F.softplus(native[bg]).sum(-1)-F.softplus(corr[bg]).sum(-1))
        result['background']={'gated_bce_gain':_quantiles(bg_gain),
                              'ungated_bce_gain':_quantiles(bg_cand),
                              'gate':_quantiles(all_g[bg]),
                              'mean_signed_delta':_safe_mean(all_d[bg])}
    if not fg.numel():
        return result
    if not (len(fg)==len(lab)==len(quality)==len(uid)==len(sizes)):
        raise ValueError('Mismatch in foreground match shapes')
    target=F.one_hot(lab,native.shape[-1]).float()*quality[:,None]
    def per_loss(v):
        return F.binary_cross_entropy_with_logits(v[fg],target,reduction='none').sum(-1)
    orig_l=per_loss(native)
    ungated_l=per_loss(corr)
    gated_l=per_loss(final)
    gain=(orig_l-ungated_l)
    g_gain=(orig_l-gated_l)
    fg_gate=all_g[fg]; gt_shift=all_d[fg,lab]
    true_margin_before=_class_margin(native[fg],lab)
    true_margin_cand=_class_margin(corr[fg],lab)
    true_margin_gated=_class_margin(final[fg],lab)
    useful=gain>utility_margin
    harmful=gain<-utility_margin
    levels_local=fg%sum(npos)
    boundaries=torch.tensor(np.cumsum(npos)[:-1],device=fg.device)
    level_ids=torch.bucketize(levels_local,boundaries,right=True)
    def stats_for(mask):
        if not mask.any():return {'count':0}
        idx=torch.nonzero(mask).flatten()
        return {'count':int(idx.numel()),'candidate_bce_gain':_quantiles(gain[idx]),
                'gated_bce_gain':_quantiles(g_gain[idx]),
                'gate':_quantiles(fg_gate[idx]),
                'gt_class_delta':_quantiles(gt_shift[idx]),
                'margin_improvement_ungated':_quantiles((true_margin_cand-true_margin_before)[idx]),
                'margin_improvement_gated':_quantiles((true_margin_gated-true_margin_before)[idx]),
                'useful_fraction_gt_1e-3':float(useful[idx].float().mean()),
                'harmful_fraction_lt_neg_1e-3':float(harmful[idx].float().mean())}
    result['foreground']={
        'all':stats_for(torch.ones_like(gain,dtype=torch.bool)),
        'gate_utility_auc_1e-3':_auc_gate(fg_gate,useful,harmful),
        'utility_abstention_fraction_1e-3':float((~useful & ~harmful).float().mean()),
        'gate_when_useful':_quantiles(fg_gate[useful]),
        'gate_when_harmful':_quantiles(fg_gate[harmful]),
        'by_quality':{
            'lt03':stats_for(quality<.3),
            '03to05':stats_for((quality>=.3)&(quality<.5)),
            '05to07':stats_for((quality>=.5)&(quality<.7)),
            'ge07':stats_for(quality>=.7)},
        'by_effective_gt_min_side_px':{
            'lt4':stats_for(sizes<4),
            '4to8':stats_for((sizes>=4)&(sizes<8)),
            '8to16':stats_for((sizes>=8)&(sizes<16)),
            '16to32':stats_for((sizes>=16)&(sizes<32)),
            'ge32':stats_for(sizes>=32)},
        'by_fpn_level':{'P3':stats_for(level_ids==0),'P4':stats_for(level_ids==1),
                        'P5':stats_for(level_ids==2)},
        'utility_threshold_sensitivity':{
            str(th):{'useful':int((gain>th).sum()),'harmful':int((gain<-th).sum()),
                     'auc':_auc_gate(fg_gate,gain>th,gain<-th)}
            for th in (0.,1e-4,1e-3,1e-2)}
        }
    return result



def summarize_batches(batches):
    """Weighted descriptive means; NEVER substitute for paired confidence intervals."""
    report={}
    for view in ('clean','perturbed'):
        arr=[batch[view] for batch in batches]
        def weighted_value(rows,stat):
            vals=[(x['count'],x[stat]['mean']) for x in rows
                  if x.get('count',0)>0 and isinstance(x.get(stat),dict)
                  and x[stat].get('mean') is not None]
            denom=sum(count for count,_ in vals)
            return sum(count*val for count,val in vals)/denom if denom else None
        fg=[a['foreground'].get('all',{}) for a in arr]
        metrics={'total_fg_anchors':sum(x.get('count',0) for x in fg),
                 'fg_ungated_mean_bce_gain':weighted_value(fg,'candidate_bce_gain'),
                 'fg_gated_mean_bce_gain':weighted_value(fg,'gated_bce_gain'),
                 'fg_mean_gt_class_delta':weighted_value(fg,'gt_class_delta'),
                 'fg_mean_margin_improvement':weighted_value(fg,'margin_improvement_ungated'),
                 'by_size':{},'by_level':{}}
        for group,dest in (('by_effective_gt_min_side_px','by_size'),('by_fpn_level','by_level')):
            keys=arr[0]['foreground'][group]
            for key in keys:
                group_rows=[x['foreground'][group][key] for x in arr]
                metrics[dest][key]={'anchors':sum(x.get('count',0) for x in group_rows),
                    'ungated_mean_bce_gain':weighted_value(group_rows,'candidate_bce_gain'),
                    'gated_mean_bce_gain':weighted_value(group_rows,'gated_bce_gain'),
                    'mean_gt_class_delta':weighted_value(group_rows,'gt_class_delta')}
        auc=[a['foreground'].get('gate_utility_auc_1e-3') for a in arr]
        auc=[x for x in auc if x is not None]
        metrics['per_batch_gate_auc_mean_1e-3']=sum(auc)/len(auc) if auc else None
        metrics['per_batch_gate_auc_defined']=len(auc)
        metrics['per_batch_gate_auc_total']=len(arr)
        metrics['all_anchor_abs_delta_mean']=sum(a['all_abs_delta']['mean'] for a in arr)/len(arr)
        metrics['all_anchor_gate_mean']=sum(a['all_gate']['mean'] for a in arr)/len(arr)
        report[view]=metrics
    return report

def run(args):
    from ._pgsma_pilot_core import atomic_json
    from .pgsma_bounded_pilot import safe_checkpoint_state, seed_everything, verify_batch
    from .audit_pgsma_mechanism_probe import state_hash
    if not 1<=args.batches<=64 or not 1<=args.batch_size<=4:
        raise ValueError('batches must be 1..64 and batch_size 1..4')
    seed_everything(args.seed)
    from exps.default.tde_yolox_s_pgsma import Exp
    exp=Exp(); exp.data_num_workers=0; exp.ema=False
    if exp.allow_full_training:
        raise RuntimeError('Full training must remain locked')
    exp.scpa=False; exp.pgsma_variant='pgsma'
    payload,weights=safe_checkpoint_state(args.checkpoint)
    info=payload.get('pgsma_pilot')
    if not isinstance(info,dict) or info.get('variant')!='pgsma_simota' or int(info.get('successful_semantic_updates',0))<1:
        raise ValueError('Need trained patched PG-SMA SimOTA pilot checkpoint')
    model=exp.get_model()
    model.load_state_dict(weights,strict=True)
    model=model.to(args.device).train()
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
            layer.eval()
    before=state_hash(model)
    loader=exp.get_data_loader(args.batch_size,False,no_aug=not args.mosaic,cache_img=False)
    iterator=iter(loader)
    reports=[]
    for step in range(args.batches):
        try:batch=next(iterator)
        except StopIteration:
            iterator=iter(loader);batch=next(iterator)
        images,targets=batch[:2]
        verify_batch(images,targets,exp)
        images=images.float().to(args.device);targets=targets.float().to(args.device)
        images,targets=exp.preprocess(images,targets,exp.input_size)
        valid=images[:,3:4]>.5
        with torch.no_grad():
            model(images[:,:3].contiguous(),targets,valid_image_mask=valid)
            c=model.head.pgsma_cache
            clean=_all_stats(model.semantic,c['features'],c['native_cls'],c['native_obj'],c['match'])
            perturbed=model.semantic._perturb(c['features'],args.seed+step*73)
            pert_logits=[]
            for i,f in enumerate(perturbed):
                cls=model.head.cls_preds[i]
                pert_logits.append(F.conv2d(f,cls.weight.detach().float(),
                                            cls.bias.detach().float() if cls.bias is not None else None))
            pert=_all_stats(model.semantic,perturbed,pert_logits,c['native_obj'],c['match'])
        reports.append({'batch':step+1,'clean':clean,'perturbed':pert})
        print(f'[gate-geometry] {step+1}/{args.batches} '
              f'FG={clean["counts"]["fg_anchors"]} '
              f'gate_auc={clean["foreground"].get("gate_utility_auc_1e-3")}',flush=True)
    if state_hash(model)!=before:
        raise RuntimeError('Frozen model state changed')
    report={'status':'CHECKS_PASSED','checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'model_state_unchanged':True,'scope':'Frozen SOURCE training batches, no validation, no adverse',
            'mosaic':bool(args.mosaic),'seed':args.seed,'summary':summarize_batches(reports),'batches':reports,
            'limitations':['Per-batch anchor-weighted statistics; do not interpret as per-GT CI',
                           'Gate utility is defined using BCE against IoU-soft GT labels, not COCO AP',
                           'Tiny utility differences can produce unstable binary signs',
                           'AUC only defined when enough useful/harmful anchors exist']}
    atomic_json(args.json,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261110)
    p.add_argument('--device',default='cuda')
    p.add_argument('--mosaic',action='store_true')
    p.add_argument('--json',required=True)
    args=p.parse_args()
    try:
        report=run(args)
        print(json.dumps({'status':report['status'],'batches':len(report['batches']),
                          'model_state_unchanged':report['model_state_unchanged'],
                          'summary':report['summary']},indent=2))
        return 0
    except Exception as err:
        from ._pgsma_pilot_core import atomic_json
        blocked={'status':'BLOCKED','error':type(err).__name__+': '+str(err)}
        atomic_json(args.json,blocked)
        print(json.dumps(blocked,indent=2));return 2


if __name__=='__main__':
    sys.exit(main())
