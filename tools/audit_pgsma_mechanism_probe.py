"""Pilot E: read-only semantic clean-versus-perturbed utility and hard-negative overlap audit.

No backprop, optimizer, weight changes, evaluation labels used for tuning, or adverse data.
Requires existing vNext.3 Pilot C checkout, including pgsma_bounded_pilot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from ._pgsma_pilot_core import atomic_json, group_scale
from .pgsma_bounded_pilot import seed_everything, verify_batch, safe_checkpoint_state

BINS=('lt4','4to8','8to16','16to32','ge32')


def state_hash(model):
    h=hashlib.sha256()
    for name,x in sorted(model.state_dict().items()):
        h.update(name.encode('utf8'))
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def anchor_centers(features,strides,device):
    """Exact P3/P4/P5 center order in training cache's [B, concatenated-level] layout."""
    if len(features)!=len(strides):
        raise ValueError('feature/stride count mismatch')
    parts=[]
    for x,stride in zip(features,strides):
        h,w=x.shape[-2:]
        yy,xx=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float32),
                             torch.arange(w,device=device,dtype=torch.float32),indexing='ij')
        parts.append(torch.stack(((xx+.5)*float(stride),(yy+.5)*float(stride)),dim=-1).reshape(-1,2))
    return torch.cat(parts,dim=0)


def valid_gt_info(targets):
    """Return GT boxes and size labels in the exact order of head's global gt_uid."""
    out=[]
    for row in targets:
        mask=(row[:,3]>0)&(row[:,4]>0)&torch.isfinite(row).all(-1)
        gt=row[mask]
        sizes=[group_scale(float(v)) for v in gt[:,3:5].min(-1).values.detach().cpu()]
        out.append((gt[:,1:5],sizes))
    return out


def hard_negative_geometry(bg_ids,centers,gt_info,proximity_multiplier=1.5):
    """Selected unmatched, high-objectness anchors overlapping/near annotated GT rectangles.

    Center geometry only: this does NOT assert that such anchors are mislabeled,
    nor use decoded prediction IoU. Returns aggregate overlap counts by GT size.
    """
    if proximity_multiplier < 1.0:
        raise ValueError('proximity_multiplier must be >= 1')
    total=int(centers.shape[0])
    ids=bg_ids.reshape(-1).long()
    result={'selected_bg_anchors':int(ids.numel()),'center_inside_gt':0,
            'center_inside_dilated_gt':0,'inside_gt_by_size':{k:0 for k in BINS},
            'near_gt_by_size':{k:0 for k in BINS}}
    if not len(ids):return result
    for b,(gt,sizes) in enumerate(gt_info):
        chosen=(ids//total)==b
        if not chosen.any() or not len(gt):continue
        xy=centers[(ids[chosen]%total)]
        # [anchors,G] membership; this requires no tensors larger than 256*maxGT on normal batches
        offsets=(xy[:,None,:]-gt[None,:,:2]).abs()
        inside=(offsets <= gt[None,:,2:4]/2).all(-1)
        near=(offsets <= gt[None,:,2:4]*(proximity_multiplier/2)).all(-1)
        result['center_inside_gt']+=int(inside.any(-1).sum())
        result['center_inside_dilated_gt']+=int(near.any(-1).sum())
        for k in BINS:
            size_mask=torch.tensor([s==k for s in sizes],device=inside.device,dtype=torch.bool)
            if size_mask.any():
                result['inside_gt_by_size'][k]+=int(inside[:,size_mask].any(-1).sum())
                result['near_gt_by_size'][k]+=int(near[:,size_mask].any(-1).sum())
    return result


def per_gt_size_loss_stats(baseline,corrected,match,gt_info):
    """Soft-IoU target foreground BCE, averaging matched anchors per GT."""
    ids=match['fg_ids'].long()
    labels=match['fg_labels'].long()
    qualities=match['fg_quality'].float()
    uid=match['gt_uid'].long()
    result={k:{'gt_with_positives':0,'matched_anchors':0,'baseline_loss_sum_per_gt':0.,
               'corrected_loss_sum_per_gt':0.,'gain_sum_per_gt':0.} for k in BINS}
    if not ids.numel():return result
    nclass=baseline.shape[-1]
    y=F.one_hot(labels,nclass).float()*qualities[:,None]
    before=F.binary_cross_entropy_with_logits(baseline.reshape(-1,nclass)[ids],y,reduction='none').sum(-1)
    after=F.binary_cross_entropy_with_logits(corrected.reshape(-1,nclass)[ids],y,reduction='none').sum(-1)
    names=[s for _,v in gt_info for s in v]
    if uid.min()<0 or uid.max()>=len(names):raise ValueError('out-of-range gt_uid')
    for g in torch.unique(uid).tolist():
        mask=uid==int(g)
        item=result[names[int(g)]]
        item['gt_with_positives']+=1
        item['matched_anchors']+=int(mask.sum())
        b=float(before[mask].mean());a=float(after[mask].mean())
        item['baseline_loss_sum_per_gt']+=b
        item['corrected_loss_sum_per_gt']+=a
        item['gain_sum_per_gt']+=b-a
    return result


def semantic_frame(semantic,feats,logits,obj,match,gt_info,mode='pgsma'):
    ds=[];gs=[]
    for i,(feature,cls,ob) in enumerate(zip(feats,logits,obj)):
        val=semantic._candidate(i,feature,cls,ob,mode=mode,freeze_reference=True)
        ds.append(val['delta']);gs.append(val['gate'])
    base=semantic._flatten(logits)
    delta=semantic._flatten(ds)
    gate=semantic._flatten(gs)
    final=base+gate*delta
    raw_candidate=base+delta
    ids=match['fg_ids'];bg=match['bg_ids']
    lp=semantic._sample_losses
    nf,nb=lp(base,ids,match['fg_labels'],match['fg_quality'],match['gt_uid'],bg,False)
    cf,cb=lp(raw_candidate,ids,match['fg_labels'],match['fg_quality'],match['gt_uid'],bg,False)
    gf,gb=lp(final,ids,match['fg_labels'],match['fg_quality'],match['gt_uid'],bg,False)
    return {'fg_improvement':float(nf-gf), 'bg_improvement':float(nb-gb),
            'total_improvement':float(nf+nb-gf-gb),
            'candidate_fg_improvement':float(nf-cf),
            'candidate_bg_improvement':float(nb-cb),
            'mean_abs_correction':float(delta.abs().mean()),
            'all_anchor_gate_mean':float(gate.mean()),
            'fg_gate_mean':float(gate.reshape(-1,1)[ids].mean()) if ids.numel() else None,
            'hard_bg_gate_mean':float(gate.reshape(-1,1)[bg].mean()) if bg.numel() else None,
            'by_gt_size':per_gt_size_loss_stats(base,final,match,gt_info)}


def merge_reports(samples):
    out={'batches':len(samples),'clean':{},'perturbed':{},'hard_negatives':{}}
    for variant in ('clean','perturbed'):
        arr=[x[variant] for x in samples]
        keys=['fg_improvement','bg_improvement','total_improvement','candidate_fg_improvement',
              'candidate_bg_improvement','mean_abs_correction','all_anchor_gate_mean',
              'fg_gate_mean','hard_bg_gate_mean']
        out[variant]={k:{'mean':sum(v[k] for v in arr if v[k] is not None)/
                    max(sum(v[k] is not None for v in arr),1),
                    'min':min((v[k] for v in arr if v[k] is not None),default=None),
                    'max':max((v[k] for v in arr if v[k] is not None),default=None)} for k in keys}
        out[variant]['by_gt_size']={}
        for k in BINS:
            tot={fld:sum(x['by_gt_size'][k][fld] for x in arr)
                 for fld in ('gt_with_positives','matched_anchors','baseline_loss_sum_per_gt',
                             'corrected_loss_sum_per_gt','gain_sum_per_gt')}
            gt=tot['gt_with_positives']
            out[variant]['by_gt_size'][k]={'gt_with_positives':gt,'matched_anchors':tot['matched_anchors'],
                'mean_improvement_per_gt':tot['gain_sum_per_gt']/gt if gt else None,
                'mean_native_loss_per_gt':tot['baseline_loss_sum_per_gt']/gt if gt else None,
                'mean_corrected_loss_per_gt':tot['corrected_loss_sum_per_gt']/gt if gt else None}
    hard=[x['hard_negative_geometry'] for x in samples]
    count=sum(x['selected_bg_anchors'] for x in hard)
    out['hard_negatives']={'selected_bg_anchors':count,
        'center_inside_gt':sum(x['center_inside_gt'] for x in hard),
        'center_inside_dilated_gt':sum(x['center_inside_dilated_gt'] for x in hard),
        'fraction_center_inside_gt':sum(x['center_inside_gt'] for x in hard)/count if count else None,
        'fraction_center_inside_dilated_gt':sum(x['center_inside_dilated_gt'] for x in hard)/count if count else None,
        'inside_gt_by_size':{s:sum(x['inside_gt_by_size'][s] for x in hard) for s in BINS},
        'near_gt_by_size':{s:sum(x['near_gt_by_size'][s] for x in hard) for s in BINS}}
    out['per_batch']=[{'batch':i+1,
        'clean_fg':x['clean']['fg_improvement'],
        'pert_fg':x['perturbed']['fg_improvement'],
        'clean_bg':x['clean']['bg_improvement'],
        'pert_bg':x['perturbed']['bg_improvement'],
        'hard_bg_inside_gt':x['hard_negative_geometry']['center_inside_gt']}
        for i,x in enumerate(samples)]
    return out


def run(args):
    if not (1<=args.batches<=64 and 1<=args.batch_size<=4):
        raise ValueError('1..64 batches and batch_size 1..4 required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    seed_everything(args.seed)
    from exps.default.tde_yolox_s_pgsma import Exp
    exp=Exp();exp.data_num_workers=0;exp.ema=False
    if exp.allow_full_training:raise RuntimeError('80-epoch guard unexpectedly unlocked')
    exp.scpa=False
    exp.pgsma_variant='pgsma'
    payload,checkpoint_weights=safe_checkpoint_state(args.checkpoint)
    meta=payload.get('pgsma_pilot')
    if (not isinstance(meta,dict) or meta.get('variant')!='pgsma_simota' or
            int(meta.get('successful_semantic_updates',0))<1):
        raise RuntimeError('Expected trained SimOTA PG-SMA pilot checkpoint')
    model=exp.get_model()
    model.load_state_dict(checkpoint_weights,strict=True)
    source={'checkpoint':str(args.checkpoint),'sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'matched_tensors':len(checkpoint_weights),'is_pilot_warmstart':True,'variant':meta['variant']}
    model=model.to(args.device).train()
    # Frozen BN state even though training forward is required to obtain assignment cache.
    for layer in model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
    before=state_hash(model)
    loader=exp.get_data_loader(args.batch_size,False,no_aug=not args.mosaic,cache_img=False)
    it=iter(loader)
    samples=[]
    for ix in range(args.batches):
        try:batch=next(it)
        except StopIteration:it=iter(loader);batch=next(it)
        images,targets=batch[:2]
        verify_batch(images,targets,exp)
        images=images.float().to(args.device)
        targets=targets.float().to(args.device)
        images,targets=exp.preprocess(images,targets,exp.input_size)
        valid=images[:,3:4]>.5
        rgb=images[:,:3].contiguous()
        with torch.no_grad():
            model(rgb,targets,valid_image_mask=valid)
            cache=model.head.pgsma_cache
            match=cache['match']
            gt=valid_gt_info(targets)
            clean=semantic_frame(model.semantic,cache['features'],cache['native_cls'],cache['native_obj'],match,gt)
            pert_feats=model.semantic._perturb(cache['features'],args.seed+ix*73)
            pert_log=[]
            for level,feat in enumerate(pert_feats):
                cls=model.head.cls_preds[level]
                pert_log.append(F.conv2d(feat,cls.weight.detach().float(),
                                        cls.bias.detach().float() if cls.bias is not None else None))
            pert=semantic_frame(model.semantic,pert_feats,pert_log,cache['native_obj'],match,gt)
            centers=anchor_centers(cache['features'],model.head.strides,rgb.device)
            hard=hard_negative_geometry(match['bg_ids'],centers,gt)
        samples.append({'clean':clean,'perturbed':pert,'hard_negative_geometry':hard})
        print(f'[mechanism] {ix+1}/{args.batches} clean_fg={clean["fg_improvement"]:+.4f} '
              f'pert_fg={pert["fg_improvement"]:+.4f} hard_bg_inside_gt={hard["center_inside_gt"]}',flush=True)
    after=state_hash(model)
    if before!=after:raise RuntimeError('Model state changed in read-only audit')
    report={'status':'CHECKS_PASSED','training_performed':False,'model_state_unchanged':True,
        'checkpoint':source,'seed':args.seed,'batches':args.batches,'batch_size':args.batch_size,
        'mosaic_enabled':args.mosaic,'test_type':'frozen train-batch clean-vs-synthetic-pert utility',
        'metrics':merge_reports(samples),'limits':[
          'Train minibatches and native assigned GT, NOT clean validation AP or adverse-domain utility',
          'Hard negatives inside/near GT are NOT automatically incorrect supervision',
          'Clean features are extracted from same source training forward as perturbed features',
          'Mean BCE improvement can be dominated by class/background terms despite per-GT reporting',
          'No model, gate, optimizer, or label changes; no causal conclusion without ablation']}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--batches',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--device',default='cuda')
    p.add_argument('--mosaic',action='store_true',help='Use configured mixed Mosaic instead of no-Aug loader')
    p.add_argument('--json',required=True)
    args=p.parse_args()
    try:
        report=run(args)
        atomic_json(args.json,report)
        print(json.dumps({'status':report['status'],'model_state_unchanged':True,
                          'clean':report['metrics']['clean'],
                          'perturbed':report['metrics']['perturbed'],
                          'hard_negatives':report['metrics']['hard_negatives']},indent=2))
        return 0
    except Exception as e:
        record={'status':'BLOCKED','error':type(e).__name__+': '+str(e)}
        atomic_json(args.json,record)
        print(json.dumps(record,indent=2))
        return 2

if __name__=='__main__':sys.exit(main())
