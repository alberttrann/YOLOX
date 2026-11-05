#!/usr/bin/env python3
"""Tiny CLEAN-source task-aligned meta-training pilot (no long YOLOX run).

Only optimizes the new SafeSpatialTTT sidecar, not native YOLOX weights,
EMA, prototype banks, SimOTA matcher, or BN. Uses native fixed assignments
and exact detection->SSL mixed derivatives. Holds out separate source-TRAIN
images for a *development* check, not a final validation claim.

Example (PowerShell):
 python tools/pilot_safe_ttt_meta.py `
   --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
   --model-exp exps/default/tde_yolox_s.py --weights ema `
   --train-indices 0 1 --holdout-indices 10 37 `
   --steps 2 --output audits/safe_ttt_pilot.json

NO ACDC in this script. All labels are source-clean TRAIN labels.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch

from audit_safe_ttt_yolox import (setup_model,get_clean_train_example,
     get_size_from_native,head_loss_pass,audit_one,record_model_hash)
from fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity
from native_dark2_router import Dark2AuditRouter
from safe_ttt_model import SafeSpatialTTT


def fixed_native_targets_step(model,safe,images,labels,optimizer,
                              nonregression_weight=0.5,max_grad_norm=5.0,
                              objective_mode="detector_total",background_weight=0.1):
    if images.shape[0]!=1:raise RuntimeError('Pilot requires B=1')
    with Dark2AuditRouter(model,safe) as router, NativeMatchReplay(model.head) as matcher:
        router.set_mode('native')
        with torch.no_grad():
            native_out=head_loss_pass(model,images,labels)
        router.require_one_hit(1);matcher.finalize()
        native=reconstruct_tde_loss(model.head,matcher)
        assert_head_loss_parity(native_out,native)

        router.set_mode('episode',meta_train=True)
        matcher.replay()
        with torch.enable_grad():
            meta_out=head_loss_pass(model,images,labels)
            router.require_one_hit(1);matcher.finalize()
            adapted=reconstruct_tde_loss(model.head,matcher)
            assert_head_loss_parity(meta_out,adapted)
            if objective_mode not in ('detector_total','scale_balanced'):
                raise ValueError('Unknown objective_mode')
            nonregression=adapted['total']*0.0
            scale_violations={}
            for scale in ('small','medium','large'):
                if native['counts'][scale]==0:
                    continue
                # Per-GT matched positive weighting prevents rare micro objects
                # from being drowned by more common large matched anchors.
                count=native['counts'][scale]
                normalization=native['num_foreground_normalizer']/max(count,1)
                before=sum(native['scales'][scale].values()).detach()*normalization
                after=sum(adapted['scales'][scale].values())*normalization
                viol=torch.relu(after-before)
                nonregression=nonregression+viol
                scale_violations[scale]=float(viol.detach())
            # Only scale groups present in the clean source sample are active.
            active_count=max(1,len(scale_violations))
            nonregression=nonregression/active_count
            if objective_mode=='scale_balanced':
                positive_means=[sum(adapted['scales'][sz].values())*
                                native['num_foreground_normalizer']/native['counts'][sz]
                    for sz in ('small','medium','large') if native['counts'][sz]>0]
                if positive_means:
                    det_obj=sum(positive_means)/len(positive_means)
                else:
                    det_obj=adapted['total']*0.0
                # Normalized *per background anchor*, not per foreground.
                # Include background as an explicit term, not falsely assign it
                # to an object-scale group.
                bg_count=max(1,native['counts']['background'])
                bg_mean=(adapted['components']['background_objectness']*
                         native['num_foreground_normalizer']/bg_count)
                det_obj=det_obj+background_weight*bg_mean
            else:
                det_obj=adapted['total']
            # Both modes retain exact detection->inner SSL mixed derivatives.
            loss=det_obj+nonregression_weight*nonregression
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite exact meta pilot loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        predictor_grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in safe.predictor.parameters() if p.grad is not None))
        adapter_grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in safe.adapter.parameters() if p.grad is not None))
        if not math.isfinite(predictor_grad) or not math.isfinite(adapter_grad):
            raise FloatingPointError('Nonfinite safe-stage meta gradient')
        if predictor_grad==0:
            raise RuntimeError('Meta-training failed to reach SSL predictor from detector')
        grad_norm=float(torch.nn.utils.clip_grad_norm_(safe.parameters(),max_grad_norm,error_if_nonfinite=True))
        optimizer.step()
        return {'native_detection_loss':float(native['total'].detach()),
                'adapted_detection_loss':float(adapted['total'].detach()),
                'meta_objective':float(loss.detach()),
                'scale_nonregression':scale_violations,
                'meta_predictor_grad_norm':predictor_grad,
                'adapter_grad_norm':adapter_grad,
                'preclip_total_grad_norm':grad_norm,
                'step_policy':safe.step_policy,
                'objective_mode':objective_mode,
                'per_scale_nonregression_weight':nonregression_weight,
                'background_weight':background_weight}


def load_source_example(dataset,index,device,seed):
    torch.manual_seed(seed+index)
    random.seed(seed+index)
    try:
        import numpy as np
        np.random.seed((seed+index)%(2**32))
    except ImportError:pass
    return get_clean_train_example(dataset,index,device)


def evaluate_source_holdout(model,safe,dataset,indices,device,seed):
    report=[]
    for i in indices:
        x,y=load_source_example(dataset,i,device,seed)
        d=audit_one(model,safe,x,y,meta_probe=False)
        report.append({'index':i,'native':d['native_losses']['total'],
                       'adapted':d['updated_losses']['total'],
                       'delta':d['updated_losses']['total']-d['native_losses']['total'],
                       'cosine':d['gradients']['total']['cosine'],
                       'grad_norm':d['grad_norm'],
                       'logit_effects':d['prediction_effects'],
                       'scales':{k:{metric:d['updated_losses'].get(f'{k}_{metric}',0.0)-d['native_losses'].get(f'{k}_{metric}',0.0)
                              for metric in ('box','classification','positive_objectness')}
                              for k in ('small','medium','large')}})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--weights',choices=('ema','raw'),default='ema')
    p.add_argument('--train-indices',nargs='+',type=int,default=[0,1])
    p.add_argument('--holdout-indices',nargs='+',type=int,default=[10,37])
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--lr',type=float,default=0.0005)
    p.add_argument('--objective-mode',choices=('detector_total','scale_balanced'),default='detector_total')
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--basis-gain',type=float,default=1.0)
    p.add_argument('--step-policy',choices=('legacy_floor','normalized'),default='normalized')
    p.add_argument('--min-signal-norm',type=float,default=1e-8)
    p.add_argument('--device',default='auto')
    p.add_argument('--output',required=True)
    p.add_argument('--save-sidecar',default=None)
    args=p.parse_args()
    if args.steps<1 or args.steps>12:
        p.error('Pilot steps must be 1..12; this is not an epoch-training script')
    if not args.train_indices or not args.holdout_indices or set(args.train_indices)&set(args.holdout_indices):
        p.error('Need disjoint, nonempty train and heldout source-TRAIN indices')
    if args.lr<=0:p.error('Learning rate must be positive')
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():
        raise RuntimeError('Refusing source pilot: train_ann does not identify clean training data')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model,_=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    dataset=exp.get_dataset(cache=False)
    before_hash=record_model_hash(model)
    x,_=load_source_example(dataset,args.train_indices[0],device,args.seed)
    channels=get_size_from_native(model,x)
    torch.manual_seed(args.seed)
    safe=SafeSpatialTTT(channels,rank=args.rank,step_policy=args.step_policy,
             min_signal_norm=args.min_signal_norm,basis_gain=args.basis_gain).to(device).train()
    optimizer=torch.optim.AdamW(safe.parameters(),lr=args.lr,weight_decay=0.0)
    out={'protocol':{'checkpoint':args.checkpoint,'model_exp':args.model_exp,
                     'train_indices':args.train_indices,'holdout_indices':args.holdout_indices,
                     'steps':args.steps,'lr':args.lr,'rank':args.rank,'basis_gain':args.basis_gain,
                     'step_policy':args.step_policy,'weights':args.weights,
                     'objective_mode':args.objective_mode,'clean_train_only':True,
                     'native_sha256':before_hash,'fixed_native_matches':True},
          'before_holdout':evaluate_source_holdout(model,safe,dataset,args.holdout_indices,device,args.seed),
          'pilot_steps':[]}
    for step in range(args.steps):
        i=args.train_indices[step%len(args.train_indices)]
        images,labels=load_source_example(dataset,i,device,args.seed)
        result=fixed_native_targets_step(model,safe,images,labels,optimizer,
                                   objective_mode=args.objective_mode)
        result['step']=step+1;result['source_train_index']=i
        out['pilot_steps'].append(result)
        print('STEP',step+1,'TRAIN_IDX',i,'ADAPT_DET',result['adapted_detection_loss'],
              'META_PRED_GRAD',result['meta_predictor_grad_norm'],
              'TOTAL_GRAD',result['preclip_total_grad_norm'])
    out['after_holdout']=evaluate_source_holdout(model,safe,dataset,args.holdout_indices,device,args.seed)
    after_hash=record_model_hash(model)
    if before_hash!=after_hash:raise RuntimeError('Native YOLOX checkpoint mutated during meta pilot')
    out['native_model_mutations']=[]
    for a,b in zip(out['before_holdout'],out['after_holdout']):
        print('HOLDOUT_IDX',a['index'],'DELTA_BEFORE',a['delta'],'DELTA_AFTER',b['delta'],
              'COS_BEFORE',a['cosine'],'COS_AFTER',b['cosine'])
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print('SAVED',path)
    if args.save_sidecar:
        dst=Path(args.save_sidecar)
        if dst.exists():raise FileExistsError('Refusing to overwrite safe sidecar')
        dst.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'safe_ttt':safe.cpu().state_dict(),
                    'configuration':out['protocol'],'note':'source-only pilot sidecar, not a detector checkpoint'},dst)
        print('SAVED_SIDECAR',dst)
    print('NO CLAIM: a tiny source-label pilot cannot prove ACDC AP/AR improvement or robust unseen-domain abstention')

if __name__=='__main__':main()
