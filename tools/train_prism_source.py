#!/usr/bin/env python3
"""Train ONE PRISM-TTT v1 sidecar on source BDD clean TRAIN only.

Frozen native checkpoint, one exact differentiable inner step, fixed native
matches, GT-aware outer ranking/coverage/scale/background risk, learned source
reliability, source gradient/gate calibration, disjoint source holdout.

Place prism_ttt/ and this script in YOLOX root, run from root. Does NOT train
or alter the YOLOX model; creates separate PRISM checkpoint/report.
"""
from __future__ import annotations
import argparse,json,random,sys,os,math
from pathlib import Path
import torch
from yolox.exp import get_exp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_ttt.bridge import PRISMYOLOX
from prism_ttt.core import PRISMTTT,VERSION
from prism_ttt.meta import source_pair_step,source_view_result,optimizer_step,parameter_groups,source_risk_gate_loss
from prism_ttt.source_protocol import (load_native,native_sha256,source_train_example,
                                       source_shift,save_sidecar,gate_calibrate_from_reports)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--weights',choices=['ema','raw'],default='ema')
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--train-start',type=int,default=300)
    p.add_argument('--train-count',type=int,default=256)
    p.add_argument('--calib-start',type=int,default=2000)
    p.add_argument('--calib-count',type=int,default=64)
    p.add_argument('--holdout-start',type=int,default=3000)
    p.add_argument('--holdout-count',type=int,default=64)
    p.add_argument('--episodes-per-step',type=int,default=4)
    p.add_argument('--lr',type=float,default=0.00005)
    p.add_argument('--allow-early-checkpoint-diagnostic',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--report',required=True)
    args=p.parse_args()
    if args.train_count<2 or args.train_count%args.episodes_per_step or args.calib_count<16 or args.holdout_count<8:
        p.error('Training must be divisible by episodes/step; calibration>=16, holdout>=8')
    if not(0<args.lr<.01):p.error('Invalid learning rate')
    groups=[set(range(start,start+count)) for start,count in ((args.train_start,args.train_count),
             (args.calib_start,args.calib_count),(args.holdout_start,args.holdout_count))]
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):p.error('TRAIN / CALIB / HOLDOUT IDs overlap')
    exp=get_exp(args.model_exp,None)
    if 'clean' not in str(exp.train_ann).lower():raise RuntimeError('Source train annotation must be clearly CLEAN')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    native,ck=load_native(exp,args.checkpoint,device,args.weights,args.seed)
    epoch=int(ck.get('start_epoch',1))
    if epoch<20 and not args.allow_early_checkpoint_diagnostic:
        raise RuntimeError('Epoch<20 considered diagnostic-only. Use --allow-early-checkpoint-diagnostic to acknowledge immature detector')
    dataset=exp.get_dataset(cache=False)
    if any(i<0 or i>=len(dataset) for g in groups for i in g):raise IndexError('Source index out of bounds')
    native_hash=native_sha256(native)
    native.head.training=True
    # Infer actual runtime P3/P4/P5 channels. Never hardcode channels.
    x0,_=source_train_example(dataset,args.train_start,device,args.seed)
    from prism_ttt.legacy_router import Dark2AuditRouter
    with Dark2AuditRouter(native) as router:
        router.set_mode('native')
        with torch.no_grad():f0=native.backbone(x0,ttt_prob=0.)
        router.require_one_hit(1)
    channels=tuple(int(f.shape[1]) for f in f0)
    del f0
    if len(channels)!=3:raise RuntimeError('Unexpected number of FPN features')
    torch.manual_seed(args.seed)
    sidecar=PRISMTTT(channels).to(device).train()
    bridge=PRISMYOLOX(native,sidecar)
    source_indices={'train':[args.train_start,args.train_count],
      'calibrate':[args.calib_start,args.calib_count],
      'holdout':[args.holdout_start,args.holdout_count]}

    # Offline gradient-magnitude calibration from source calibration scenes.
    # This uses no labels or adverse target conditions.
    observed=[]
    for i in range(args.calib_start,args.calib_start+args.calib_count):
        x,_=source_train_example(dataset,i,device,args.seed)
        with torch.no_grad():fs=bridge.native_features(x)
        sidecar.eval()
        with torch.no_grad():prop=sidecar.propose(fs,meta_train=False)
        observed.append(prop.grad.detach().cpu())
    sidecar.source_calibrate_gradients(torch.cat(observed,0))
    sidecar.train()
    print('CALIB source gradient scale',float(sidecar.gradient_scale.median()))

    optimizer=torch.optim.AdamW(parameter_groups(sidecar,args.lr),weight_decay=1e-4)
    trace=[]; optimizer.zero_grad(set_to_none=True)
    from itertools import islice
    steps=args.train_count//args.episodes_per_step
    for step in range(steps):
        rows=[]
        for t in range(args.episodes_per_step):
            i=args.train_start+step*args.episodes_per_step+t
            x,y=source_train_example(dataset,i,device,args.seed)
            view=source_shift(x,i,args.seed,kind='train')
            rows.extend(source_pair_step(bridge,x,view,y,accum=args.episodes_per_step,
                              clean_weight=1.,gate_weight=.1,tail_weight=.5,bg_weight=.5))
        grad=optimizer_step(sidecar,optimizer,max_norm=.5)
        trace.append({'step':step+1,'gradient_norm':grad,
            'clean_delta_mean':sum(r['delta'] for r in rows if r['view']=='clean')/args.episodes_per_step,
            'shifted_delta_mean':sum(r['delta'] for r in rows if r['view']=='shifted')/args.episodes_per_step})
        print('STEP',step+1,'GRAD',round(grad,7),
              'CLEAN',round(trace[-1]['clean_delta_mean'],7),
              'SHIFT',round(trace[-1]['shifted_delta_mean'],7),flush=True)
    if native_sha256(native)!=native_hash:raise RuntimeError('Native checkpoint state mutated')

    # Strict source-only gate calibration: disjoint from training and holdout.
    records=[]
    sidecar.eval()
    for i in range(args.calib_start,args.calib_start+args.calib_count):
        x,y=source_train_example(dataset,i,device,args.seed)
        shifted=source_shift(x,i,args.seed,kind='train')
        for view in (x,shifted):
            r=source_view_result(bridge,view,y,meta_train=False)
            _,gd=source_risk_gate_loss(bridge,r)
            records.append((gd['gate_scores'],gd['labels'],gd['present']))
    cal=gate_calibrate_from_reports(sidecar,records,min_examples=16)
    print('SOURCE_GATE_CALIB',cal)
    if native_sha256(native)!=native_hash:raise RuntimeError('Native changed during policy calibration')

    # Heldout evaluation: evaluate BOTH clean and a withheld blur family,
    # no optimizer steps / threshold fitting on the holdout examples.
    validation=[]
    for i in range(args.holdout_start,args.holdout_start+args.holdout_count):
        x,y=source_train_example(dataset,i,device,args.seed)
        for name,view in [('clean',x),('heldout_appearance',source_shift(x,i,args.seed,kind='heldout'))]:
          for policy in ('on','auto'):
            r=source_view_result(bridge,view,y,meta_train=False,mode=policy)
            diff=r['parts'];gt=r['native_proxy']['gt_support'];bg=float(diff['background_delta'].detach())
            validation.append({'index':i,'view':name,'policy':policy,
                'accept_fraction':float(r['gate_applied'].mean()) if r['gate_applied'] is not None else None,
                'native':float(r['native']['total'].detach()),
                'adapted':float(r['adapted']['total'].detach()),
                'delta':float(diff['total_delta'].detach()),
                'background_delta':bg,'gt_support':gt,
                'scale_proxy_delta':{s:(float((r['adapted_proxy']['scale_proxy'][s]-
                                                r['native_proxy']['scale_proxy'][s]).detach())
                                              if gt[s]>0 else None) for s in ('small','medium','large')}})
    def summary(view,policy):
        rows=[r for r in validation if r['view']==view and r['policy']==policy]
        return {'n':len(rows),'mean_delta':sum(r['delta'] for r in rows)/len(rows),
            'helpful':sum(r['delta'] < -2e-5 for r in rows),
            'harmful':sum(r['delta'] > 2e-5 for r in rows),
            'bg_harm':sum(r['background_delta']>2e-5 for r in rows),
            'mean_accept_fraction':sum(r['accept_fraction'] for r in rows)/len(rows),
            'scale_image_coverage':{s:sum(r['gt_support'][s]>0 for r in rows)
                  for s in ('small','medium','large')}}
    results={'version':VERSION,'source_only':True,'checkpoint_epoch':epoch,
      'native_sha256':native_hash,'channels':channels,
      'indices':source_indices,'train_steps':trace,'calibration':cal,
      'source_holdout':{v:{pol:summary(v,pol) for pol in ('on','auto')}
                             for v in ('clean','heldout_appearance')},
      'holdout_records':validation,'UNPROVEN':['AP/AR','unseen-adverse-DG','OOD gating safety','runtime']}
    if native_sha256(native)!=native_hash:raise RuntimeError('Frozen native changed after training')
    out=Path(args.report);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Refusing to overwrite source research report')
    out.write_text(json.dumps(results,indent=2,allow_nan=False))
    save_sidecar(args.output,sidecar,native_hash=native_hash,native_checkpoint=args.checkpoint,
          native_weights=args.weights,epoch=epoch,seed=args.seed,
          source_indices=source_indices,metrics=results['source_holdout'])
    print('SAVED_SIDECAR',args.output,'SAVED_REPORT',args.report)
    print('Source surrogate results ONLY. Full AP/AR and unbiased OOD evaluation remain unverified.')

if __name__=='__main__':main()
