#!/usr/bin/env python3
"""Clean DG/TTT inference evaluation with strict native model and safe sidecar.

Requires prior source-only meta pilot to export the SafeSpatialTTT sidecar.
No ACDC labels are ever used inside online adaptation. The evaluator may use
labeled validation annotations for AP/AR *only*, never for training or updates.

Usage after source-only pilot, fixed final protocol:
 python tools/eval_safe_ttt_sidecar.py `
   --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
   --safe-weights audits/safe_ttt_source_sidecar.pth `
   --model-exp exps/default/tde_yolox_s.py `
   --data-exp exps/default/test_acdc_fog.py --ttt-mode on -b 1

This is single GPU / CPU evaluator, not DDP or TensorRT deployment.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from yolox.exp import get_exp
from safe_ttt_model import SafeSpatialTTT
from native_dark2_router import Dark2AuditRouter
from audit_safe_ttt_yolox import setup_model,get_size_from_native,record_model_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--safe-weights',default=None)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--data-exp',default=None)
    p.add_argument('--weights',choices=('ema','raw'),default='ema')
    p.add_argument('--ttt-mode',choices=('off','on'),default='off')
    p.add_argument('-b','--batch-size',type=int,default=1)
    p.add_argument('--conf',type=float,default=.001)
    p.add_argument('--nms',type=float,default=.60)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--device',default='auto')
    p.add_argument('--fp16',action='store_true')
    args=p.parse_args()
    if args.ttt_mode=='on' and not args.safe_weights:
        p.error('TTT on requires a source-trained safe sidecar; random online TTT is not a valid accuracy benchmark')
    if args.batch_size<1 or not (0<=args.conf<=1) or not(0<args.nms<=1):
        p.error('Invalid batch size/confidence/NMS')
    if args.fp16 and not torch.cuda.is_available():
        p.error('FP16 evaluation is intended for CUDA here')
    if args.device!='auto' and args.device.startswith('cpu') and args.fp16:
        p.error('FP16 CPU evaluation unsupported')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    exp=get_exp(args.model_exp,None)
    data_exp=get_exp(args.data_exp,None) if args.data_exp else None
    if data_exp is not None:
        for key in ('data_dir','val_ann','test_ann','test_size'):
            setattr(exp,key,getattr(data_exp,key))
    exp.test_conf=args.conf
    exp.nmsthre=args.nms
    model,_=setup_model(exp,args.checkpoint,device,args.weights,args.seed)
    model.head.training=False
    model.eval()
    safe=None
    if args.ttt_mode=='on':
        ck=torch.load(args.safe_weights,map_location='cpu',weights_only=False)
        if not isinstance(ck,dict) or not isinstance(ck.get('safe_ttt'),dict) or not isinstance(ck.get('configuration'),dict):
            raise RuntimeError('Safe sidecar file must contain safe_ttt and source-only configuration')
        config=ck['configuration']
        if not config.get('clean_train_only'):
            raise RuntimeError('Refusing sidecar without explicit source-clean provenance')
        if os.path.normcase(os.path.abspath(config['checkpoint']))!=os.path.normcase(os.path.abspath(args.checkpoint)):
            raise RuntimeError('Sidecar was source-trained against a different native checkpoint')
        if config.get('native_sha256') is not None and record_model_hash(model)!=config['native_sha256']:
            raise RuntimeError('Native checkpoint hash changed since the sidecar was trained')
        if config.get('weights')!=args.weights:
            raise RuntimeError('Sidecar native EMA/raw provenance mismatch')
        C=get_size_from_native(model,torch.zeros(1,3,*exp.test_size,device=device,dtype=torch.float32))
        safe=SafeSpatialTTT(C,rank=int(config['rank']),
               step_policy=config['step_policy'],basis_gain=float(config['basis_gain']),
               min_signal_norm=float(config.get('min_signal_norm',1e-8))).to(device)
        safe.load_state_dict(ck['safe_ttt'],strict=True)
        safe.eval().requires_grad_(False)
    evaluator=exp.get_evaluator(args.batch_size,False,False,False)
    evaluator.per_class_AP=True
    evaluator.per_class_AR=True
    print('PROTOCOL',json.dumps({'native_checkpoint':args.checkpoint,'safe_weights':args.safe_weights,
           'mode':args.ttt_mode,'weights':args.weights,'data_exp':args.data_exp,
           'conf':args.conf,'nms':args.nms,'batch_size':args.batch_size,'device':str(device),
           'fp16':args.fp16},sort_keys=True))
    with Dark2AuditRouter(model,safe) as router:
        router.set_mode('episode' if args.ttt_mode=='on' else 'native',meta_train=False)
        *_,summary=evaluator.evaluate(model,False,args.fp16,None,None,exp.test_size)
        if router.hit<1:
            raise RuntimeError('No dark2 native route was intercepted. Evaluation invalid.')
    print(summary)
    print('EVAL_DONE: This evaluates AP/AR but does not provide online target labels to TTT.')

if __name__=='__main__':main()
