#!/usr/bin/env python3
"""Strict PRISM-TTT eval, TTT off/on/auto, true COCO evaluator AP/AR.

TTT-on uses unlabeled target features only. Does NOT train/tune on target data.
Off is legacy-dark2-TTT-disabled native vNext.2, not its old TTT-on setting.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from yolox.exp import get_exp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_ttt.bridge import PRISMYOLOX
from prism_ttt.core import PRISMTTT
from prism_ttt.legacy_router import Dark2AuditRouter
from prism_ttt.source_protocol import load_native,load_sidecar,native_sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--safe-weights',default=None)
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s.py')
    p.add_argument('--data-exp',default=None)
    p.add_argument('--weights',choices=['ema','raw'],default='ema')
    p.add_argument('--ttt-mode',choices=['off','on','auto'],default='off')
    p.add_argument('-b','--batch-size',type=int,default=1)
    p.add_argument('--conf',type=float,default=.001)
    p.add_argument('--nms',type=float,default=.60)
    p.add_argument('--device',default='auto')
    p.add_argument('--fp16',action='store_true')
    args=p.parse_args()
    if args.ttt_mode!='off' and not args.safe_weights:p.error('on/auto requires source-trained sidecar')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if args.fp16 and device.type!='cuda':p.error('FP16 requires CUDA')
    exp=get_exp(args.model_exp,None)
    if args.data_exp:
        dat=get_exp(args.data_exp,None)
        for key in ('data_dir','val_ann','test_ann','test_size'):
            setattr(exp,key,getattr(dat,key))
    exp.test_conf=args.conf;exp.nmsthre=args.nms
    native,ck=load_native(exp,args.checkpoint,device,args.weights)
    native.eval();native.head.training=False
    sha=native_sha256(native)
    # Channel shape from a native-only pass. No dummy sidecar forward.
    with Dark2AuditRouter(native) as router:
        router.set_mode('native')
        with torch.no_grad():fs=native.backbone(torch.zeros(1,3,*exp.test_size,device=device),ttt_prob=0.)
        router.require_one_hit(1)
    channels=tuple(int(x.shape[1]) for x in fs)
    if args.ttt_mode=='off':sidecar=PRISMTTT(channels).to(device).eval().requires_grad_(False)
    else:sidecar,_=load_sidecar(args.safe_weights,sha,channels,device,args.weights)
    bridge=PRISMYOLOX(native,sidecar,mode=args.ttt_mode,native_amp=args.fp16).eval()
    # Root head eval is required by the standard COCO evaluator.
    assert not bridge.head.training
    evaluator=exp.get_evaluator(args.batch_size,False,False,False)
    evaluator.per_class_AP=True;evaluator.per_class_AR=True
    print('PROTOCOL',json.dumps({'mode':args.ttt_mode,'source_sidecar':args.safe_weights,
       'native_checkpoint':args.checkpoint,'native_sha256':sha,'weights':args.weights,
       'test_data_exp':args.data_exp,'batch':args.batch_size,'conf':args.conf,
       'nms':args.nms,'fp16':args.fp16,'legacy_ttt':'disabled','amp_policy':'autocast_native_and_head_only'},sort_keys=True))
    # Evaluator is expected to use torch.no_grad(), NOT inference_mode().
    *_,summary=evaluator.evaluate(bridge,False,False,None,None,exp.test_size)
    assert native_sha256(native)==sha,'Native checkpoint state mutated in evaluation'
    print(summary)
    print('TTT_ACCEPTANCE',{'images_seen':int(bridge.images_seen.item()),
          'fraction_by_level':(bridge.accepted_levels/bridge.images_seen.clamp_min(1)).cpu().tolist(),
          'source_thresholds':sidecar.gate.thresholds.cpu().tolist()})
    if device.type=='cuda':print('CUDA_MAX_MEMORY_ALLOCATED_MB',torch.cuda.max_memory_allocated(device)/1048576)
    print('EVAL_DONE: AP/AR uses labels for METRICS only; unlabeled TTT has no GT access.')

if __name__=='__main__':main()
