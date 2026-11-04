"""Small real validation-image smoke for raw detector, native/PGSMA delta, NMS.

This is NOT an AP/AR benchmark; for AP/AR use the complete COCO evaluator
on a fully trained and carefully taxonomy-mapped checkpoint.
"""
from __future__ import annotations
import argparse
import sys
import torch
from ._pgsma_preflight_common import (load_exp,load_torch_checkpoint,
    unwrap_state_dict,normalize_state_keys,checkpoint_coverage,dump_json)


def audit(args):
    exp=load_exp();exp.data_num_workers=0
    model=exp.get_model()
    ckpt_used=None
    if args.checkpoint:
        ckpt=load_torch_checkpoint(args.checkpoint)
        state,key=unwrap_state_dict(ckpt,args.prefer_ema)
        cov=checkpoint_coverage(model,normalize_state_keys(state))
        model.load_state_dict(cov['matched'],strict=False)
        ckpt_used={'source':args.checkpoint,'subkey':key,'matched_tensors':len(cov['matched'])}
        if (cov['groups']['backbone.backbone.dark2']['element_ratio'] or 0)<.85:
            raise RuntimeError('Pretrained checkpoint dark2 coverage too low')
    device=torch.device(args.device)
    model=model.to(device).eval()
    loader=exp.get_eval_loader(batch_size=args.batch_size,is_distributed=False,testdev=False)
    try:from yolox.utils import postprocess
    except ImportError as e:raise RuntimeError('Cannot import YOLOX yolox.utils.postprocess') from e
    it=iter(loader)
    output={'device':args.device,'checkpoint':ckpt_used,'test_size':list(exp.test_size),
            'batches':[],'status':'CHECKS_PASSED',
            'note':'No AP claims: outputs are raw real-image inference + NMS smoke only.'}
    for idx in range(args.batches):
        try:batch=next(it)
        except StopIteration:break
        img=batch[0].float().to(device)
        if img.ndim!=4 or img.shape[1]!=3:raise ValueError('Eval loader image shape invalid')
        b={'image_shape':list(img.shape)}
        preds={}
        with torch.inference_mode():
            for mode in ('native','pgsma'):
                model.set_inference_mode(mode)
                r=model(img)
                if r.ndim!=3 or r.shape[2]!=5+exp.num_classes:
                    raise ValueError(f'{mode} output shape {r.shape} inconsistent with K={exp.num_classes}')
                if not torch.isfinite(r).all():raise FloatingPointError(f'Nonfinite {mode} output')
                preds[mode]=r.float()
        a=preds['native'];b_1=preds['pgsma']
        if not torch.allclose(a[...,:5],b_1[...,:5],atol=0,rtol=0):
            raise AssertionError('PGSMA changed native boxes/objectness')
        b['box_objectness_exact_parity']=True
        b['max_class_probability_delta']=float((a[...,5:]-b_1[...,5:]).abs().max())
        b['output_shape']=list(a.shape)
        b['candidate_counts']={}
        for mode,pred in preds.items():
            # YOLOX postprocess receives already sigmoid-calibrated scores.
            det=postprocess(pred.detach().clone(),exp.num_classes,exp.test_conf,exp.nmsthre,
                            class_agnostic=False)
            if len(det)!=img.shape[0]:raise AssertionError('NMS output batch count mismatch')
            b['candidate_counts'][mode]=[int(d.shape[0]) if d is not None else 0 for d in det]
            for d in det:
                if d is not None and not torch.isfinite(d).all():
                    raise FloatingPointError(f'Nonfinite NMS {mode} detection')
        output['batches'].append(b)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--prefer-ema',action='store_true')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--batches',type=int,default=2)
    p.add_argument('--json',default=None)
    a=p.parse_args()
    try:res=audit(a)
    except Exception as exc:res={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
    dump_json(res,a.json)
    sys.exit(2 if res['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
