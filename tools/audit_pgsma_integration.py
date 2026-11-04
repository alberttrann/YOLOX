"""Run INSIDE the user's complete YOLOX checkout after applying overlay.

No 80-epoch training. Exercises genuine upstream head/data dependencies and
checks native identity, semantic onset, one training step, dynamic shapes.
"""
import argparse
import copy
import json
import os
import sys
import torch


def audit(checkpoint=None, device='cpu'):
    from exps.default.tde_yolox_s_pgsma import Exp
    e=Exp();model=e.get_model().to(device)
    model.head.inference_semantic_mode='pgsma'
    results={'device':device,'class_count':model.num_classes,
             'registered_params':sum(p.numel() for p in model.parameters()),
             'native_params':sum(p.numel() for p in model.native_parameters()),
             'semantic_params':sum(p.numel() for p in model.semantic_parameters())}
    if checkpoint:
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        source=ckpt.get('model',ckpt)
        target=model.state_dict()
        matched={k:v for k,v in source.items() if k in target and target[k].shape==v.shape}
        # Baseline COCO checkpoint heads differ in class count. Print coverage by prefix.
        model.load_state_dict(matched, strict=False)
        results['checkpoint_coverage']={
            prefix:{'matched':sum(k.startswith(prefix) for k in matched),
                    'expected':sum(k.startswith(prefix) for k in target)}
            for prefix in ('backbone.backbone.stem','backbone.backbone.dark2',
                           'backbone.backbone.dark3','backbone.backbone.dark4',
                           'backbone.backbone.dark5','backbone.C3_',
                           'head.cls_convs','head.reg_convs')}
        if results['checkpoint_coverage']['backbone.backbone.dark2']['matched']==0:
            raise RuntimeError('No pretrained dark2 weights matched. Check checkpoint mapping.')
    assert not any('mask_token' in k or 'memory_banks' in k for k in model.state_dict())
    torch.manual_seed(15)
    model.train()
    x=torch.randn(2,3,64,96,device=device)
    y=torch.zeros(2,8,5,device=device)
    y[0,0]=torch.tensor([0,20,20,12,10],device=device)
    y[1,0]=torch.tensor([min(1,e.num_classes-1),48,30,10,12],device=device)
    native=model(x,y)
    results['native_loss']=float(native['total_loss'].detach())
    results['coverage']={k:int(native[k]) for k in ('num_fg_total','num_gt_total',
                                                   'uncovered_gt','core_positives')}
    assert results['coverage']['uncovered_gt']==0,results
    assert torch.isfinite(native['total_loss'])
    sem=model.semantic_losses(seed=101)
    results['semantic_losses']={k:float(v.detach()) for k,v in sem.items()}
    assert all(torch.isfinite(v) for v in sem.values())
    (sem['prototype']+sem['candidate']+sem['gate']+sem['teacher']).backward()
    assert all(p.grad is None for p in model.native_parameters()),'Semantic touched native gradients'
    model.zero_grad(set_to_none=True)
    native['total_loss'].backward()
    assert all(p.grad is None for p in model.semantic_parameters()),'Native touched semantic gradients'
    assert any(p.grad is not None for p in model.native_parameters())
    model.eval(); shapes=((64,96),(96,64),(128,128))
    for height,width in shapes:
        x=torch.randn(1,3,height,width,device=device)
        with torch.no_grad():
            model.set_inference_mode('native');n=model(x)
            model.set_inference_mode('pgsma');p=model(x)
        assert n.shape==p.shape and torch.isfinite(p).all()
        if int(model.semantic.semantic_step)==0:
            assert (n-p).abs().max()<1e-3, 'Initial PG-SMA not an identity'
        results.setdefault('shape_checks',[]).append([height,width,list(p.shape)])
    # A real optimizer-state equivalence test is included in the isolated
    # synthetic suite. Repeat on CUDA + AMP on the exact target checkout.
    results['status']='synthetic-upstream-preflight-pass'
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    result=audit(args.checkpoint,args.device)
    print(json.dumps(result,indent=2))
