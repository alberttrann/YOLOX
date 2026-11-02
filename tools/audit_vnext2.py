"""tdevNext vNext.2 reproducible architectural inventory & gradient probe.
Run from repository root:
    python tools/audit_vnext2.py --checkpoint weights/yolox_s.pth
    python tools/audit_vnext2.py --device cuda --probe --height 128 --width 256 --batch-size 2
No datasets or annotations required for the synthetic probe.
"""
import argparse
import json
from pathlib import Path

import torch
from yolox.exp import get_exp


def inspect_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    remapped = dict(state)
    for k, v in state.items():
        if 'dark2.' in k and 'backbone_stage' not in k:
            remapped[k.replace('dark2.', 'dark2.backbone_stage.')] = v
    current = model.state_dict()
    compatible = {k for k in current if k in remapped and current[k].shape == remapped[k].shape}
    elems = sum(current[k].numel() for k in compatible)
    count = sum(v.numel() for v in current.values())
    print(f'PRETRAIN_COVERAGE matched={elems:,}/{count:,} ({100*elems/max(1,count):.2f}%)')
    for prefix in [
        'backbone.backbone.stem','backbone.backbone.dark2.backbone_stage',
        'backbone.backbone.dark3','backbone.backbone.dark4',
        'backbone.backbone.dark5','backbone.C3_p4','backbone.C3_p3',
        'backbone.C3_n3','backbone.C3_n4','head.stems','head.reg_preds',
        'head.obj_preds','head.cls_preds']:
        keys=[k for k in current if k.startswith(prefix)]
        matches=sum(k in compatible for k in keys)
        print(f'  {prefix:<52} {matches:>4}/{len(keys):<4}')
    return compatible


def summarize_gradients(model):
    buckets={
        'TTT residual':'backbone.backbone.dark2.residual_',
        'TTT projector':'backbone.backbone.dark2.projector.',
        'SPI':'backbone.spi_highway.',
        'Tribrid p3':'backbone.C3_p3.',
        'Tribrid p4':'backbone.C3_p4.',
        'Tribrid n3':'backbone.C3_n3.',
        'Tribrid n4':'backbone.C3_n4.',
        'Backbone dark4':'backbone.backbone.dark4.',
        'Backbone dark5':'backbone.backbone.dark5.',
        'Objectness':'head.obj_preds.',
        'Regression':'head.reg_preds.',
        'Classification':'head.cls_preds.',
        'Memory bank':'head.memory_banks.',
        'Latent projector':'head.latent_projectors.',
        'Affordance':'affordance_head.',
    }
    for label,prefix in buckets.items():
        ps=[p for n,p in model.named_parameters() if n.startswith(prefix)]
        count=sum(p.numel() for p in ps)
        grads=[p.grad for p in ps if p.grad is not None]
        all_finite=all(torch.isfinite(g).all().item() for g in grads)
        l1=sum(g.detach().float().abs().sum().item() for g in grads)
        print(f'{label:<24} params={count:>10,} grad_tensors={len(grads):>3} '
              f'finite={all_finite} grad_abs_sum={l1:.6g}')


def cosines(model, outputs):
    prefix_sets=[
        ('P3 neck', 'backbone.C3_p3.'),
        ('P4 neck', 'backbone.C3_p4.'),
        ('dark4', 'backbone.backbone.dark4.'),
        ('dark5', 'backbone.backbone.dark5.'),
    ]
    for label,prefix in prefix_sets:
        params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if not params: continue
        gd=torch.autograd.grad(outputs['det_loss'],params,retain_graph=True,allow_unused=True)
        ga=torch.autograd.grad(outputs['aux_total'],params,retain_graph=True,allow_unused=True)
        dot=torch.zeros((),device=outputs['det_loss'].device)
        dd=dot.clone();aa=dot.clone()
        for d,a in zip(gd,ga):
            if d is None or a is None:continue
            d=d.detach().float();a=a.detach().float()
            dot+=torch.sum(d*a);dd+=torch.sum(d*d);aa+=torch.sum(a*a)
        cosine=(dot/(torch.sqrt(dd*aa)+1e-12)).item()
        print(f'GRAD_COSINE {label:<18} {cosine:+.5f} det={dd.sqrt().item():.5g} aux={aa.sqrt().item():.5g}')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exp',default='exps/default/tde_yolox_s.py')
    ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--probe',action='store_true')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--height',type=int,default=128)
    ap.add_argument('--width',type=int,default=256)
    ap.add_argument('--batch-size',type=int,default=2)
    args=ap.parse_args()
    torch.manual_seed(7119)
    exp=get_exp(args.exp,None)
    model=exp.get_model()
    ps=list(model.named_parameters())
    print('TOTAL_REGISTERED',sum(p.numel() for _,p in ps))
    print('TOTAL_TRAINABLE',sum(p.numel() for _,p in ps if p.requires_grad))
    for key in ('enable_ttt','enable_spi','enable_global_context','enable_dense_local',
                'enable_backbone_attention','enable_memory','enable_memory_aux',
                'enable_proto_aux','enable_intra_aux','enable_rec_aux','enable_afford_aux'):
        print('FLAG',key,getattr(exp,key,'undefined'))
    if args.checkpoint:
        inspect_checkpoint(model,args.checkpoint)
    if not args.probe:return
    model.to(args.device).train()
    model.set_meta_training_state(1,exp.max_epoch)
    x=torch.randn(args.batch_size,3,args.height,args.width,device=args.device)
    y=torch.zeros(args.batch_size,120,5,device=args.device)
    for i in range(args.batch_size):
        y[i,0]=torch.tensor([i%9,args.width*(0.3+0.1*(i%3)),
                             args.height*(0.3+0.1*(i%3)),
                             args.width*0.15,args.height*0.12],device=args.device)
    out=model(x,y)
    print('FORWARD_LOSSES',json.dumps({k:float(v.detach()) for k,v in out.items() if torch.is_tensor(v) and v.numel()==1}))
    if not torch.isfinite(out['total_loss']):
        raise FloatingPointError('Full forward loss nonfinite')
    cosines(model,out)
    out['total_loss'].backward()
    summarize_gradients(model)

if __name__=='__main__':
    main()
