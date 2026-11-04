"""Real-checkout semantic gradient-route probe: no full detector training.

Reports level-by-level predictor candidate/JEPA gradient magnitudes/cosines.
Run with the *actual* upstream YOLOX source (not the bundled test double).
"""
import argparse
import json
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--height',type=int,default=96)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--batch',type=int,default=2)
    args=p.parse_args()
    if args.height%32 or args.width%32:
        raise ValueError('Height and width must be divisible by 32')
    from exps.default.tde_yolox_s_pgsma import Exp
    exp=Exp();model=exp.get_model().to(args.device).train()
    torch.manual_seed(71)
    images=torch.randn(args.batch,3,args.height,args.width,device=args.device)
    labels=torch.zeros(args.batch,8,5,device=args.device)
    for b in range(args.batch):
        labels[b,0]=torch.tensor([b%exp.num_classes,args.width*.5,
                                   args.height*.5,16,12],device=args.device)
    native=model(images,labels)
    sem=model.semantic_losses(seed=3001)
    result={'native_loss':float(native['total_loss'].detach()),
            'semantic_losses':{k:float(v.detach()) for k,v in sem.items()},
            'levels':{}}
    for i,predictor in enumerate(model.semantic.predictors):
        params=list(predictor.parameters())
        grads={}
        for name in ('teacher','candidate'):
            raw=torch.autograd.grad(sem[name],params,retain_graph=True,allow_unused=True)
            grads[name]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,raw)]
        c=grads['candidate'];j=grads['teacher']
        dot=sum((x*y).sum() for x,y in zip(c,j)).item()
        cn=sum(x.square().sum() for x in c).sqrt().item()
        jn=sum(x.square().sum() for x in j).sqrt().item()
        result['levels'][f'P{i+3}']={'candidate_grad_norm':cn,
            'teacher_grad_norm':jn,'gradient_cosine':dot/(cn*jn+1e-16),
            'terminal_candidate_grad_norm':c[-2].norm().item()}
    print(json.dumps(result,indent=2))
    if not all(torch.isfinite(v) for v in sem.values()):
        raise FloatingPointError('Nonfinite semantic objective')
    if not any(row['terminal_candidate_grad_norm']>1e-12 for row in result['levels'].values()):
        raise RuntimeError('No useful candidate gradient to any predictor terminal layer')

if __name__=='__main__':main()
