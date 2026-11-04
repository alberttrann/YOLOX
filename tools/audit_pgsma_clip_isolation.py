"""Safe adversarial optimizer-gradient-clipping test; never starts model training.

It uses real optimizer ownership, synthetic deliberately huge semantic gradients,
and the patched actual trainer source. Does NOT load image data or weights.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import sys
import torch


def audit(device='cpu'):
    from yolox.core.trainer import Trainer
    src=inspect.getsource(Trainer.train_one_iter)
    for expected in ('self.semantic_optimizer.zero_grad(set_to_none=True)',
                     "for p in group['params'] if p.requires_grad",
                     'native_params, max_norm=10.0'):
        if expected not in src:
            raise AssertionError(f'Missing expected native clip isolation code: {expected}')
    dev=torch.device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    native=torch.nn.Parameter(torch.tensor([6.0,8.0],device=dev))
    semantic=torch.nn.Parameter(torch.tensor([3.0,4.0],device=dev))
    native_optim=torch.optim.SGD([native],lr=.1)
    semantic_optim=torch.optim.AdamW([semantic],lr=.001)
    native.grad=torch.tensor([6.0,8.0],device=dev)
    semantic.grad=torch.tensor([6e5,8e5],device=dev)
    native_params=[p for group in native_optim.param_groups for p in group['params'] if p.requires_grad]
    native_ids={id(p) for p in native_params}
    if id(semantic) in native_ids: raise AssertionError('Semantic parameter entered native optimizer')
    initial_grad=native.grad.detach().clone()
    native_norm=torch.nn.utils.clip_grad_norm_(native_params,max_norm=10.0,error_if_nonfinite=True)
    if not torch.allclose(native.grad,initial_grad,atol=2e-6):
        raise AssertionError('Semantic gradients influenced native clipping')
    semantic_optim.zero_grad(set_to_none=True)
    if semantic.grad is not None: raise AssertionError('Semantic grads were not cleared')
    # Force actual clipping with norm 20, still unaffected by semantic state.
    native.grad=torch.tensor([12.0,16.0],device=dev)
    semantic.grad=torch.tensor([6e5,8e5],device=dev)
    native_norm2=torch.nn.utils.clip_grad_norm_(native_params,max_norm=10.0,error_if_nonfinite=True)
    if not torch.allclose(native.grad,torch.tensor([6.,8.],device=dev),atol=3e-6):
        raise AssertionError('Native-only clipping did not produce the expected norm')
    return {'status':'CHECKS_PASSED','device':str(dev),
            'trainer_source_checked':True,'sem_grad_norm':1e6,
            'native_grad_norm_before_1':float(native_norm),
            'native_grad_norm_before_2':float(native_norm2),
            'native_grad_after_clip_2':native.grad.detach().cpu().tolist(),
            'note':'Synthetic extreme gradients; proves parameter selection and source contract, not real multi-step training.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--json',default='audits/clip_isolation.json')
    a=p.parse_args()
    try:report=audit(a.device)
    except Exception as exc:report={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
    dest=Path(a.json);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    sys.exit(2 if report['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
