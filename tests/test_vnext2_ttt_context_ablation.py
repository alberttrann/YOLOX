"""CPU regression tests for opt-in vNext.2 spatial-reconstruction experiment.

Requires existing tests/test_vnext2_per_image_eval.py and your YOLOX source.
No checkpoint or ACDC downloads required.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
import os
YOLOX = Path(os.environ.get('VNEXT2_SOURCE_ROOT', ROOT))

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Reuse the exact vNext.2 class and mocked import bootstrap in the existing test.
existing = load('test_vnext2_per_image_eval', YOLOX/'tests'/'test_vnext2_per_image_eval.py')
ctx = load('vnext2_ttt_context_ablation', ROOT/'yolox'/'models'/'ttt_context_ablation.py')

class MockDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.backbone=nn.Module()
        self.backbone.backbone.dark2=existing.model()


def fresh(mode='context_mse'):
    m=MockDetector()
    ctx.install_contextual_ttt(m,mode)
    return m,m.backbone.backbone.dark2


def test_baseline_not_modified_and_strict_checkpoint_compatible():
    torch.manual_seed(11)
    a=MockDetector()
    keys_before=list(a.state_dict())
    b=copy.deepcopy(a)
    ctx.install_contextual_ttt(b,'context_mse')
    assert list(b.state_dict())==keys_before
    b.load_state_dict(a.state_dict(),strict=True)
    assert not hasattr(a.backbone.backbone.dark2,'ttt_context_mode')
    ctx.install_contextual_ttt(b,'context_mse')
    try:
        ctx.install_contextual_ttt(b,'context_balanced')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Allowed mode change on already-installed model')


def test_context_no_direct_masked_value_leakage():
    obs=torch.randn(2,8,10,14)
    mask=torch.zeros(2,1,10,14)
    mask[:,:,2:6,2:6]=1
    token=torch.zeros(1,8,1,1)
    a,_=ctx.contextual_corruption(obs,mask,token,use_context=True)
    perturbed=obs+mask*torch.randn_like(obs)*80
    b,_=ctx.contextual_corruption(perturbed,mask,token,use_context=True)
    assert torch.equal(a,b)
    assert torch.isfinite(a).all()
    zero_mask=torch.ones_like(mask)
    o,_=ctx.contextual_corruption(obs,zero_mask,token,use_context=True)
    assert torch.equal(o,token.expand_as(o))


def test_balanced_region_loss_per_image_count_invariance():
    target=torch.zeros(2,8,4,4)
    pred=torch.ones_like(target)
    mask=torch.zeros(2,1,4,4)
    mask[0,:,0,0]=1
    mask[1,:,:2,:2]=1
    losses=ctx.per_image_region_loss(pred,target,mask,balanced=True)
    assert torch.allclose(losses,torch.ones_like(losses))
    full=ctx.per_image_region_loss(pred,target,mask,balanced=False)
    assert torch.allclose(full,torch.ones_like(full))


def test_eval_batch_independent_state_safe_and_deepcopy():
    torch.manual_seed(7)
    m,t=fresh('context_mse')
    m.eval().requires_grad_(False)
    saved={k:v.detach().clone() for k,v in m.state_dict().items()}
    A=torch.randn(1,8,12,16)
    B=torch.randn(1,8,12,16)+0.6
    with torch.no_grad():
        aa=t(torch.cat((A,A),0))
        ab=t(torch.cat((A,B),0))
        ba=t(torch.cat((B,A),0))
    assert torch.allclose(aa[0],ab[0],atol=2e-6,rtol=2e-6)
    assert torch.allclose(aa[0],ba[1],atol=2e-6,rtol=2e-6)
    assert all(torch.equal(v,saved[k]) for k,v in m.state_dict().items())
    assert not ab.requires_grad
    clone=copy.deepcopy(m)
    assert clone.backbone.backbone.dark2._forward_residual_ttt.__self__ is clone.backbone.backbone.dark2
    with torch.no_grad():
        out=clone.backbone.backbone.dark2(A)
    assert torch.isfinite(out).all()


def test_training_gradients_all_modes():
    for mode in sorted(ctx._MODES):
        torch.manual_seed(19)
        m,t=fresh(mode)
        m.train()
        x=torch.randn(2,8,12,16,requires_grad=True)
        out=t(x,run_ttt=True)
        loss=out.square().mean()+t.current_rec_loss
        loss.backward()
        for key,param in [('native',t.backbone_stage.weight),('scale',t.residual_scale),
                          ('bias',t.residual_bias),('meta_lr',t.ttt_lrs['residual_scale']),
                          ('projector',t.projector.net[-1].weight)]:
            assert param.grad is not None and bool(torch.isfinite(param.grad).all()),(mode,key)
        assert t.backbone_stage.weight.grad.abs().sum()>0
        assert t.projector.net[-1].weight.grad.abs().sum()>0
        assert float(t.current_rec_loss.detach())>0


def test_native_identity_and_inference_guard():
    m,t=fresh()
    m.eval().requires_grad_(False)
    x=torch.randn(1,8,12,16)
    with torch.no_grad():
        assert torch.equal(t(x,run_ttt=False),t.backbone_stage(x))
    try:
        with torch.inference_mode():
            t(x,run_ttt=True)
    except RuntimeError as ex:
        assert 'inference_mode' in str(ex)
    else:
        raise AssertionError('Accepted inference_mode TTT')

if __name__ == '__main__':
    torch.set_num_threads(1)
    for name, func in list(globals().items()):
        if name.startswith('test_') and callable(func):
            func()
            print('PASS',name)
