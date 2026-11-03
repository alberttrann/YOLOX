"""Regression tests for vNext.2 evaluation-only per-image episodic TTT.

Run from YOLOX repo: python tests/test_vnext2_per_image_eval.py
No GPU, checkpoint, ACDC files or eval.py required.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get('VNEXT2_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
for pkg_name in ('yolox', 'yolox.models'):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []
    sys.modules[pkg_name] = pkg
for name in ('network_blocks', 'ttt_modules'):
    path = ROOT / 'yolox' / 'models' / (name+'.py')
    spec = importlib.util.spec_from_file_location('yolox.models.'+name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
TTT = sys.modules['yolox.models.ttt_modules'].TTTAdaptiveStage


def model():
    torch.manual_seed(101)
    t = TTT(nn.Conv2d(8, 8, 3, padding=1), 8, residual_adapter=True)
    t.set_epoch_state(2, 80)
    # Keep nonzero fast weights to rule out equality from zero initialization.
    with torch.no_grad():
        t.residual_scale.normal_(0, 0.03)
        t.residual_bias.normal_(0, 0.03)
        # Current projector is zero-last-layer initially; perturb for testing
        # so the adaptation gradients are not trivially from identity alone.
        t.projector.net[-1].weight.normal_(0, 0.005)
    return t


def closeness(a, b, atol=2e-6, rtol=2e-6):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        float((a-b).abs().max()), float((a-b).square().mean().sqrt()))


def test_frozen_eval_no_grad_and_state_immutable():
    t = model().eval().requires_grad_(False)
    orig = {k: v.clone() for k,v in t.state_dict().items()}
    A = torch.randn(1,8,12,16)
    with torch.no_grad():
        out = t(A, run_ttt=True)
    assert out.shape == A.shape and bool(torch.isfinite(out).all())
    assert not out.requires_grad
    assert all(torch.equal(v,orig[k]) for k,v in t.state_dict().items())


def test_eval_companion_and_permutation_independence():
    t = model().eval().requires_grad_(False)
    A = torch.randn(1,8,12,16)
    B = torch.randn(1,8,12,16) + 0.4
    with torch.no_grad():
        aa = t(torch.cat((A,A),0), run_ttt=True)
        ab = t(torch.cat((A,B),0), run_ttt=True)
        ba = t(torch.cat((B,A),0), run_ttt=True)
        bb = t(torch.cat((B,B),0), run_ttt=True)
    closeness(ab[0],aa[0]); closeness(ab[1],bb[1])
    closeness(ab[0],ba[1]); closeness(ab[1],ba[0])
    # Native head output does not participate in this direct stage test;
    # those inference results must be verified on the full model separately.


def test_batch_one_parity_against_unmodified_shared_fast_update():
    t = model().eval().requires_grad_(False)
    A = torch.randn(1,8,12,16)
    # The old shared-B implementation has equivalent mathematics for B=1
    # if it uses ephemeral frozen EMA leaves. Call it as a control.
    with torch.no_grad():
        out_new = t(A, run_ttt=True)
        t.eval_ttt_batch_policy = 'shared'
        # The base source snapshot predates the frozen-EMA hotfix. Temporarily
        # unfreeze only these two local fast-weight params for legacy parity;
        # the real evaluator remains fully frozen with per-image adaptation.
        t.residual_scale.requires_grad_(True)
        t.residual_bias.requires_grad_(True)
        try:
            out_old = t(A, run_ttt=True)
        finally:
            t.residual_scale.requires_grad_(False)
            t.residual_bias.requires_grad_(False)
            t.eval_ttt_batch_policy = 'per_image'
    closeness(out_new, out_old, atol=3e-6, rtol=3e-6)



def test_legacy_shared_batch_coupling_is_detectable():
    # Positive control: the old shared update must FAIL independence for
    # intentionally different companion images. Its frozen-EMA hotfix is not
    # present in the isolated source snapshot, so only fast params are unfrozen.
    t = model().eval().requires_grad_(False)
    t.eval_ttt_batch_policy = 'shared'
    t.residual_scale.requires_grad_(True)
    t.residual_bias.requires_grad_(True)
    A = torch.randn(1,8,12,16)
    B = torch.randn(1,8,12,16) * 3 + 2
    with torch.no_grad():
        aa = t(torch.cat([A,A],0),run_ttt=True)
        ab = t(torch.cat([A,B],0),run_ttt=True)
    effect = float((aa[:1]-ab[:1]).abs().max())
    assert effect > 0, 'Positive-control shared-batch coupling was not detected'


def test_training_backward_preserves_meta_parameter_gradients():
    t = model().train()
    A = torch.randn(2,8,12,16, requires_grad=True)
    t.eval_ttt_batch_policy = 'per_image'  # must NOT affect training branch
    y = t(A, run_ttt=True)
    objective = y.square().mean() + t.current_rec_loss
    objective.backward()
    for key, param in [
        ('backbone', t.backbone_stage.weight),
        ('scale',t.residual_scale),('bias',t.residual_bias),
        ('meta lr', t.ttt_lrs['residual_scale']),
        ('projector',t.projector.mask_token),
    ]:
        assert param.grad is not None and bool(torch.isfinite(param.grad).all()),key
    assert t.backbone_stage.weight.grad.abs().sum()>0


def test_no_ttt_identity_and_inference_mode_guard():
    t = model().eval().requires_grad_(False)
    A = torch.randn(1,8,12,16)
    with torch.no_grad():
        closeness(t(A, run_ttt=False),t.backbone_stage(A),atol=0,rtol=0)
    try:
        with torch.inference_mode():
            t(A,run_ttt=True)
    except RuntimeError as ex:
        assert 'inference_mode' in str(ex)
    else:
        raise AssertionError('TTT unexpectedly ran inside inference_mode')


if __name__=='__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            f()
            print('PASS',name)
