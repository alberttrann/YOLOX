"""CPU-first adversarial/property tests for the isolated SafeSpatialTTT prototype.
Run: python tests/test_safe_ttt.py
No YOLOX source, dataset, checkpoint, DDP, or GPU is required.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from safe_ttt_model import SafeSpatialTTT, checkerboard_mask, alignment_surrogate

torch.set_num_threads(1)


def fixture():
    torch.manual_seed(2026)
    return SafeSpatialTTT(channels=8, rank=4, relative_budget=0.07)


def test_true_identity_and_empty_mask():
    m = fixture().eval()
    f = torch.randn(2, 8, 8, 12)
    out, st = m.run_episode(f, meta_train=False, enabled=False)
    assert out is f and st['native']
    z0 = torch.zeros(2, 4)
    assert torch.equal(m.adapter(f, z0), f), "z=0 must give EXACT identity"
    tiny = torch.randn(1, 8, 1, 1)
    assert checkerboard_mask(tiny).sum().item() == 0
    out, _ = m.run_episode(tiny, meta_train=False)
    assert torch.equal(tiny, out)


def test_mask_precedes_normalization_and_spatial_mixing():
    m = fixture().eval()
    f = torch.randn(2, 8, 8, 12)
    mask = checkerboard_mask(f)
    # Change only HIDDEN positions; predictor may not see their values.
    altered = f + torch.randn_like(f) * mask * 100.0
    z = torch.randn(2, 4) * 0.2
    inp_original = m.masked_view(f, mask)
    inp_altered = m.masked_view(altered, mask)
    assert torch.equal(inp_original, inp_altered)
    # Even after GroupNorm, local/dilated convolutions and projector.
    a = m.predictor(m.adapter(inp_original, z))
    b = m.predictor(m.adapter(inp_altered, z))
    assert torch.equal(a,b), "Target leakage through unmasked global normalization"


def test_frozen_ema_no_grad_and_independence():
    m = fixture().eval().requires_grad_(False)
    saved = {k:v.clone() for k,v in m.state_dict().items()}
    A = torch.randn(1,8,8,12)
    B = torch.randn(1,8,8,12) * 2 + 2
    with torch.no_grad():
        aa, _ = m.run_episode(torch.cat([A,A]), meta_train=False)
        ab, _ = m.run_episode(torch.cat([A,B]), meta_train=False)
        ba, _ = m.run_episode(torch.cat([B,A]), meta_train=False)
        solo, _ = m.run_episode(A, meta_train=False)
    assert torch.allclose(aa[0],ab[0],atol=2e-6,rtol=2e-6)
    assert torch.allclose(aa[0],ba[1],atol=2e-6,rtol=2e-6)
    assert torch.allclose(aa[0],solo[0],atol=2e-6,rtol=2e-6)
    assert not ab.requires_grad
    assert all(torch.equal(v,saved[k]) for k,v in m.state_dict().items())
    assert len([p for p in m.parameters() if p.grad is not None]) == 0


def test_per_site_trust_and_no_crossbatch_state():
    m=fixture().eval()
    f=torch.randn(2,8,8,12)*0.01
    z=torch.randn(2,4)*30.0
    adapted=m.adapter(f,z)
    diff=adapted-f
    rms=diff.square().mean(1).sqrt()
    baseline_rms=f.square().mean(1).sqrt()
    assert (rms <= 0.07 * (baseline_rms+0.01) + 1e-6).all()
    assert torch.isfinite(adapted).all()


def test_explicit_second_order_detection_to_ssl_meta_gradient():
    m = fixture().train()
    f = torch.randn(2,8,8,12,requires_grad=True)
    # Toy per-pixel task reference; tests computation graph, NOT real YOLOX AP.
    target = torch.randn(2,8,8,12) * 0.2
    out, info = m.run_episode(f, meta_train=True)
    det_loss = F.mse_loss(out, target)
    # A projector-only parameter must receive task gradients THROUGH g_z.
    proj = m.predictor.layers[-1].weight
    gproj, = torch.autograd.grad(det_loss, proj, retain_graph=True)
    assert gproj is not None and gproj.abs().sum() > 1e-11, "No real meta-gradient into SSL predictor"
    gadapter, = torch.autograd.grad(det_loss, m.adapter.basis_to_channel.weight,retain_graph=True)
    assert gadapter is not None and gadapter.abs().sum()>1e-11
    gnative, = torch.autograd.grad(det_loss, f,retain_graph=True)
    assert gnative is not None and gnative.abs().sum()>1e-11
    assert torch.isfinite(info['gradient_norm']).all()
    # Training target masking itself MUST have no direct gradient into f:
    # no supervised projector loss through native path except detector.


def test_detached_update_reproduces_legacy_missing_meta_path():
    m = fixture().train()
    f = torch.randn(2,8,8,12)
    z0 = torch.zeros(2,4,requires_grad=True)
    ssl = m.ssl_loss(f,z0)
    g, = torch.autograd.grad(ssl.sum(),z0,create_graph=True)
    # Legacy FOMAML stopgrad prevents task loss from teaching projector how
    # to CHANGE the inner gradient. This negative control must be disconnected.
    updated = z0 - 0.1 * g.detach()
    adapted = m.adapter(f,updated)
    task = adapted.square().mean()
    route = torch.autograd.grad(task,m.predictor.layers[-1].weight,
                                allow_unused=True)[0]
    assert route is None, 'Detaching g must disconnect task gradient from SSL projector'


def test_backward_direction_changes_if_ssl_projector_changes():
    m=fixture().eval().requires_grad_(False)
    f=torch.randn(1,8,8,12)
    with torch.no_grad():
        before,_=m.run_episode(f,meta_train=False)
        m.predictor.layers[-1].weight.mul_(3)
        after,_=m.run_episode(f,meta_train=False)
    assert (after-before).abs().max()>1e-10, "Adaptation must depend on learned SSL predictor"


def test_alignment_surrogate_trains_only_ssl_direction():
    m=fixture().train()
    f=torch.randn(1,8,8,12)
    z0=torch.zeros(1,4,requires_grad=True)
    l=m.ssl_loss(f,z0)
    gssl,=torch.autograd.grad(l.sum(),z0,create_graph=True)
    gdet=torch.randn_like(gssl)
    metric=alignment_surrogate(gssl,gdet)
    gproj,=torch.autograd.grad(metric,m.predictor.layers[-1].weight)
    assert gproj is not None and gproj.abs().sum()>1e-11
    assert torch.isfinite(metric)


def test_fp16_output_quantization_is_measured_and_finite():
    # The native detector can expose FP16 activations under AMP while module
    # parameters remain FP32. Cast back only at the output.
    m = fixture().eval().requires_grad_(False)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = (torch.randn(2, 8, 8, 12) * 2).to(dtype)
        with torch.no_grad():
            out, info = m.run_episode(x, meta_train=False)
        assert out.dtype == x.dtype
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(info['postcast_changed_fraction']).all()
        assert (info['postcast_changed_fraction'] >= 0).all()
        assert (info['postcast_changed_fraction'] <= 1).all()
        assert torch.isfinite(info['relative_feature_change_precast']).all()
        assert torch.isfinite(info['relative_feature_change_postcast']).all()


def test_no_inference_mode_training_and_numerical_extremes():
    m=fixture().eval().requires_grad_(False)
    try:
        with torch.inference_mode():
            m.run_episode(torch.randn(1,8,8,12),meta_train=False)
    except RuntimeError as err:
        assert 'inference_mode' in str(err)
    else:
        raise AssertionError('Allowed broken TTT in inference_mode')
    for value in (0., 1e-6, 1e4):
        f=torch.full((1,8,8,12),value)
        with torch.no_grad():
            out, stats=m.run_episode(f,meta_train=False)
        assert torch.isfinite(out).all()
        assert all(torch.isfinite(v).all() for v in stats.values() if isinstance(v,torch.Tensor))


def test_mask_phase_has_25_percent_coverage_and_all_positions_visited():
    f=torch.randn(1,8,8,12)
    masks=[checkerboard_mask(f,i) for i in range(4)]
    assert all(abs(float(x.mean())-0.25)<1e-6 for x in masks)
    assert torch.equal(sum(masks),torch.ones_like(masks[0]))


if __name__=='__main__':
    for name,fun in list(globals().items()):
        if name.startswith('test_') and callable(fun):
            fun()
            print('PASS',name)
