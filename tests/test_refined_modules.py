"""CPU smoke tests against standalone source snapshot; no full YOLOX installation needed.
Run from copied repo root: python tests/test_refined_modules.py
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "yolox/models/network_blocks.py").exists():
    # When running the patch bundle, use source_overlay for standalone checks.
    ROOT = ROOT / "source_overlay"
for name in ("yolox", "yolox.models"):
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
for name in ("network_blocks", "ttt_modules", "tribrid_neck", "darknet"):
    path = ROOT / "yolox" / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"yolox.models.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

blocks = sys.modules["yolox.models.network_blocks"]
tribrid = sys.modules["yolox.models.tribrid_neck"]
mods = sys.modules["yolox.models.ttt_modules"]
darknet = sys.modules["yolox.models.darknet"]
torch.manual_seed(1234)

def check_frozen_ema_eval_ttt():
    """Eval TTT must work on a completely frozen EMA copy."""

    native = torch.nn.Sequential(
        blocks.BaseConv(
            8,
            8,
            3,
            1,
        ),
        blocks.CSPLayer(
            8,
            8,
            n=1,
            use_gn=False,
        ),
    )

    adapter = mods.TTTAdaptiveStage(
        native,
        in_channels=8,
        mask_ratio=0.25,
        residual_adapter=True,
    )

    adapter.eval()
    adapter.set_epoch_state(
        1,
        80,
    )

    # Simulate ModelEMA.
    adapter.requires_grad_(False)

    before_scale = (
        adapter.residual_scale
        .detach()
        .clone()
    )

    before_bias = (
        adapter.residual_bias
        .detach()
        .clone()
    )

    x = torch.randn(
        2,
        8,
        8,
        8,
    )

    # Simulate COCOEvaluator.
    with torch.no_grad():
        y = adapter(
            x,
            run_ttt=True,
        )

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    # No inference graph escapes.
    assert not y.requires_grad

    # TTT must be episodic.
    # The persistent EMA copy must never be mutated.
    assert torch.equal(
        adapter.residual_scale,
        before_scale,
    )

    assert torch.equal(
        adapter.residual_bias,
        before_bias,
    )

def check_csp_compatibility():
    x = torch.randn(2, 32, 16, 16)
    for alternative in (tribrid.C2f_Tribrid, blocks.C2fA):
        native = blocks.CSPLayer(32, 32, n=1, shortcut=False)
        candidate = alternative(32, 32, n=1, shortcut=False)
        assert set(native.state_dict()).issubset(set(candidate.state_dict()))
        candidate.load_state_dict(native.state_dict(), strict=False)
        native.eval(); candidate.eval()
        with torch.no_grad():
            assert torch.equal(native(x), candidate(x))
        if alternative is tribrid.C2f_Tribrid:
            candidate.train()
            candidate(x).square().mean().backward()
            assert torch.isfinite(candidate.gate.grad).all()
            assert candidate.gate.grad.abs().item() > 0


def check_dark2_compatibility():
    native = torch.nn.Sequential(
        blocks.BaseConv(32, 64, 3, 2), blocks.CSPLayer(64, 64, n=1, use_gn=False)
    )
    stage = darknet.CSPDarknet(
        0.33, 0.5, pretrained_compatible_dark2=True
    ).dark2
    stage.backbone_stage.load_state_dict(native.state_dict())
    native.eval(); stage.eval()
    x = torch.randn(1, 32, 20, 28)
    with torch.no_grad():
        assert torch.equal(native(x), stage(x, run_ttt=False))


def check_ttt_mask_token_grad():
    stage = torch.nn.Sequential(blocks.BaseConvGN(8, 8, 3, 1, target_groups=8))
    ttt = mods.TTTAdaptiveStage(stage, in_channels=8, mask_ratio=0.25)
    ttt.train(); ttt.set_epoch_state(4, 80)
    out = ttt(torch.randn(1, 8, 8, 8, requires_grad=True), run_ttt=True)
    (out.square().mean() + ttt.current_rec_loss).backward()
    grad = ttt.projector.mask_token.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0


if __name__ == "__main__":
    check_csp_compatibility()
    check_dark2_compatibility()
    check_ttt_mask_token_grad()
    print("PASS: CSP compatibility, gate gradient, dark2 compatibility, TTT mask token")


def check_residual_ttt_bn_and_gradients():
    native = torch.nn.Sequential(
        blocks.BaseConv(8, 8, 3, 1), blocks.CSPLayer(8, 8, n=1, use_gn=False)
    )
    adapter = mods.TTTAdaptiveStage(
        native, in_channels=8, mask_ratio=0.25, residual_adapter=True
    )
    adapter.train()
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    out = adapter(x, run_ttt=True)
    assert out.shape == x.shape and torch.isfinite(out).all()
    (out.square().mean() + adapter.current_rec_loss).backward()
    for name, param in (
        ('backbone conv', native[0].conv.weight),
        ('residual scale', adapter.residual_scale),
        ('residual bias', adapter.residual_bias),
        ('projector mask', adapter.projector.mask_token),
    ):
        assert param.grad is not None, f'{name} disconnected'
        assert torch.isfinite(param.grad).all(), f'{name} non-finite'
    assert native[0].conv.weight.grad.abs().sum().item() > 0
    assert adapter.residual_scale.grad.abs().sum().item() > 0
    assert adapter.projector.mask_token.grad.abs().sum().item() > 0
    adapter.eval()
    with torch.no_grad():
        eval_out = adapter(x.detach(), run_ttt=True)
    assert torch.isfinite(eval_out).all()


def check_bn_forward_exactly_once():
    model = darknet.CSPDarknet(
        0.33, 0.5, ttt_residual_adapter=True, pretrained_compatible_dark2=True
    )
    model.train()
    count = [0]
    hook = model.dark2.backbone_stage.register_forward_hook(
        lambda _m, _i, _o: count.__setitem__(0, count[0] + 1)
    )
    x = torch.randn(4, 3, 64, 64)
    res = model(x, ttt_prob=0.5)
    hook.remove()
    assert count[0] == 1, f'dark2 BN executed {count[0]} times'
    assert all(torch.isfinite(y).all() for y in res.values())


def check_memory_gradient_and_oiou():
    bank = mods.EngramMemoryBank(9, 128, 4)
    a = torch.randn(2, 16, 128, requires_grad=True)
    retrieved, logits = bank(a)
    (retrieved.square().mean() + logits.square().mean()).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert bank.prototypes.grad is not None and torch.isfinite(bank.prototypes.grad).all()
    import importlib.util
    spec = importlib.util.spec_from_file_location('yolox.models.losses', ROOT / 'yolox/models/losses.py')
    losses = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses)
    pred = torch.tensor([[120., 140., 40., 25.], [200., 300., 15., 10.], [2., 2., 1., 1.]], requires_grad=True)
    target = torch.tensor([[120., 140., 40., 25.], [200., 330., 15., 10.], [3., 2., 1., 1.]])
    losses.IOUloss(loss_type='oiou')(pred, target).mean().backward()
    assert torch.isfinite(pred.grad).all(), pred.grad


if __name__ == '__main__':
    check_residual_ttt_bn_and_gradients()
    check_bn_forward_exactly_once()
    check_memory_gradient_and_oiou()
    check_frozen_ema_eval_ttt()
    print('PASS: residual TTT, BN single pass, memory grads, OIoU gradients')
