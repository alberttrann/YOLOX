"""tdevNext vNext.3: export-oriented frequency-selective classification residual.

No test-time optimization; no FFT or phase reconstruction; all inference ops are
real-valued Conv2d/pooling/pointwise ops.  `spectral` and `spatial` have EXACTLY
matching trainable parameter shapes, enabling a fair learned-capacity control.

Note: local 3x3 DCT kernels provide frequency-selective *responses*. They do
not mathematically guarantee phase/weather/scale invariance, and validation
must decide whether frequency structure improves actual detection.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def _dct_basis(n: int, modes=((0, 1), (1, 0), (2, 2))):
    """Orthonormal separable DCT-II kernels (real, fixed, zero mean)."""
    yy = torch.arange(n, dtype=torch.float32) + 0.5
    xx = torch.arange(n, dtype=torch.float32) + 0.5
    basis = []
    for u, v in modes:
        if u >= n or v >= n or (u == 0 and v == 0):
            raise ValueError("invalid DCT mode")
        by = math.sqrt((1 if u == 0 else 2) / n) * torch.cos(math.pi * u * yy / n)
        bx = math.sqrt((1 if v == 0 else 2) / n) * torch.cos(math.pi * v * xx / n)
        basis.append(by[:, None] * bx[None, :])
    return torch.stack(basis)


class LocalSpectralResidual(nn.Module):
    """Classification-only, zero-initialized, bounded residual feature adapter.

    mode='spectral': DCT-derived local depthwise filters.
    mode='spatial': parameter-matched local pooling controls.
    No hard weather thresholds, fixed spectral intensity scaling, or learned
    source-domain statistics.  The spectral basis is a model capacity choice.
    """
    def __init__(self, channels: int, mode: str = "spectral"):
        super().__init__()
        if mode not in ("spectral", "spatial"):
            raise ValueError("mode must be spectral or spatial")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.mode = mode
        # Fixed DCT responses are not learned and aren't checkpoint parameters.
        basis = _dct_basis(3)
        # Grouped-convolution channel layout requires [C, 3, 1, 3, 3].
        kernels = basis.unsqueeze(0).expand(channels, -1, -1, -1).reshape(channels * 3, 1, 3, 3)
        self.register_buffer("dct_kernels", kernels.contiguous(), persistent=False)
        self.mix = nn.Conv2d(channels * 4, channels, 1, groups=channels, bias=True)
        self.project = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = nn.Conv2d(channels, channels, 1, groups=channels, bias=True)
        # Zero output at initialization => EXACT native classifier input at step 0.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _responses(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if self.mode == "spectral":
            # Static 3x3 real kernels: standard Conv op in ONNX/TensorRT.
            dct = F.conv2d(x, self.dct_kernels.to(dtype=x.dtype), padding=1, groups=c)
            r = dct.reshape(b, c, 3, h, w)
            return torch.cat((x.unsqueeze(2), r), dim=2).reshape(b, c * 4, h, w)
        r1 = F.avg_pool2d(x, 3, stride=1, padding=1)
        r2 = F.max_pool2d(x, 3, stride=1, padding=1)
        r3 = F.avg_pool2d(x, 5, stride=1, padding=2)
        return torch.stack((x, r1, r2, r3), dim=2).reshape(b, c * 4, h, w)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected BCHW with C={self.channels}, got {tuple(x.shape)}")
        h = F.silu(self.mix(self._responses(x)))
        delta = self.project(h)
        # Relative RMS trust region; no pixel-unit threshold or fixed correction
        # amplitude. Bounds the residual energy by the native channel energy.
        # Smoothed RMS is crucial: sqrt(mean(delta²)) at delta==0 has
        # undefined backward and creates NaN even with a zero-initialized
        # projector. The epsilon depends only on numerical dtype.
        eps = torch.finfo(x.dtype).eps
        energy_x = (x.float().square().mean((-2, -1), keepdim=True) + eps * eps).sqrt()
        energy_delta = (delta.float().square().mean((-2, -1), keepdim=True) + eps * eps).sqrt()
        attenuation = energy_x / (energy_x + energy_delta)
        residual = torch.sigmoid(self.gate(x)) * (delta.float() * attenuation)
        return x + residual.to(x.dtype)


class NativeDark2Stage(nn.Module):
    """Keep vNext2 state_dict path `dark2.backbone_stage.*` with zero adaptation.

    The BN-compatible pretrained convolutional stage is reused without copying
    or reinitializing its weights. TTT projectors and meta-LRs are unregistered.
    """
    def __init__(self, backbone_stage: nn.Module):
        super().__init__()
        self.backbone_stage = backbone_stage

    def forward(self, x, run_ttt=None):
        return self.backbone_stage(x)

    def set_epoch_state(self, epoch, max_epochs):
        pass
