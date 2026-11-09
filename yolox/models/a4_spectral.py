"""A4 local fixed-basis spectral evidence with identity-preserving bypass.

No global FFT or test-time adaptation. Optional matched random orthogonal basis
permits a genuinely operator-matched research control (same convolution ops).
"""
import torch
import torch.nn.functional as F
from .fourier_invariance import LocalSpectralResidual


class A4SpectralResidual(LocalSpectralResidual):
    def __init__(self, channels, basis='dct'):
        super().__init__(channels, mode='spectral')
        if basis not in ('dct', 'orthogonal_control'):
            raise ValueError('Unknown frequency basis')
        self.basis = basis
        if basis == 'orthogonal_control':
            # Fixed, repeatable, orthonormal zero-mean filters. Same conv kernels,
            # multiplication counts, parameters, and receptive fields as DCT.
            generator = torch.Generator(device='cpu').manual_seed(20261106)
            raw = torch.randn(9, 9, generator=generator)
            raw[0, :] = 1
            q, _ = torch.linalg.qr(raw.T)
            rows = q.T[1:4]  # three non-DC orthogonal vectors
            bank = rows.reshape(3, 1, 3, 3).repeat(channels, 1, 1, 1)
            self.dct_kernels = bank.contiguous()
        self.band_logits = torch.nn.Parameter(torch.zeros(3))
        self.enabled = True
        self.telemetry = False
        self.last_probe = None

    def _responses(self, x):
        b, c, h, w = x.shape
        banks = F.conv2d(x, self.dct_kernels.to(dtype=x.dtype), padding=1, groups=c)
        banks = banks.reshape(b, c, 3, h, w)
        # Softmax*3 yields identity response weights at step 0; conditions are
        # inferred from task gradients rather than manually-coded weather labels.
        weighting = (3 * torch.softmax(self.band_logits, dim=0)).to(x.dtype)
        banks = banks * weighting.reshape(1, 1, 3, 1, 1)
        return torch.cat((x.unsqueeze(2), banks), dim=2).reshape(b, c * 4, h, w)

    def forward(self, x):
        if not self.enabled:
            return x
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected BCHW with {self.channels} channels')
        h = F.silu(self.mix(self._responses(x)))
        delta = self.project(h)
        eps = torch.finfo(x.dtype).eps
        xe = (x.float().square().mean((-2, -1), keepdim=True) + eps**2).sqrt()
        de = (delta.float().square().mean((-2, -1), keepdim=True) + eps**2).sqrt()
        r = torch.sigmoid(self.gate(x).float()) * delta.float() * (xe / (xe + de))
        if self.telemetry:
            self.last_probe = (x.detach(), r.detach())
        return x + r.to(dtype=x.dtype)
