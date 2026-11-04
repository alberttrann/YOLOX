"""Exact identity SPI at init; lossless stride-4 folding, no weather-specific clipping."""
import torch
from torch import nn
from torch.nn import functional as F


class SPIResidual(nn.Module):
    def __init__(self, in_channels_dark2=64, out_channels_p3=128):
        super().__init__()
        channels = in_channels_dark2 * 4
        self.unshuffle = nn.PixelUnshuffle(2)
        self.local = nn.Conv2d(channels, channels, 3, padding=1,
                               groups=channels, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(channels, out_channels_p3, 1, bias=False),
            nn.GroupNorm(min(8, out_channels_p3), out_channels_p3),
            nn.SiLU(),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, dark2, p3):
        # PixelUnshuffle and residual addition already enforce exact geometry;
        # Python tensor-shape conditions are trace-specialized in legacy ONNX.
        contribution = self.project(self.local(self.unshuffle(dark2)))
        return p3 + torch.tanh(self.gate) * contribution
