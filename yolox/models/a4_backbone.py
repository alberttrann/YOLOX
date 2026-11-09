"""A4 bounded SPI route retains a guaranteed native P3 contribution."""
import torch
from torch import nn
from .fourier_backbone import NoClippingSPI, prepare_backbone


class A4IdentitySPI(NoClippingSPI):
    def __init__(self, in_channels_dark2, out_channels_p3):
        super().__init__(in_channels_dark2, out_channels_p3)
        self.native_skip_gain = nn.Parameter(torch.zeros(()))
        self.enabled = True
        self.telemetry = False
        self.last_probe = None

    def forward(self, dark2_feat, p3_feat):
        if not self.enabled:
            return p3_feat
        mixed = super().forward(dark2_feat, p3_feat)
        # Skip cannot be fully cancelled; max correction = 1/4 of the
        # original softmax route difference, irrespective of dataset or scale.
        correction = 0.25 * torch.tanh(self.native_skip_gain) * (mixed - p3_feat)
        if self.telemetry:
            self.last_probe = (p3_feat.detach(), correction.detach())
        return p3_feat + correction


def prepare_a4_neck(neck):
    neck = prepare_backbone(neck, remove_spi_clipping=True)
    old = neck.spi_highway
    if not isinstance(old, NoClippingSPI):
        raise RuntimeError('Unexpected SPI structure; refusing silent A4 migration')
    new = A4IdentitySPI(
        in_channels_dark2=old.projector.conv.in_channels // 4,
        out_channels_p3=old.router[-1].out_channels // 2)
    new.load_state_dict({**new.state_dict(), **old.state_dict()}, strict=True)
    neck.spi_highway = new
    return neck
