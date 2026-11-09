"""Preserved vNext.2 CSP/Tribrid/SPI, with no inference-time gradient path.

All modifications are explicit module replacements at build time; no source-file
rewrites, no monkeypatches, no special weather/dataset branches.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .ttt_modules import mHC_AttnRes_SPI


class NativeDark2Stage(nn.Module):
    def __init__(self, stage):
        super().__init__()
        if (not any(isinstance(m, nn.BatchNorm2d) for m in stage.modules()) or
            any(isinstance(m, nn.GroupNorm) for m in stage.modules())):
            raise RuntimeError(
                "Expected BN-pretrained-compatible dark2.backbone_stage. Found "
                "a different dark2 normalization implementation. Refusing to "
                "discard pretrained spatial features or silently use GN."
            )
        self.backbone_stage=stage

    def forward(self, x, run_ttt=False):
        return self.backbone_stage(x)

    def set_epoch_state(self, epoch, max_epochs):
        pass


class NoClippingSPI(mHC_AttnRes_SPI):
    """Drop irreversible +/-0.15 feature clipping, preserve ALL SPI weights.

    Same module names and parameter dimensions as the installed SPI branch.
    Source state is loaded STRICTLY: unsupported local SPI layout fails loudly.
    """
    def forward(self, dark2_feat, p3_feat):
        if dark2_feat.shape[-2] != p3_feat.shape[-2] * 2 or \
           dark2_feat.shape[-1] != p3_feat.shape[-1] * 2:
            raise ValueError("SPI requires stride4->stride8 exact 2x spatial geometry")
        u=self.unshuffle(dark2_feat)
        s=self.coord_attn(self.anti_alias_dw(u))
        projected=self.projector(s)
        B,C,H,W=p3_feat.shape
        routing_logits=self.router(torch.cat((p3_feat,projected),dim=1)).view(B,2,C,H,W)
        weights=torch.softmax(routing_logits,dim=1)
        return weights[:,0]*p3_feat+weights[:,1]*projected


def prepare_backbone(neck, *, remove_spi_clipping=True):
    """Resolve and preserve vNext.2 BN dark2; patch no source files."""
    if not hasattr(neck, 'backbone') or not hasattr(neck.backbone, 'dark2'):
        raise RuntimeError('Expected YOLOPAFPN.backbone.dark2 stage')
    dark2=neck.backbone.dark2
    if not hasattr(dark2,'backbone_stage'):
        raise RuntimeError('Expected existing vNext2 dark2.backbone_stage; wrong model source')
    neck.backbone.dark2=NativeDark2Stage(dark2.backbone_stage)
    if remove_spi_clipping:
        old=neck.spi_highway
        if not isinstance(old,mHC_AttnRes_SPI):
            raise RuntimeError('SPI base type mismatch; source revision must be reviewed')
        if not all(hasattr(old,name) for name in ('unshuffle','anti_alias_dw','coord_attn','projector','router')):
            raise RuntimeError('SPI module layout is incompatible with no-clipping implementation')
        new=NoClippingSPI(in_channels_dark2=old.projector.conv.in_channels//4,
                          out_channels_p3=old.router[-1].out_channels//2)
        new.load_state_dict(old.state_dict(), strict=True)
        neck.spi_highway=new
    # The saliency_floor is a tensor amplitude floor inside a module, not an
    # architectural requirement. Remove hard-coded per-level intervention.
    for m in neck.modules():
        if hasattr(m,'saliency_floor'):
            m.saliency_floor=0.0
    return neck
