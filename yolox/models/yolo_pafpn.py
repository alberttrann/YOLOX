#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv
from .tribrid_neck import C2f_Tribrid
from .ttt_modules import mHC_AttnRes_SPI


class YOLOPAFPN(nn.Module):
    """
    TDE-YOLOX Multi-Scale Path Aggregation Network (PAFPN).
    
    Features:
      - Sub-Pixel Phase Injection (SPI) Highway: Routes folded stride-4 phase channels 
        directly into Level P3 via mHC-AttnRes convex router, capturing sub-16px targets 
        without adding an expensive P2 prediction head.
      - Asymmetric Scale Sparsity: P3 (s_tilde=0.05), P4 (s_tilde=0.10), P5 (s_tilde=1.0 dense).
      - Full C2f_Tribrid integration across all multi-scale pyramid nodes.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        ttt_lr=0.02,
        ttt_noise_std=0.08,
        ttt_mask_ratio=0.20,
    ):
        super().__init__()
        # Backbone is configured to extract "dark2" in addition to standard pyramid levels
        backbone_features = ("dark2",) + tuple(in_features)
        self.backbone = CSPDarknet(
            depth,
            width,
            out_features=backbone_features,
            depthwise=depthwise,
            act=act,
            ttt_lr=ttt_lr,
            ttt_noise_std=ttt_noise_std,
            ttt_mask_ratio=ttt_mask_ratio,
        )
        self.in_features = in_features
        self.in_channels = in_channels

        # Channel dimensions for YOLOX-S (width=0.5: dark2=64, P3=128, P4=256, P5=512)
        c_dark2 = int(in_channels[0] * width * 0.5)  # 64
        c_p3 = int(in_channels[0] * width)            # 128
        c_p4 = int(in_channels[1] * width)            # 256
        c_p5 = int(in_channels[2] * width)            # 512

        # 1. Sub-Pixel Phase Injection (SPI) Highway
        self.spi_highway = mHC_AttnRes_SPI(in_channels_dark2=c_dark2, out_channels_p3=c_p3)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Lateral Convolutions
        self.lateral_conv0 = BaseConv(c_p5, c_p4, 1, 1, act=act)
        self.reduce_conv1 = BaseConv(c_p4, c_p3, 1, 1, act=act)
        self.bu_conv2 = BaseConv(c_p3, c_p3, 3, 2, act=act)
        self.bu_conv1 = BaseConv(c_p4, c_p4, 3, 2, act=act)

        # 2. C2f_Tribrid Multi-Scale Fusion Blocks with Asymmetric Sparsity Allocation
        # Top-down fusion: C3_p4 (P4 Level: moderate sparsity floor = 0.10)
        self.C3_p4 = C2f_Tribrid(
            c_p4 * 2,
            c_p4,
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
            saliency_floor=0.10,
        )

        # Top-down fusion: C3_p3 (P3 Level: high sparsity floor = 0.05)
        self.C3_p3 = C2f_Tribrid(
            c_p3 * 2,
            c_p3,
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
            saliency_floor=0.05,
        )

        # Bottom-up fusion: C3_n3 (P4 Level: moderate sparsity floor = 0.10)
        self.C3_n3 = C2f_Tribrid(
            c_p3 * 2,
            c_p4,
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
            saliency_floor=0.10,
        )

        # Bottom-up fusion: C3_n4 (P5 Level: dense associative binding = 1.0)
        self.C3_n4 = C2f_Tribrid(
            c_p4 * 2,
            c_p5,
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
            saliency_floor=1.0,
        )

    def forward(self, input, ttt_prob=None):
        out_features = self.backbone(input, ttt_prob=ttt_prob)

        # Extract TTT-adapted Stage 1 features and standard backbone levels
        dark2_feat = out_features["dark2"]  # [B, 64, H/4, W/4]
        x2 = out_features["dark3"]          # [B, 128, H/8, W/8] (P3)
        x1 = out_features["dark4"]          # [B, 256, H/16, W/16] (P4, C2fA)
        x0 = out_features["dark5"]          # [B, 512, H/32, W/32] (P5, C2fA+SPP)

        # SPI Highway: Fold stride-4 phase channels and convex-route into P3
        # Guarantees sub-pixel centroid retention without extra anchors or NMS boxes
        x2_enriched = self.spi_highway(dark2_feat, x2)

        # Top-Down Pathway
        fpn_out0 = self.lateral_conv0(x0)              # 512 -> 256 (P5 -> P4)
        f_out0 = self.upsample(fpn_out0)               # [B, 256, H/16, W/16]
        f_out0 = torch.cat([f_out0, x1], dim=1)        # [B, 512, H/16, W/16]
        f_out0 = self.C3_p4(f_out0)                    # [B, 256, H/16, W/16]

        fpn_out1 = self.reduce_conv1(f_out0)           # 256 -> 128 (P4 -> P3)
        f_out1 = self.upsample(fpn_out1)               # [B, 128, H/8, W/8]
        f_out1 = torch.cat([f_out1, x2_enriched], dim=1) # [B, 256, H/8, W/8] (SPI-Enriched)
        pan_out2 = self.C3_p3(f_out1)                  # [B, 128, H/8, W/8] (N3)

        # Bottom-Up Pathway
        p_out1 = self.bu_conv2(pan_out2)               # 128 -> 128 (N3 -> N4)
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)  # [B, 256, H/16, W/16]
        pan_out1 = self.C3_n3(p_out1)                  # [B, 256, H/16, W/16] (N4)

        p_out0 = self.bu_conv1(pan_out1)               # 256 -> 256 (N4 -> N5)
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)  # [B, 512, H/32, W/32]
        pan_out0 = self.C3_n4(p_out0)                  # [B, 512, H/32, W/32] (N5)

        # Output tuple (N3, N4, N5) feeds detection head and auxiliary affordance decoder
        return (pan_out2, pan_out1, pan_out0)

    def switch_to_deploy(self):
        """Triggers RepDWConv kernel folding across all C2f_Tribrid neck blocks."""
        self.C3_p4.switch_to_deploy()
        self.C3_p3.switch_to_deploy()
        self.C3_n3.switch_to_deploy()
        self.C3_n4.switch_to_deploy()