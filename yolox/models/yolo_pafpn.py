#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.7: Master BQSA Neck with Physical Optical Turbidity Integration

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import BaseConv
from .tribrid_neck import (
    P5ExclusiveDenseAttention,
    C2f_BQSA_P4,
    C2f_BQSA_P3,
    ScaleAttnRes,
)
from .aux_semantic_head import WeakSemanticAffordanceHead


class YOLOPAFPN(nn.Module):
    """
    TDE-YOLOX v3.7 Master PAFPN:
      - Senses physical atmospheric turbidity directly at input via DarkChannelTurbiditySensor.
      - Top-Down Pass: BQSA (P5 Dense Attention -> P4 HySparse2 -> P3 Synchronized Reuse).
      - Bottom-Up Pass: Symmetrical Scale-AttnRes depth-attention highways.
    """
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        ttt_lr=0.05,
        ttt_noise_std=0.08
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels

        # Compute channel dimensions
        c3, c4, c5 = [int(x * width) for x in in_channels]

        # Training-Only Semantic Affordance Decoder
        self.aux_sem_head = WeakSemanticAffordanceHead(
            in_channels_n3=c3, in_channels_n4=c4, in_channels_n5=c5, num_classes=9, hidden_dim=c3
        )

        self.backbone = CSPDarknet(
            depth, width, out_features=in_features,
            depthwise=depthwise, act=act,
            ttt_lr=ttt_lr, ttt_noise_std=ttt_noise_std
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-Down BQSA Components
        self.lateral_c5 = BaseConv(c5, c4, 1, 1, act=act)
        self.p5_dense_attn = P5ExclusiveDenseAttention(c5)
        self.p4_bqsa = C2f_BQSA_P4(c4 + c4, c4, n=round(3 * depth), sparsity_ratio=0.35)
        self.reduce_p4 = BaseConv(c4, c3, 1, 1, act=act)
        self.p3_bqsa = C2f_BQSA_P3(c3 + c3, c3, n=round(3 * depth), sparsity_ratio=0.35)

        # Bottom-Up Downsamplers
        self.down_n3 = BaseConv(c3, c4, 3, 2, act=act)
        self.down_n4 = BaseConv(c4, c5, 3, 2, act=act)

        # ScaleAttnRes Highways
        self.attnres_n3 = ScaleAttnRes(dim=c3, num_sources=2)
        self.attnres_n4 = ScaleAttnRes(dim=c4, num_sources=3)
        self.attnres_n5 = ScaleAttnRes(dim=c5, num_sources=3)

    def forward(self, input, targets=None, ttt_prob=None, epoch=0, max_epochs=80):
        # 1. Forward through Backbone with Physical Dark Channel Sensing
        out_features, cma_loss, phase_loss, h_atmo = self.backbone(
            input, ttt_prob=ttt_prob, epoch=epoch, max_epochs=max_epochs
        )
        c3 = out_features["dark3"]
        c4 = out_features["dark4"]
        c5 = out_features["dark5"]

        # 2. TOP-DOWN REASONING PASS (BQSA Cascade anchored to h_atmo)
        p5_feat, p5_saliency, h_ratio = self.p5_dense_attn(c5, h_atmo=h_atmo)

        # Dynamic Skip-Matched Interpolation
        p5_up = F.interpolate(self.lateral_c5(p5_feat), size=c4.shape[-2:], mode="nearest")
        p4_in = torch.cat([p5_up, c4], dim=1)
        p4_out, p4_indices, indexer_loss = self.p4_bqsa(p4_in, p5_saliency, h_ratio, targets=targets)

        p4_up = F.interpolate(self.reduce_p4(p4_out), size=c3.shape[-2:], mode="nearest")
        p3_in = torch.cat([p4_up, c3], dim=1)
        p3_out = self.p3_bqsa(p3_in, p4_indices)

        # 3. BOTTOM-UP SCALE-ATTNRES GRADIENT HIGHWAYS
        n3_out = self.attnres_n3([p3_out, c3])
        n4_out = self.attnres_n4([p4_out, self.down_n3(n3_out), c4])
        n5_out = self.attnres_n5([p5_feat, self.down_n4(n4_out), c5])

        # 4. TRAINING-ONLY WEAK SEMANTIC AFFORDANCE DECODER (Zero test latency)
        if self.training:
            sem_pred = self.aux_sem_head(n3_out, n4_out, n5_out)
        else:
            sem_pred = None

        return ((n3_out, n4_out, n5_out), (p3_out, p4_out, p5_feat), h_ratio), cma_loss, phase_loss, indexer_loss, sem_pred