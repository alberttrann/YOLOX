#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.3: Master BQSA Neck & Symmetrical Scale-AttnRes PAFPN

import math
import torch
import torch.nn as nn
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
    TDE-YOLOX v3.3 Master PAFPN:
      - Top-Down Pass: BQSA (P5 Dense Attention -> P4 HySparse2 Gumbel -> P3 Ambient BQSA)
      - Bottom-Up Pass: Symmetrical Scale-AttnRes depth-attention highways
      - Cleaned contract: Threads epoch & max_epochs down to CMA Backbone
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

        # FIX: Compute c3, c4, c5 FIRST before referencing them!
        c3, c4, c5 = [int(x * width) for x in in_channels]

        # Instantiate Training-Only Semantic Affordance Decoder
        self.aux_sem_head = WeakSemanticAffordanceHead(
            in_channels_n3=c3, in_channels_n4=c4, num_classes=9, hidden_dim=c3
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
        self.p4_bqsa = C2f_BQSA_P4(c4 + c4, c4, n=round(3 * depth), sparsity_ratio=0.25)
        self.reduce_p4 = BaseConv(c4, c3, 1, 1, act=act)
        self.p3_bqsa = C2f_BQSA_P3(c3 + c3, c3, n=round(3 * depth), sparsity_ratio=0.25)

        # Bottom-Up Downsamplers
        self.down_n3 = BaseConv(c3, c4, 3, 2, act=act)
        self.down_n4 = BaseConv(c4, c5, 3, 2, act=act)

        # ScaleAttnRes Highways with cleaned parameter contracts
        self.attnres_n3 = ScaleAttnRes(dim=c3, num_sources=2)
        self.attnres_n4 = ScaleAttnRes(dim=c4, num_sources=3)
        self.attnres_n5 = ScaleAttnRes(dim=c5, num_sources=3)

    def forward(self, input, targets=None, ttt_prob=None, epoch=0, max_epochs=80):
        # 1. Forward through Backbone with CMA Curriculum
        out_features, cma_loss, phase_loss = self.backbone(
            input, ttt_prob=ttt_prob, epoch=epoch, max_epochs=max_epochs
        )
        c3 = out_features["dark3"]
        c4 = out_features["dark4"]
        c5 = out_features["dark5"]

        # 2. TOP-DOWN REASONING PASS (BQSA Cascade)
        p5_feat, p5_saliency, h_s = self.p5_dense_attn(c5)

        _p5_h, _p5_w = p5_feat.shape[-2], p5_feat.shape[-1]
        _h_s_fallback = math.log(max(2, _p5_h * _p5_w))
        h_s = torch.nan_to_num(h_s, nan=_h_s_fallback, posinf=_h_s_fallback, neginf=0.0)

        p4_in = torch.cat([self.upsample(self.lateral_c5(p5_feat)), c4], dim=1)
        p4_out, p4_indices, indexer_loss = self.p4_bqsa(p4_in, p5_saliency, h_s, targets=targets)

        p3_in = torch.cat([self.upsample(self.reduce_p4(p4_out)), c3], dim=1)
        p3_out = self.p3_bqsa(p3_in, p4_indices)

        # 3. BOTTOM-UP SCALE-ATTNRES GRADIENT HIGHWAY
        n3_out = self.attnres_n3([p3_out, c3])
        n4_out = self.attnres_n4([p4_out, self.down_n3(n3_out), c4])
        n5_out = self.attnres_n5([p5_feat, self.down_n4(n4_out), c5])

        # 4. TRAINING-ONLY WEAK SEMANTIC AFFORDANCE DECODER (Zero test-time overhead)
        if self.training:
            sem_pred = self.aux_sem_head(n3_out, n4_out)
        else:
            sem_pred = None

        return ((n3_out, n4_out, n5_out), (p3_out, p4_out, p5_feat), h_s), cma_loss, phase_loss, indexer_loss, sem_pred