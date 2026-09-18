#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Hierarchical BQSA & Symmetrical Scale-AttnRes PAFPN

import torch
import torch.nn as nn
from .darknet import CSPDarknet
from .network_blocks import BaseConv
from .tribrid_neck import P5ExclusiveDenseAttention, C2f_BQSA_P4, C2f_BQSA_P3, ScaleAttnRes


class YOLOPAFPN(nn.Module):
    """
    TDE-YOLOX v3.1 Master Neck:
      - Top-Down Pass: BQSA (P5 Dense Exclusive -> P4 Gumbel Reindex -> P3 Block Reuse)
      - Bottom-Up Pass: Symmetrical Scale-AttnRes depth softmax attention across N3, N4, N5
      - Preserves hierarchical anchor access for the Engram Head
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
        
        # 1. Backbone with GroupNorm Focus & Functional TTT-Dark2
        self.backbone = CSPDarknet(
            depth, width, out_features=in_features,
            depthwise=depthwise, act=act,
            ttt_lr=ttt_lr, ttt_noise_std=ttt_noise_std
        )
        
        c3, c4, c5 = [int(x * width) for x in in_channels]  # 128, 256, 512 for width=0.5
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Top-Down BQSA Components
        self.lateral_c5 = BaseConv(c5, c4, 1, 1, act=act)
        self.p5_dense_attn = P5ExclusiveDenseAttention(c5)
        self.p4_bqsa = C2f_BQSA_P4(c4 + c4, c4, n=round(3 * depth))
        self.reduce_p4 = BaseConv(c4, c3, 1, 1, act=act)
        self.p3_bqsa = C2f_BQSA_P3(c3 + c3, c3, n=round(3 * depth))
        
        # Bottom-Up Downsamplers
        self.down_n3 = BaseConv(c3, c4, 3, 2, act=act)
        self.down_n4 = BaseConv(c4, c5, 3, 2, act=act)
        
        # Bottom-Up Scale-AttnRes Highways with Spatial Area Normalization Factors
        # N3 (80x80): spatial_ratio = 6400 / 400 = 16.0
        self.attnres_n3 = ScaleAttnRes(dim=c3, num_sources=2, spatial_ratio=16.0)
        # N4 (40x40): spatial_ratio = 1600 / 400 = 4.0
        self.attnres_n4 = ScaleAttnRes(dim=c4, num_sources=3, spatial_ratio=4.0)
        # N5 (20x20): spatial_ratio = 400 / 400 = 1.0
        self.attnres_n5 = ScaleAttnRes(dim=c5, num_sources=3, spatial_ratio=1.0)

    def forward(self, input, targets=None, ttt_prob=None):
        # 1. Forward through Backbone
        out_features, proj_loss, phase_loss = self.backbone(input, ttt_prob=ttt_prob)
        c3 = out_features["dark3"]  # [B, 128, 80, 80]
        c4 = out_features["dark4"]  # [B, 256, 40, 40]
        c5 = out_features["dark5"]  # [B, 512, 20, 20]
        
        # 2. TOP-DOWN REASONING PASS (BQSA Cascade)
        p5_feat, p5_saliency, h_s = self.p5_dense_attn(c5)
        
        p4_in = torch.cat([self.upsample(self.lateral_c5(p5_feat)), c4], dim=1)
        p4_out, p4_indices, indexer_loss = self.p4_bqsa(p4_in, p5_saliency, h_s, targets=targets)
        
        p3_in = torch.cat([self.upsample(self.reduce_p4(p4_out)), c3], dim=1)
        p3_out = self.p3_bqsa(p3_in, p4_indices)
        
        # 3. BOTTOM-UP SCALE-ATTNRES GRADIENT HIGHWAY (E4, R4)
        n3_out = self.attnres_n3([p3_out, c3])
        n4_out = self.attnres_n4([p4_out, self.down_n3(n3_out), c4])
        n5_out = self.attnres_n5([p5_feat, self.down_n4(n4_out), c5])
        
        # Hierarchical anchor tuple returned cleanly to head (R1)
        # Scale 0 (P3) queries P4_out; Scale 1 (P4) queries P5_feat; Scale 2 (P5) queries P5_feat
        return ((n3_out, n4_out, n5_out), (p3_out, p4_out, p5_feat)), proj_loss, phase_loss, indexer_loss