#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.8: 100% Pretrained PAFPN Sovereignty & Identity-Guarded TDE Residuals

import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer
from .tribrid_neck import (
    P5ExclusiveDenseAttention,
    BQSA_P4_Assist,
    BQSA_P3_Assist,
    ScaleAttnRes,
)
from .aux_semantic_head import WeakSemanticAffordanceHead


class YOLOPAFPN(nn.Module):
    """
    TDE-YOLOX v3.8 Master PAFPN:
      - 100% Pretrained COCO Feature Pipeline (C3_p4, C3_p3, C3_n3, C3_n4).
      - Corrected PAFPN lateral wiring ensures exact [256, 512] channel matches for bottom-up CSPLayers.
      - BQSA & ScaleAttnRes operate as pure identity-guarded residuals (0.0% noise on Day 1).
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

        c3, c4, c5 = [int(x * width) for x in in_channels]

        self.aux_sem_head = WeakSemanticAffordanceHead(
            in_channels_n3=c3, in_channels_n4=c4, in_channels_n5=c5, num_classes=9, hidden_dim=c3
        )

        self.backbone = CSPDarknet(
            depth, width, out_features=in_features,
            depthwise=depthwise, act=act,
            ttt_lr=ttt_lr, ttt_noise_std=ttt_noise_std
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1. TOP-DOWN PRETRAINED BLOCKS
        self.lateral_c5 = BaseConv(c5, c4, 1, 1, act=act)
        self.C3_p4 = CSPLayer(c4 + c4, c4, n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act)
        self.reduce_p4 = BaseConv(c4, c3, 1, 1, act=act)
        self.C3_p3 = CSPLayer(c3 + c3, c3, n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act)

        # 2. BOTTOM-UP PRETRAINED BLOCKS (Corrected Input Channels!)
        self.bu_conv2 = BaseConv(c3, c3, 3, 2, act=act)
        # c3 (down_n3) + c3 (p4_lat) = c3 + c3
        self.C3_n3 = CSPLayer(c3 + c3, c4, n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act)
        self.bu_conv1 = BaseConv(c4, c4, 3, 2, act=act)
        # c4 (down_n4) + c4 (p5_lat) = c4 + c4
        self.C3_n4 = CSPLayer(c4 + c4, c5, n=round(3 * depth), shortcut=False, depthwise=depthwise, act=act)

        # 3. TDE RESIDUAL ENHANCEMENTS
        self.p5_dense_attn = P5ExclusiveDenseAttention(c5)
        self.bqsa_p4 = BQSA_P4_Assist(c4, block_size=4, sparsity_ratio=0.35)
        self.bqsa_p3 = BQSA_P3_Assist(c3, block_size=8, sparsity_ratio=0.35)
        self.grc_n4 = ScaleAttnRes(dim=c4, num_sources=3)
        self.grc_n5 = ScaleAttnRes(dim=c5, num_sources=3)

    def forward(self, input, targets=None, ttt_prob=None, epoch=0, max_epochs=80):
        out_features, cma_loss, phase_loss, h_atmo = self.backbone(
            input, ttt_prob=ttt_prob, epoch=epoch, max_epochs=max_epochs
        )
        c3 = out_features["dark3"]
        c4 = out_features["dark4"]
        c5 = out_features["dark5"]

        # P5 Dense Attention Residual
        p5_feat, p5_saliency, h_ratio = self.p5_dense_attn(c5, h_atmo=h_atmo)

        # Top-Down P4
        p5_lat = self.lateral_c5(p5_feat)  # Dimension-reduced C5 -> C4
        p5_up = F.interpolate(p5_lat, size=c4.shape[-2:], mode="nearest")
        p4_base = self.C3_p4(torch.cat([p5_up, c4], dim=1))
        p4_delta, p4_indices, indexer_loss = self.bqsa_p4(p4_base, p5_saliency=p5_saliency, h_s=h_ratio, targets=targets)
        p4_out = p4_base + p4_delta

        # Top-Down P3
        p4_lat = self.reduce_p4(p4_out)  # Dimension-reduced C4 -> C3
        p4_up = F.interpolate(p4_lat, size=c3.shape[-2:], mode="nearest")
        p3_base = self.C3_p3(torch.cat([p4_up, c3], dim=1))
        p3_delta, _ = self.bqsa_p3(p3_base, inherited_topk_idx=p4_indices)
        p3_out = p3_base + p3_delta

        # Bottom-Up N3
        n3_out = p3_out

        # Bottom-Up N4 (Corrected Lateral Concat: p4_lat instead of p4_out!)
        n4_in = torch.cat([self.bu_conv2(n3_out), p4_lat], dim=1)
        n4_base = self.C3_n3(n4_in)
        n4_delta = self.grc_n4([c4, p4_out, n4_base])
        n4_out = n4_base + n4_delta

        # Bottom-Up N5 (Corrected Lateral Concat: p5_lat instead of p5_feat!)
        n5_in = torch.cat([self.bu_conv1(n4_out), p5_lat], dim=1)
        n5_base = self.C3_n4(n5_in)
        n5_delta = self.grc_n5([c5, p5_feat, n5_base])
        n5_out = n5_base + n5_delta

        if self.training:
            sem_pred = self.aux_sem_head(n3_out, n4_out, n5_out)
        else:
            sem_pred = None

        return ((n3_out, n4_out, n5_out), (p3_out, p4_out, p5_feat), h_ratio), cma_loss, phase_loss, indexer_loss, sem_pred
