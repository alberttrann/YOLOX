#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
from torch import nn

from .network_blocks import (
    BaseConv,
    BaseConvGN,
    C2fA,
    CSPLayer,
    DWConv,
    DWConvGN,
    Focus,
    ResLayer,
    SPPBottleneck,
)
from .ttt_modules import TTTAdaptiveStage


class Darknet(nn.Module):
    # Number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2

        num_blocks = Darknet.depth2blocks[depth]
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    """
    TDE-YOLOX Multi-Scale Backbone:
      - Stage 0 (Focus): Standard BatchNorm2d (normalizes raw input pixel distribution).
      - Stage 1 (dark2, Stride 4): Stateless GroupNorm + Functional TTT meta-learning stage.
      - Stage 2 (dark3, Stride 8): Standard CSP block.
      - Stage 3 (dark4, Stride 16): C2fA module (EMA cross-spatial attention, YOLO-UTD [13]).
      - Stage 4 (dark5, Stride 32): C2fA module + SPPBottleneck.
    """

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark2", "dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        ttt_lr=0.02,
        ttt_noise_std=0.08,
        ttt_mask_ratio=0.20,
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        ConvGN = DWConvGN if depthwise else BaseConvGN

        base_channels = int(wid_mul * 64)  # 32 for YOLOX-S
        base_depth = max(round(dep_mul * 3), 1)  # 1 for YOLOX-S

        # 0. The Static Stem (Focus Layer)
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 1. Stage 1 (dark2, Stride 4): Stateless GroupNorm Base wrapped in TTTAdaptiveStage
        dark2_base = nn.Sequential(
            ConvGN(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
                use_gn=True,  # GroupNorm enabled across all dark2 bottlenecks
            ),
        )

        self.dark2 = TTTAdaptiveStage(
            dark2_base,
            in_channels=base_channels * 2,
            init_ttt_lr=ttt_lr,
            noise_std=ttt_noise_std,
            mask_ratio=ttt_mask_ratio,
        )

        # 2. Stage 2 (dark3, Stride 8): Standard BatchNorm2d CSP Layer
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
                use_gn=False,
            ),
        )

        # 3. Stage 3 (dark4, Stride 16): C2fA Block (YOLO-UTD EMA Attention [13])
        # Preserves spatial centroid representation for small targets prior to Neck aggregation
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            C2fA(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                shortcut=True,
                depthwise=depthwise,
                act=act,
            ),
        )

        # 4. Stage 4 (dark5, Stride 32): SPPBottleneck + C2fA Block
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            C2fA(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x, ttt_prob=None):
        outputs = {}

        # Step 0: Initial Focus Stem
        x = self.stem(x)
        outputs["stem"] = x

        # Step 1: Stochastic Meta-Learning Gating for Stage 1
        run_ttt = False
        if not self.training:
            if ttt_prob is None or ttt_prob > 0.0:
                run_ttt = True
        else:
            prob = ttt_prob if ttt_prob is not None else 0.0
            if prob > 0.0 and torch.rand(1).item() < prob:
                run_ttt = True

        x = self.dark2(x, run_ttt=run_ttt)
        outputs["dark2"] = x  # Expose dark2 for the SPI Highway

        # Steps 2-4: Deep Backbone Forwarding
        x = self.dark3(x)
        outputs["dark3"] = x

        x = self.dark4(x)
        outputs["dark4"] = x

        x = self.dark5(x)
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}