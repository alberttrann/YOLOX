#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
from torch import nn

from .network_blocks import (
    BaseConv,
    BaseConvGN,
    CSPLayer,
    DWConv,
    DWConvGN,
    Focus,
    ResLayer,
    SPPBottleneck,
)
from .ttt_modules import TTTAdaptiveStage


class Darknet(nn.Module):
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
    TDE-YOLOX CSPDarknet Backbone:
      - Stage 0 (Focus): Standard BatchNorm2d (normalizes raw input pixels).
      - Stage 1 (dark2): GroupNorm-isolated functional meta-adaptive stage (TTTAdaptiveStage).
      - Stages 2-4 (dark3, dark4, dark5): Standard pre-trained BatchNorm2d CSP layers.
    """

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
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

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # 0. The Static Stem (Focus Layer)
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 1. Stage 1 (dark2): Stateless GroupNorm Base wrapped in TTTAdaptiveStage
        dark2_base = nn.Sequential(
            ConvGN(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
                use_gn=True,
            ),
        )

        self.dark2 = TTTAdaptiveStage(
            dark2_base,
            in_channels=base_channels * 2,
            init_ttt_lr=ttt_lr,
            noise_std=ttt_noise_std,
            mask_ratio=ttt_mask_ratio,
        )

        # 2. Stage 2 (dark3): Deep Backbone (Standard BatchNorm2d)
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

        # 3. Stage 3 (dark4): Deep Backbone (Standard BatchNorm2d)
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
                use_gn=False,
            ),
        )

        # 4. Stage 4 (dark5): Deep Backbone + SPP (Standard BatchNorm2d)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
                use_gn=False,
            ),
        )

    def forward(self, x, ttt_prob=None):
        outputs = {}

        # Step 0: Initial Focus Stem
        x = self.stem(x)
        outputs["stem"] = x

        # Step 1: Stochastic Meta-Learning Decision for Stage 1
        run_ttt = False
        if not self.training:
            if ttt_prob is None or ttt_prob > 0.0:
                run_ttt = True
        else:
            prob = ttt_prob if ttt_prob is not None else 0.0
            if prob > 0.0 and torch.rand(1).item() < prob:
                run_ttt = True

        x = self.dark2(x, run_ttt=run_ttt)
        outputs["dark2"] = x

        # Steps 2-4: Deep Backbone Forwarding
        x = self.dark3(x)
        outputs["dark3"] = x

        x = self.dark4(x)
        outputs["dark4"] = x

        x = self.dark5(x)
        outputs["dark5"] = x

        return {k: v for k, v in outputs.items() if k in self.out_features}