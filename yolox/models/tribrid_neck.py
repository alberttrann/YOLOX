#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import (
    ContinuousSaliencyAttention,
    CoordinateAttention,
    MBConvConditioner,
    SimAM,
)


class C2f_Tribrid(nn.Module):
    """
    LW-DETR C2f Block + Paradigm 1 Linear Associative Attention Neck Block.
    
    Topology:
      - Branch A: Dense local convolutional bottlenecks (preserves high-frequency spatial gradients).
      - Branch B: Continuous Global Context Mixer:
          SimAM -> CoordinateAttention -> MBConvConditioner -> ContinuousSaliencyAttention.
      - Gated Merge: Output = Branch A + tanh(gate) * Branch B (initialized to 0.01).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, depthwise=False, act="silu", e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = BaseConv(c1, c2, 1, 1, act=act)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, 1, act=act)

        # Branch A: Dense Local Bottlenecks
        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                act=act,
                use_gn=False,
            )
            for _ in range(n)
        )

        # Branch B: Continuous Global Context Mixer (Paradigm 1)
        self.global_mixer = nn.Sequential(
            SimAM(),
            CoordinateAttention(self.c),
            MBConvConditioner(self.c),
            ContinuousSaliencyAttention(self.c),
        )

        # Gated Skip Connection (Initialized to 0.05 for stable training warmup)
        self.gate = nn.Parameter(torch.ones(1) * 0.05)

    def forward(self, x):
        # 1. Channel Split via 1x1 Projection
        y = list(self.cv1(x).chunk(2, dim=1))
        # y[0] is clean dense bypass, y[1] is base for local bottlenecks

        # 2. Dense Local Path (Branch A)
        y.extend(m(y[-1]) for m in self.m)

        # 3. Continuous Global Context Path (Branch B)
        # Applied to the deepest local bottleneck representation
        global_context = self.global_mixer(y[-1])

        # 4. Bounded Gated Residual Addition
        y[-1] = y[-1] + torch.tanh(self.gate) * global_context

        # 5. Cross-Stage Feature Aggregation
        return self.cv2(torch.cat(y, dim=1))