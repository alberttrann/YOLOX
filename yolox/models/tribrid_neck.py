#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import (
    ContinuousSaliencyAttention,
    SafeSimAM,
    UnifiedRCA,
    UnifiedRMBConv,
)


class C2f_Tribrid(nn.Module):
    """
    TDE-YOLOX C2f_Tribrid Block:
      - Branch A: Dense local convolutional bottlenecks (preserves high-frequency spatial gradients).
      - Branch B: Multi-Scale Context Engine:
          SafeSimAM (local 17x17 moving-window normalization)
          -> UnifiedRCA (glare-clipped hybrid 1D-2D coordinate attention)
          -> UnifiedRMBConv (receptive-adaptive split-dilation + RepDWConv + ContrastAwareSE)
          -> ContinuousSaliencyAttention (high-rank linear attention with macro presence envelope).
      - Bounded Residual Merge: Y = Branch A + 0.20 * tanh(gate) * Branch B.
      - Exact Identity Initialization: gate is initialized to 0.0 at Step 0, computing
        the exact mathematical forward pass of standard pre-trained CSPLayer.
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        depthwise=False,
        act="silu",
        e=0.5,
        saliency_floor=0.15,
    ):
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

        # Branch B: Context & Structural Conditioning Engine (internal channel width = self.c)
        self.global_mixer = nn.Sequential(
            SafeSimAM(eps=1e-4),
            UnifiedRCA(self.c, reduction=16),
            UnifiedRMBConv(self.c),
            ContinuousSaliencyAttention(self.c, num_heads=2, saliency_floor=saliency_floor),
        )

        # Step-0 Identity Initialization: gate = 0.0 -> tanh(0) = 0.0 -> 100% pre-trained CSPLayer pass
        # The 0.20 scalar bound prevents unregularized gate drift from blowing up in late epochs
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. Channel Split via 1x1 Projection
        y = list(self.cv1(x).chunk(2, dim=1))

        # 2. Dense Local Path (Branch A)
        y.extend(m(y[-1]) for m in self.m)

        # 3. Continuous Global Context Path (Branch B)
        global_context = self.global_mixer(y[-1])
        # Defensive immunity: prevents 0.0 * NaN from turning y[-1] into NaN
        global_context = torch.nan_to_num(global_context, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Bounded Gated Residual Addition
        y[-1] = y[-1] + 0.20 * torch.tanh(self.gate) * global_context

        # 5. Cross-Stage Feature Aggregation
        return self.cv2(torch.cat(y, dim=1))

    def switch_to_deploy(self):
        """Recursively folds RepDWConv inside UnifiedRMBConv into standard 3x3 convolutions."""
        for m in self.global_mixer:
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()