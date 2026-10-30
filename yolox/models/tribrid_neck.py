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
          SafeSimAM (zero-centered identity)
          -> UnifiedRCA (decoupled H/W dual Avg+Max pooling)
          -> UnifiedRMBConv (split-dilation d=1,2 + RepDWConv + ContrastAwareSE)
          -> ContinuousSaliencyAttention (Paradigm 1 linear attention).
      - Gated Residual Merge: Y = Branch A + tanh(gate) * Branch B (Zero-Conv initialized).
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
        saliency_floor=0.05,
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

        # Branch B: Context & Structural Conditioning Engine
        self.global_mixer = nn.Sequential(
            SafeSimAM(eps=1e-4),
            UnifiedRCA(self.c, reduction=16),
            UnifiedRMBConv(self.c),
            ContinuousSaliencyAttention(self.c, num_heads=4, saliency_floor=saliency_floor),
        )

        # Learnable Residual Gate (Initialized to 0.05; ContinuousSaliencyAttention.proj is zero-init)
        self.gate = nn.Parameter(torch.ones(1) * 0.05)

    def forward(self, x):
        # 1. Channel Split via 1x1 Projection
        y = list(self.cv1(x).chunk(2, dim=1))

        # 2. Dense Local Path (Branch A)
        y.extend(m(y[-1]) for m in self.m)

        # 3. Continuous Global Context Path (Branch B)
        # Applied to the deepest local bottleneck representation
        global_context = self.global_mixer(y[-1])

        # 4. Bounded Gated Residual Addition
        y[-1] = y[-1] + torch.tanh(self.gate) * global_context

        # 5. Cross-Stage Feature Aggregation
        return self.cv2(torch.cat(y, dim=1))

    def switch_to_deploy(self):
        """Recursively folds RepDWConv inside UnifiedRMBConv into standard 3x3 convolutions."""
        for m in self.global_mixer:
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()