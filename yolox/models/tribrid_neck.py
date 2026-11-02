
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
tdevNext vNext.2 — FP16-safe Tribrid neck.

Preserves:
- Native pretrained YOLOX CSP path
- Dense C2f residual branch
- SafeSimAM
- UnifiedRCA
- UnifiedRMBConv
- ContinuousSaliencyAttention
- Learnable global and dense-local gates

Numerical correction:
- Entire global mixer runs in FP32 under AMP.
- Gated merge runs in FP32.
- Cast back to native feature dtype only after merging.
"""

import torch
import torch.nn as nn

from .network_blocks import CSPLayer, DenseC2fLocal
from .ttt_modules import (
    ContinuousSaliencyAttention,
    SafeSimAM,
    UnifiedRCA,
    UnifiedRMBConv,
)


class C2f_Tribrid(CSPLayer):
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
        super().__init__(
            c1,
            c2,
            n=n,
            shortcut=shortcut,
            expansion=e,
            depthwise=depthwise,
            act=act,
            use_gn=False,
        )

        c_hidden = int(c2 * e)

        # Preserve all four contextual components.
        self.global_mixer = nn.Sequential(
            SafeSimAM(eps=1e-4),
            UnifiedRCA(c_hidden, reduction=16),
            UnifiedRMBConv(c_hidden),
            ContinuousSaliencyAttention(
                c_hidden,
                num_heads=2,
                saliency_floor=saliency_floor,
            ),
        )

        # A zero gate preserves the native CSP output at initialization.
        # The global mixer itself must NOT be zero-output initialized.
        self.gate = nn.Parameter(torch.zeros(1))

        self.enable_global_context = True
        self.enable_dense_local = True

        # Preserve dense local C2f learning capacity.
        self.dense_local = DenseC2fLocal(
            c1,
            c2,
            n=n,
            shortcut=shortcut,
            depthwise=depthwise,
            act=act,
            e=e,
        )
        self.dense_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. Native pretrained CSP branches.
        native = self.m(self.conv1(x))
        bypass = self.conv2(x)

        # 2. Full global-context precision island.
        if self.enable_global_context:

            # Disable the surrounding AMP autocast only for this branch.
            # Standard training still uses mixed precision elsewhere.
            with torch.autocast(
                device_type=native.device.type,
                enabled=False,
            ):
                native_f32 = native.float()

                # SafeSimAM -> RCA -> RMBConv -> Linear Attention.
                # All intermediate convolutions and reductions
                # now execute in FP32.
                context_f32 = self.global_mixer(native_f32)

                # Gate calculation also stays in FP32.
                gate_f32 = (
                    0.20 * torch.tanh(self.gate.float())
                )

                # Critical: merge BEFORE casting to FP16.
                # Prevents zero-gate multiplication with FP16 Inf.
                fused_f32 = (
                    native_f32 + gate_f32 * context_f32
                )

            # Restore the original feature dtype at the boundary.
            native = fused_f32.to(dtype=native.dtype)

        # 3. Native CSP fusion.
        out = self.conv3(
            torch.cat((native, bypass), dim=1)
        )

        # 4. Preserve dense C2f residual learning.
        if self.enable_dense_local:
            out = (
                out
                + 0.10
                * torch.tanh(self.dense_gate)
                * self.dense_local(x)
            )

        return out

    def switch_to_deploy(self):
        for module in self.global_mixer:
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
