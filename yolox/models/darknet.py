#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Fourier Phase Invariance & Pretrained Remapped Backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, BaseConvGN, CSPLayer, Focus, DWConv, DWConvGN, SPPBottleneck, ResLayer
from .ttt_modules import TTTAdaptiveStage


class Darknet(nn.Module):
    # Maintained for legacy YOLOFPN backward compatibility
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self, depth, in_channels=3, stem_out_channels=32, out_features=("dark3", "dark4", "dark5")):
        super().__init__()
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2
        num_blocks = Darknet.depth2blocks[depth]
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
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
        return nn.Sequential(
            BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
            BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            SPPBottleneck(in_channels=filters_list[1], out_channels=filters_list[0], activation="lrelu"),
            BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
        )

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
    TDE-YOLOX v3.1 CSPDarknet Backbone.
    Integrates:
      - Stage 0: Focus layer with GroupNorm (BaseConvGN)
      - Stage 1: Functional TTT-Dark2 (TTTAdaptiveStage) with GroupNorm
      - Fourier Phase-Amplitude Consistency Loss (L_phase) evaluated on Theta_0
      - Deep Stages (Dark3-Dark5): Preserved for COCO pretrained weights
    """
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        ttt_lr=0.05,
        ttt_noise_std=0.08
    ):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        ConvGN = DWConvGN if depthwise else BaseConvGN

        base_channels = int(wid_mul * 64)        # 32 for width=0.5
        base_depth = max(round(dep_mul * 3), 1)  # 1 for depth=0.33

        # 0. THE GROUPNORM STEM
        # Instance centering directly on pixels eliminates atmospheric DC bias
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 1. THE ADAPTIVE STAGE 1 (DARK2)
        # Built strictly with GroupNorm components
        dark2_base = nn.Sequential(
            ConvGN(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
                use_gn=True  # Propagates GroupNorm through all internal bottlenecks
            ),
        )
        
        self.dark2 = TTTAdaptiveStage(
            dark2_base,
            in_channels=base_channels * 2,
            init_ttt_lr=ttt_lr,
            noise_std=ttt_noise_std
        )

        # 2. DARK3 (DEEP STAGE - Stride 8)
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 3. DARK4 (DEEP STAGE - Stride 16)
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 4. DARK5 (DEEP STAGE + SPP - Stride 32)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def _fourier_amplitude_perturb(self, x):
        """
        Fourier Low-Frequency Amplitude Swapping with Gaussian Apodization Window.
        Generates photometrically perturbed views while rigidly freezing geometric phase.
        """
        B, C, H, W = x.shape
        # Compute 2D real fast Fourier transform
        fft_x = torch.fft.rfft2(x.float(), norm="backward")
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # Roll batch by 1 to sample paired clean amplitude spectrum
        amp_perm = torch.roll(amp, shifts=1, dims=0)
        
        # Construct smooth Gaussian low-frequency mask centered at DC (u=0, v=0)
        u = torch.arange(H, device=x.device)
        W_half = fft_x.shape[-1]
        v = torch.arange(W_half, device=x.device)
        
        # Normalized frequency coordinates
        u_norm = torch.min(u, H - u).float() / H
        v_norm = v.float() / (W_half * 2.0)
        dist_sq = u_norm.unsqueeze(1).pow(2) + v_norm.unsqueeze(0).pow(2)
        
        # sigma_freq = 0.05 targets lowest 5% frequencies
        mask_gauss = torch.exp(-dist_sq / (2.0 * (0.05 ** 2))).unsqueeze(0).unsqueeze(0)
        
        # Perturb low frequencies, preserve high-frequency amplitude and original phase
        alpha = 0.5
        amp_jittered = ((1.0 - alpha) * amp + alpha * amp_perm) * mask_gauss + amp * (1.0 - mask_gauss)
        
        # Invert back to pixel space
        perturbed_fft = torch.polar(amp_jittered, phase)
        x_perturbed = torch.fft.irfft2(perturbed_fft, s=(H, W), norm="backward")
        return x_perturbed.to(x.dtype)

    def forward(self, x, ttt_prob=None):
        outputs = {}
        
        # 1. EVALUATE FOURIER PHASE CONSISTENCY LOSS ON PRE-ADAPTATION PARAMS
        if self.training:
            with torch.no_grad():
                x_perturbed = self._fourier_amplitude_perturb(x)
            
            # Forward both views through Stem and Dark2 using un-adapted parameters
            stem_clean = self.stem(x)
            stem_pert = self.stem(x_perturbed)
            
            # Evaluate Dark2 with run_ttt=False to capture pre-adaptation features
            z_clean, _ = self.dark2(stem_clean, run_ttt=False)
            z_pert, _ = self.dark2(stem_pert, run_ttt=False)
            
            # Cosine distance over spatial and channel dimensions
            sim = F.cosine_similarity(z_clean.flatten(1), z_pert.flatten(1), dim=-1)
            phase_loss = (1.0 - sim).mean()
        else:
            phase_loss = torch.tensor(0.0, device=x.device)

        # 2. STANDARD ADAPTIVE FORWARD PASS
        x = self.stem(x)
        outputs["stem"] = x
        
        # Determine stochastic TTT execution
        run_ttt = False
        if not self.training:
            if ttt_prob is None or ttt_prob > 0.0:
                run_ttt = True
        else:
            prob = ttt_prob if ttt_prob is not None else 0.0
            if prob > 0.0 and torch.rand(1).item() < prob:
                run_ttt = True

        # Execute Stage 1 with functional adaptation
        x, proj_loss = self.dark2(x, run_ttt=run_ttt)
        outputs["dark2"] = x
        
        # 3. DEEP STAGES (Pretrained Conv Highways)
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        
        selected_outs = {k: v for k, v in outputs.items() if k in self.out_features}
        return selected_outs, proj_loss, phase_loss