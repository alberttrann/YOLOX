#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.7: Physical Dark Channel Turbidity Engine & Pixel-Wise Phase Anchoring

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, BaseConvGN, CSPLayer, Focus, DWConv, DWConvGN, SPPBottleneck, ResLayer
from .ttt_modules import TTTAdaptiveStage


class DarkChannelTurbiditySensor(nn.Module):
    """
    Physical Dual-Mode Adverse Condition Sensor:
      1. Atmospheric Extinction (Fog/Snow): Evaluated via Dark Channel on lower 65% ground ROI.
      2. Photon Starvation (Night/Tunnels): Evaluated via scene luminance over ground plane.
      - Clean daylight: atmo_severity = 0.0, night_severity = 0.0 => adverse = 0.00000!
      - Fog / Snow: atmo_severity in [0.65, 1.00] => adverse in [0.65, 1.00]!
      - Night: atmo_severity = 0.0, night_severity in [0.65, 1.00] => adverse in [0.65, 1.00]!
    """
    def __init__(self, patch_size=16, tau_clean=0.10, tau_fog=0.38, tau_night=0.18):
        super().__init__()
        self.patch_size = patch_size
        self.tau_clean = float(tau_clean)
        self.tau_fog = float(tau_fog)
        self.tau_night = float(tau_night)

    @torch.no_grad()
    def forward(self, x):
        max_val = x.max()
        scale = 255.0 if max_val > 2.0 else 1.0
        x_norm = x.float() / scale

        # Ground-Plane Cropping (Lower 65% removes sky and clouds)
        h_cutoff = int(x_norm.shape[-2] * 0.35)
        x_ground = x_norm[:, :, h_cutoff:, :]

        # 1. Atmospheric Turbidity (Koschmieder Law)
        min_rgb = torch.min(x_ground, dim=1, keepdim=True).values
        dark_patch = -F.max_pool2d(
            -min_rgb,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        mu_dark = dark_patch.mean(dim=[-2, -1])  # Shape: [B, 1]
        atmo_severity = torch.clamp(
            (mu_dark - self.tau_clean) / (self.tau_fog - self.tau_clean),
            min=0.0,
            max=1.0
        )

        # 2. Low-Light Photon Starvation (Night / Tunnels)
        # Rec. 601 luma: Y = 0.299R + 0.587G + 0.114B
        luma = (0.299 * x_ground[:, 0:1] + 0.587 * x_ground[:, 1:2] + 0.114 * x_ground[:, 2:3])
        mu_luma = luma.mean(dim=[-2, -1])  # Shape: [B, 1]
        night_severity = torch.clamp(
            (self.tau_night - mu_luma) / (self.tau_night - 0.05),
            min=0.0,
            max=1.0
        )

        # Unified Adverse Severity Metric
        adverse_severity = torch.max(atmo_severity, night_severity)
        return adverse_severity.to(x.dtype)


class Darknet(nn.Module):
    """Maintained for legacy YOLOFPN backward compatibility."""
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
    TDE-YOLOX v3.7 Master CSPDarknet Backbone:
      - Sensor: DarkChannelTurbiditySensor estimates physical airlight veil directly from X.
      - Stage 0: Focus stem with BaseConvGN cancels local additive DC offsets.
      - Stage 1: Functional TTT-Dark2 (TTTAdaptiveStage) with GroupNorm and CMA whitening.
      - Deep Stages (Dark3-Dark5): Preserves exact channel alignment with pretrained YOLOX weights.
      - Regularization: Pixel-wise spatial Fourier Phase loss across 32 channels in FP32.
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

        # 0. PHYSICAL OPTICAL TURBIDITY SENSOR
        self.atmo_sensor = DarkChannelTurbiditySensor(patch_size=16, tau_clean=0.10, tau_fog=0.38)

        # 1. THE PRETRAINED STEM (BaseConv with BatchNorm2d)
        # Preserves absolute scene luminance (crucial for traffic lights, signs, and night driving)
        # and matches COCO pre-trained weight distribution.
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 2. ADAPTIVE STAGE 1 (DARK2) VIA CMA
        dark2_base = nn.Sequential(
            ConvGN(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
                use_gn=True
            ),
        )
        self.dark2 = TTTAdaptiveStage(
            dark2_base,
            in_channels=base_channels * 2,
            init_ttt_lr=ttt_lr
        )

        # 3. DARK3 (DEEP STAGE - Stride 8)
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 4. DARK4 (DEEP STAGE - Stride 16)
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 5. DARK5 (DEEP STAGE + SPP - Stride 32)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def _fourier_amplitude_perturb(self, x):
        """
        Fourier Low-Frequency Amplitude Swapping with Gaussian Apodization Window.
        Synthesizes photometrically perturbed views while rigidly freezing geometric phase.
        Enforces physical [0.0, 255.0] radiance bounds to eliminate Gibbs oscillation spikes.
        """
        B, C, H, W = x.shape
        # Real 2D Fast Fourier Transform in FP32
        fft_x = torch.fft.rfft2(x.float(), norm="backward")
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # Roll batch by 1 to sample paired clean amplitude spectrum
        amp_perm = torch.roll(amp, shifts=1, dims=0) if B > 1 else amp
        
        # Construct smooth Gaussian low-frequency mask centered at DC (u=0, v=0)
        u = torch.arange(H, device=x.device)
        W_half = fft_x.shape[-1]
        v = torch.arange(W_half, device=x.device)
        
        u_norm = torch.min(u, H - u).float() / H
        v_norm = v.float() / (W_half * 2.0)
        dist_sq = u_norm.unsqueeze(1).pow(2) + v_norm.unsqueeze(0).pow(2)
        
        # Gaussian apodization window targeting lowest 5% frequencies
        mask_gauss = torch.exp(-dist_sq / (2.0 * (0.05 ** 2))).unsqueeze(0).unsqueeze(0)
        
        alpha = 0.5
        amp_jittered = ((1.0 - alpha) * amp + alpha * amp_perm) * mask_gauss + amp * (1.0 - mask_gauss)
        
        perturbed_fft = torch.polar(amp_jittered, phase)
        x_perturbed = torch.fft.irfft2(perturbed_fft, s=(H, W), norm="backward")
        
        # Enforce physical radiance bounds
        x_perturbed = torch.clamp(x_perturbed, min=0.0, max=255.0)
        return x_perturbed.to(x.dtype)

    def forward(self, x, ttt_prob=None, epoch=0, max_epochs=80):
        outputs = {}

        # 1. OPTICAL TURBIDITY SENSING (Evaluated on raw sensor radiance X)
        h_atmo = self.atmo_sensor(x)  # Shape: [B, 1], in [0.0, 1.0]
        
        # 2. ATOMIC SINGLE FOCUS STEM FORWARD
        x_clean_stem = self.stem(x)
        outputs["stem"] = x_clean_stem
        
        # 3. EVALUATE PIXEL-WISE SPATIAL FOURIER PHASE LOSS (Training Only)
        if self.training and x.shape[0] > 1:
            with torch.no_grad():
                x_perturbed = self._fourier_amplitude_perturb(x)
            stem_pert = self.stem(x_perturbed)

            # Strict FP32 calculation with epsilon guard (1e-4) to prevent half-precision division overflow
            clean_fp32 = x_clean_stem.detach().float()
            pert_fp32 = stem_pert.float()

            norm_clean = torch.norm(clean_fp32, p=2, dim=1, keepdim=True)
            norm_pert = torch.norm(pert_fp32, p=2, dim=1, keepdim=True)
            denom = torch.clamp(norm_clean * norm_pert, min=1e-4)

            sim_spatial = (clean_fp32 * pert_fp32).sum(dim=1, keepdim=True) / denom
            phase_loss = (1.0 - sim_spatial).mean().to(x.dtype)
        else:
            phase_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 4. ADAPTIVE STAGE 1 (DARK2) VIA STATIONARY MAML
        run_ttt = False
        if not self.training:
            if ttt_prob is None or ttt_prob > 0.0:
                run_ttt = True
        else:
            prob = ttt_prob if ttt_prob is not None else 0.0
            if prob > 0.0 and (prob >= 1.0 or torch.rand(1).item() < prob):
                run_ttt = True

        x, cma_loss = self.dark2(x_clean_stem, run_ttt=run_ttt, epoch=epoch, max_epochs=max_epochs)
        outputs["dark2"] = x
        
        # 5. DEEP CONVOLUTIONAL STAGES
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        
        selected_outs = {k: v for k, v in outputs.items() if k in self.out_features}
        # Returns physical h_atmo along with features and auxiliary losses
        return selected_outs, cma_loss, phase_loss, h_atmo
    
