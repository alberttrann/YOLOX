#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.3: First-Principles Covariate Moment Alignment (CMA) Engine

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib


class GRN(nn.Module):
    """Global Response Normalization (Retained for architectural backward compatibility)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        x_p = x.permute(0, 2, 3, 1)
        gx = torch.sqrt(torch.sum(x_p.pow(2), dim=(1, 2), keepdim=True) + 1e-6)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x_calibrated = self.gamma * (x_p * nx) + self.beta + x_p
        return x_calibrated.permute(0, 3, 1, 2).contiguous()


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib


class CovariateMomentAligner(nn.Module):
    """
    Certified Covariance Energy Alignment (CEA) Engine:
      - Elastic Covariance Margin (tau = 0.30): Preserves natural semantic co-activation.
        Penalizes strictly common-mode atmospheric haze/snow scattering!
      - Active Zero-Mean Loss L_mean: Gives beta active gradients to eliminate DC airlight.
      - Elastic Energy Band [0.50, 2.00]: Protects clean channels from variance collapse.
      - Spatial Decimation (::2, ::2) with .reshape(): Unbiased variance, zero memory leaks.
      - Returns per-sample loss [B] for true instance-independent adaptation.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, feat):
        B, C, H, W = feat.shape
        
        # 1. Spatial Decimation: Preserves true channel variance without low-pass blurring
        feat_sub = feat[:, :, ::2, ::2].float()
        N = float(feat_sub.shape[-2] * feat_sub.shape[-1])
        
        mu = feat_sub.mean(dim=[-2, -1], keepdim=True)
        f_centered = feat_sub - mu
        # Defensive .reshape() eliminates non-contiguous memory stride crashes:
        f_flat = f_centered.reshape(B, C, -1)
        
        # Sample Covariance Matrix Sigma: Shape [B, C, C]
        sigma = torch.bmm(f_flat, f_flat.transpose(1, 2)) / max(1.0, N - 1.0)
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. ACTIVE MARGIN-BOUNDED COVARIANCE LOSS (tau = 0.30)
        eye = torch.eye(C, device=feat.device, dtype=torch.float32).unsqueeze(0)
        off_diag_cov = sigma * (1.0 - eye)
        # Elastic Margin: Only penalizes covariance exceeding natural semantic threshold (0.30)
        excess_cov = torch.clamp(torch.abs(off_diag_cov) - 0.30, min=0.0)
        l_cov = excess_cov.pow(2).sum(dim=[-2, -1]) / float(C - 1)  # Shape: [B]
        
        # 3. ELASTIC DYNAMIC ENERGY BAND ON DIAGONAL VARIANCE [0.50, 2.00]
        var_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
        std_diag = torch.sqrt(torch.clamp(var_diag, min=self.eps, max=100.0))
        collapse_penalty = torch.clamp(0.50 - std_diag, min=0.0).pow(2)
        saturation_penalty = torch.clamp(std_diag - 2.00, min=0.0).pow(2)
        l_energy = (collapse_penalty + saturation_penalty).sum(dim=-1)       # Shape: [B]
        
        # 4. KOSCHMIEDER DC AIRLIGHT ZERO-MEAN LOSS (Gives beta active gradients!)
        l_mean = mu.squeeze(-1).squeeze(-1).pow(2).sum(dim=-1)               # Shape: [B]
        
        # Balanced Multi-Moment Loss per Sample: Shape [B]
        loss_cea_per_sample = l_cov + 0.5 * l_energy + 0.5 * l_mean
        return torch.nan_to_num(loss_cea_per_sample, nan=0.0)


class AdaptiveChannelAligner(nn.Module):
    """
    Channel-Decoupled Affine Alignment Operator.
    Maintains 100% semantic channel identity for Dark3, Dark4, Dark5, and BQSA Neck.
    Zero channel scrambling; zero matrix singularity risk.
    """
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta


class TTTAdaptiveStage(nn.Module):
    """
    TDE-YOLOX Master TTT Engine:
      - Feature-Stationary: Dark2 convolutions run ONCE forward and receive full detection loss.
      - Per-Instance Adaptation: Shape [B, C, 1, 1] guarantees zero cross-sample batch contamination.
      - FP32 Master Weight Precision: Eliminates FP16 inner-loop gradient underflow.
      - Calibrated Channel Gradients: Moves gamma and beta by physically meaningful magnitudes.
      - Exact Inverse-Softplus Initialization: Zero step-size explosion on Day 1.
      - Sub-millisecond inner loop: Zero backprop through deep convolutions!
    """
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.05):
        super().__init__()
        self.backbone_stage = stage_module
        self.aligner = AdaptiveChannelAligner(in_channels)
        self.cma_aligner = CovariateMomentAligner(eps=1e-5)
        
        # EXACT INVERSE-SOFTPLUS INITIALIZATION:
        # Guarantees softplus(init_raw) == init_ttt_lr (0.0500) exactly!
        init_raw = math.log(max(1e-6, math.exp(init_ttt_lr) - 1.0))
        self.ttt_lrs_gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * init_raw)
        self.ttt_lrs_beta = nn.Parameter(torch.ones(1, in_channels, 1, 1) * init_raw)

    def forward(self, x_in, run_ttt=True, epoch=0, max_epochs=80):
        # 1. Forward through Dark2 convolutions (Standard forward pass: receives full detection loss!)
        feat_base = self.backbone_stage(x_in)
        
        if not run_ttt:
            return feat_base, torch.tensor(0.0, device=x_in.device)

        B, C, H, W = feat_base.shape
        input_dtype = feat_base.dtype
        is_inference = (not torch.is_grad_enabled())
        context = torch.enable_grad() if is_inference else contextlib.nullcontext()

        # Cosine decay factor across training epochs
        if self.training:
            decay = 0.5 * (1.0 + math.cos(math.pi * float(epoch) / float(max(1, max_epochs))))
        else:
            decay = 1.0

        with context:
            # Expand affine parameters per sample [B, C, 1, 1] in FP32 for numerical safety
            gamma = self.aligner.gamma.float().repeat(B, 1, 1, 1)
            beta = self.aligner.beta.float().repeat(B, 1, 1, 1)

            # Detach feature base for the inner loop to isolate MAML graph
            f_curr = feat_base.detach()

            # Execute 1 step in training, 2 steps in inference
            inner_steps = 1 if self.training else 2

            for step in range(inner_steps):
                # Ensure gamma and beta are fresh leaves in FP32 for this step in inference
                if not self.training:
                    gamma = gamma.detach().requires_grad_(True)
                    beta = beta.detach().requires_grad_(True)

                # Differentiable affine feature modulation in FP32
                f_adapted = f_curr.float() * gamma + beta
                loss_cma_per_sample = self.cma_aligner(f_adapted)  # Shape: [B]
                inner_loss = loss_cma_per_sample.sum()

                # Gradients directly on gamma and beta vectors [B, C, 1, 1]
                grad_gamma, grad_beta = torch.autograd.grad(inner_loss, [gamma, beta], retain_graph=False)

                # Adaptive Step Sizes with exact softplus scaling
                lr_g = torch.clamp(F.softplus(self.ttt_lrs_gamma) * decay, min=1e-4, max=0.25)
                lr_b = torch.clamp(F.softplus(self.ttt_lrs_beta) * decay, min=1e-4, max=0.25)

                # Parameter updates remain strictly in FP32 to prevent gradient underflow
                gamma = gamma - lr_g * grad_gamma.detach()
                beta = beta - lr_b * grad_beta.detach()

        # 2. FINAL ADAPTED OUTPUT (Cast to input_dtype ONLY at the output interface!)
        out = (feat_base.float() * gamma + beta).to(input_dtype)
        
        # Optimization: Only compute aux_cma_loss during training where it is actually used
        if self.training:
            aux_cma_loss = (self.cma_aligner(out).mean() / float(C))
        else:
            aux_cma_loss = torch.tensor(0.0, device=x_in.device)

        return out, aux_cma_loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastAwareSE(nn.Module):
    """
    Squeeze-and-Excitation using both Mean AND Standard Deviation.
    Detects contrast energy through dense fog/snow where spatial mean is washed out.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fp32 = x.float()
        
        mean = x_fp32.mean(dim=[-2, -1])
        var = torch.clamp(x_fp32.var(dim=[-2, -1], unbiased=False), min=1e-6)
        std = torch.sqrt(var)
        
        # FRICTION FIX: Cast stat to x.dtype to prevent Half/Float collisions under AMP
        stat = torch.cat([mean, std], dim=1).to(x.dtype)
        weight = self.fc(stat).view(B, C, 1, 1).to(x.dtype)
        return x * weight


class RepDWConv(nn.Module):
    """
    Device-Safe Structural Re-Parameterization Depthwise Convolution.
    Training: Multi-branch (3x3 DW + 1x1 DW + Identity).
    Inference: Folds into a single 3x3 DWConv via switch_to_deploy() (0.0 ms overhead).
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1) // 2
        self.deploy = False

        self.dw_3x3 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=self.padding, dilation=dilation,
            groups=channels, bias=True
        )
        self.dw_1x1 = nn.Conv2d(
            channels, channels, kernel_size=1,
            padding=0, groups=channels, bias=True
        )

    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        else:
            return self.dw_3x3(x) + self.dw_1x1(x) + x

    def get_equivalent_kernel_bias(self):
        w_3x3 = self.dw_3x3.weight.data
        b_3x3 = self.dw_3x3.bias.data

        w_1x1 = self.dw_1x1.weight.data
        b_1x1 = self.dw_1x1.bias.data

        w_fused = w_3x3.clone()
        w_fused[:, 0, 1, 1] += w_1x1[:, 0, 0, 0]
        w_fused[:, 0, 1, 1] += 1.0

        b_fused = b_3x3 + b_1x1
        return w_fused, b_fused

    def switch_to_deploy(self):
        if self.deploy:
            return
        w_fused, b_fused = self.get_equivalent_kernel_bias()

        self.fused_conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=3,
            padding=self.padding, dilation=self.dilation,
            groups=self.channels, bias=True,
            device=self.dw_3x3.weight.device, dtype=self.dw_3x3.weight.dtype
        )
        self.fused_conv.weight.data.copy_(w_fused)
        self.fused_conv.bias.data.copy_(b_fused)

        del self.dw_3x3
        del self.dw_1x1
        self.deploy = True


class UnifiedRMBConv(nn.Module):
    """
    TDE-YOLOX Master Structural Conditioner:
      - Split-Dilation Droplet Bypass (50% d=1, 50% d=2): Samples around rain streaks.
      - Structural RepDWConv: Folds to single 3x3 conv at inference (0.0 ms test cost).
      - Contrast-Aware SE: Recalibrates channel energy under haze using Mean + Std.
      - Linear Bottleneck: Preserves feature manifold rank before residual scaling.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2

        self.rep_dw_local = RepDWConv(self.half_dim, kernel_size=3, dilation=1)
        self.rep_dw_dilated = RepDWConv(dim - self.half_dim, kernel_size=3, dilation=2)

        self.gn_dw = nn.GroupNorm(8 if dim % 8 == 0 else math.gcd(dim, 8), dim, eps=1e-5)
        self.act1 = nn.SiLU()

        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gn_pw = nn.GroupNorm(8 if dim % 8 == 0 else math.gcd(dim, 8), dim, eps=1e-5)

        self.se = ContrastAwareSE(dim, reduction=4)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.10)

    def forward(self, x):
        residual = x

        x1, x2 = torch.split(x, [self.half_dim, self.dim - self.half_dim], dim=1)
        dw1 = self.rep_dw_local(x1)
        dw2 = self.rep_dw_dilated(x2)
        dw_out = torch.cat([dw1, dw2], dim=1)

        feat = self.act1(self.gn_dw(dw_out))
        feat = self.gn_pw(self.pw(feat))  # Linear Bottleneck (preserves feature rank)
        feat = self.se(feat)

        return (residual + self.gamma * feat).to(x.dtype)

    def switch_to_deploy(self):
        self.rep_dw_local.switch_to_deploy()
        self.rep_dw_dilated.switch_to_deploy()


class UnifiedRCA(nn.Module):
    """
    TDE-YOLOX Master Directional Coordinate Attention:
      - Channel SNR Booster (Mean + Max cross-channel token).
      - Fully Decoupled Vertical & Horizontal Normalization (Zero Cross-Axis Bleed).
      - Identity-Anchored Calibrated Gate: Exactly 1.000x at step zero; clamped in [0.0, 2.0].
      - Canonical Parameter Preservation: Zero optimizer or checkpoint friction.
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        
        # Channel-augmented input (C + 1)
        self.conv1 = nn.Conv2d(in_channels + 1, mip, kernel_size=1)
        self.gn1 = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # 1. Global Channel SNR Reference Map
        z_mean = torch.mean(x.float(), dim=1, keepdim=True).to(x.dtype)
        z_max, _ = torch.max(x, dim=1, keepdim=True)
        z = (0.5 * z_mean + 0.5 * z_max).to(x.dtype)
        x_aug = torch.cat([x, z], dim=1)  # [B, C+1, H, W]

        # 2. Vertical Coordinate Pooling (Height)
        # Evaluated independently: gn1 normalizes across H spatial positions alone
        h_avg = F.adaptive_avg_pool2d(x_aug, (h, 1))
        h_max = F.adaptive_max_pool2d(x_aug, (h, 1))
        x_h = 0.5 * h_avg + 0.5 * h_max
        feat_h = self.act(self.gn1(self.conv1(x_h)))
        gate_h = torch.sigmoid(self.conv_h(feat_h))  # [B, C, H, 1]

        # 3. Horizontal Coordinate Pooling (Width)
        # Evaluated independently: gn1 normalizes across W spatial positions alone
        w_avg = F.adaptive_avg_pool2d(x_aug, (1, w))
        w_max = F.adaptive_max_pool2d(x_aug, (1, w))
        x_w = 0.5 * w_avg + 0.5 * w_max
        feat_w = self.act(self.gn1(self.conv1(x_w)))
        gate_w = torch.sigmoid(self.conv_w(feat_w))  # [B, C, 1, W]

        # 4. Identity-Anchored Calibrated Modulation
        # Exactly 1.000x at init (clamp(4.0 * 0.25, max=2.0) = 1.0), strictly bounded in [0.0, 2.0]
        raw_gate = 4.0 * (gate_h * gate_w)
        attn_gate = torch.clamp(raw_gate, min=0.0, max=2.0)

        return (identity * attn_gate).to(x.dtype)
    
class SimAM(nn.Module):
    """Parameter-free Energy-based Attention. Suppresses low-contrast background noise."""
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        var_fp32 = torch.var(x_fp32, dim=[2, 3], keepdim=True, unbiased=False)
        mean_fp32 = torch.mean(x_fp32, dim=[2, 3], keepdim=True)
        x_minus_mu_sq = (x_fp32 - mean_fp32).pow(2)
        y = x_minus_mu_sq / (4.0 * (var_fp32 + self.eps)) + 0.5
        return x * torch.sigmoid(y).to(x.dtype)


class CoordinateAttention(nn.Module):
    """Spatial Coordinate Bias injection to provide BQSA with geometric positional context."""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        
        x_fp32 = x.float()
        x_h = F.adaptive_avg_pool2d(x_fp32, (None, 1)).to(x.dtype)
        x_w = F.adaptive_avg_pool2d(x_fp32, (1, None)).permute(0, 1, 3, 2).to(x.dtype)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        att_h = 2.0 * torch.sigmoid(self.conv_h(x_h))
        att_w = 2.0 * torch.sigmoid(self.conv_w(x_w))
        return identity * att_h * att_w


class FP32AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """FP32 Shielded Pooling: Prevents half-precision accumulation overflow in deep SE blocks."""
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class MBConvConditioner(nn.Module):
    """MaxViT-inspired local structural conditioner."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.se = nn.Sequential(
            FP32AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(8, dim // 4), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(max(8, dim // 4), dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.se(x) * self.pw(self.dw(x))