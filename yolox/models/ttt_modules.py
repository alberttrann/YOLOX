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
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, feat):
        B, C, H, W = feat.shape
        feat_sub = feat[:, :, ::2, ::2].float()
        N = float(feat_sub.shape[-2] * feat_sub.shape[-1])

        mu = feat_sub.mean(dim=[-2, -1], keepdim=True)
        f_centered = feat_sub - mu
        f_flat = f_centered.reshape(B, C, -1)

        sigma = torch.bmm(f_flat, f_flat.transpose(1, 2)) / max(1.0, N - 1.0)
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. ACTIVE MARGIN-BOUNDED COVARIANCE LOSS (tau = 0.30)
        eye = torch.eye(C, device=feat.device, dtype=torch.float32).unsqueeze(0)
        off_diag_cov = sigma * (1.0 - eye)
        excess_cov = torch.clamp(torch.abs(off_diag_cov) - 0.30, min=0.0)
        l_cov = excess_cov.pow(2).sum(dim=[-2, -1]) / float(C - 1)

        # 3. ELASTIC DYNAMIC ENERGY BAND ON DIAGONAL VARIANCE [0.50, 2.00]
        var_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
        std_diag = torch.sqrt(torch.clamp(var_diag, min=self.eps, max=100.0))
        l_energy = (torch.clamp(0.50 - std_diag, min=0.0).pow(2) + 
                    torch.clamp(std_diag - 2.00, min=0.0).pow(2)).sum(dim=-1)

        # 4. KOSCHMIEDER DC AIRLIGHT SHIFT RELATIVE TO CLEAN SiLU MEAN (+0.40)
        # Prevents driving beta negative and destroying Dark3 activation energy
        l_mean = (mu.squeeze(-1).squeeze(-1) - 0.40).pow(2).sum(dim=-1)

        loss_cea = l_cov + 0.5 * l_energy + 0.5 * l_mean
        return torch.nan_to_num(loss_cea, nan=0.0)


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
    TDE-YOLOX v3.7 Master TTTAdaptiveStage with Progressive Modulation Warmup (Zone 5 Fix).
    Features:
      - Clean Training Warmup: Ramps modulation strength over first 5 epochs (ttt_ramp = min(1, e/5)).
      - Preserves pre-trained COCO Dark2 -> Dark3 feature stationarity on Day 1 (~97% identity).
      - Test-Time Invariance: Operates at 100% unconstrained adaptation during all evaluation passes.
      - Positivity Guard: Keeps gamma strictly clamped in [0.10, 3.00].
    """
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.05):
        super().__init__()
        self.backbone_stage = stage_module
        self.aligner = AdaptiveChannelAligner(in_channels)
        self.cma_aligner = CovariateMomentAligner(eps=1e-5)
        init_raw = math.log(max(1e-6, math.exp(init_ttt_lr) - 1.0))
        self.ttt_lrs_gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * init_raw)
        self.ttt_lrs_beta = nn.Parameter(torch.ones(1, in_channels, 1, 1) * init_raw)

    def forward(self, x_in, run_ttt=True, epoch=0, max_epochs=65):
        feat_base = self.backbone_stage(x_in)
        if not run_ttt:
            return feat_base, torch.tensor(0.0, device=x_in.device)

        B, C, H, W = feat_base.shape
        input_dtype = feat_base.dtype
        is_inference = (not torch.is_grad_enabled())
        context = torch.enable_grad() if is_inference else contextlib.nullcontext()

        # STATIONARY 1-STEP MAML INNER LOOP:
        with context:
            gamma = self.aligner.gamma.float().repeat(B, 1, 1, 1)
            beta = self.aligner.beta.float().repeat(B, 1, 1, 1)
            f_curr = feat_base.detach()

            if not self.training:
                gamma = gamma.detach().requires_grad_(True)
                beta = beta.detach().requires_grad_(True)

            f_adapted = f_curr.float() * gamma + beta
            loss_cma = self.cma_aligner(f_adapted).sum()
            grad_gamma, grad_beta = torch.autograd.grad(loss_cma, [gamma, beta], retain_graph=False)

            lr_g = torch.clamp(F.softplus(self.ttt_lrs_gamma), min=1e-4, max=0.25)
            lr_b = torch.clamp(F.softplus(self.ttt_lrs_beta), min=1e-4, max=0.25)

            gamma_adapted = gamma - lr_g * grad_gamma.detach()
            beta_adapted = beta - lr_b * grad_beta.detach()

            # Positivity Guard: Prevents feature sign inversion
            gamma_adapted = torch.clamp(gamma_adapted, min=0.10, max=3.00)

        # PROGRESSIVE MODULATION WARMUP (Criterion 5 / Zone 5 Fix):
        # On Day 1 clean training, ttt_ramp = 0.20 => 80% identity preservation.
        # At inference/testing, ttt_ramp === 1.00000 => 100% adaptation on adverse weather!
        if self.training:
            current_ep = float(epoch) if epoch > 0 else 1.0
            ttt_ramp = min(1.0, max(0.0, current_ep / 5.0))
        else:
            ttt_ramp = 1.0

        gamma_eff = 1.0 + ttt_ramp * (gamma_adapted - 1.0)
        beta_eff = ttt_ramp * beta_adapted

        out = (feat_base.float() * gamma_eff + beta_eff).to(input_dtype)
        aux_cma_loss = (self.cma_aligner(out).mean() / float(C)) if self.training else torch.tensor(0.0, device=x_in.device)
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
    """Decoupled Coordinate Attention with Dedicated Axis Normalization (v3.6)."""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        
        # Dedicated Vertical (H) Pipeline
        self.conv_h_in = nn.Conv2d(in_channels + 1, mip, kernel_size=1)
        self.gn_h = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)

        # Dedicated Horizontal (W) Pipeline
        self.conv_w_in = nn.Conv2d(in_channels + 1, mip, kernel_size=1)
        self.gn_w = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        z_mean = torch.mean(x.float(), dim=1, keepdim=True).to(x.dtype)
        z_max, _ = torch.max(x, dim=1, keepdim=True)
        z = (0.5 * z_mean + 0.5 * z_max).to(x.dtype)
        x_aug = torch.cat([x, z], dim=1)

        # Vertical Coordinate Pooling (Height)
        x_h = 0.5 * F.adaptive_avg_pool2d(x_aug, (h, 1)) + 0.5 * F.adaptive_max_pool2d(x_aug, (h, 1))
        feat_h = self.act(self.gn_h(self.conv_h_in(x_h)))
        gate_h = torch.sigmoid(self.conv_h(feat_h))

        # Horizontal Coordinate Pooling (Width)
        x_w = 0.5 * F.adaptive_avg_pool2d(x_aug, (1, w)) + 0.5 * F.adaptive_max_pool2d(x_aug, (1, w))
        feat_w = self.act(self.gn_w(self.conv_w_in(x_w)))
        gate_w = torch.sigmoid(self.conv_w(feat_w))

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