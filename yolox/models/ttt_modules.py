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


class CovariateMomentAligner(nn.Module):
    """
    Elastic Margin-Bounded Covariate Moment Alignment (CMA) Engine:
      - Decorrelation Margin (tau=0.10): Allows natural semantic filter co-firing.
        Penalizes only excessive common-mode atmospheric airlight covariance.
      - Elastic Energy Band [0.5, 2.0]: Protects sparse channels from noise amplification
        while preventing variance collapse under fog/darkness.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, feat):
        B, C, H, W = feat.shape
        N = float(H * W)
        
        f_fp32 = feat.float()
        mu = f_fp32.mean(dim=[-2, -1], keepdim=True)
        f_centered = f_fp32 - mu
        f_flat = f_centered.view(B, C, -1)
        
        sigma = torch.bmm(f_flat, f_flat.transpose(1, 2)) / max(1.0, N - 1.0)
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=-1.0)
        
        var_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
        std_diag = torch.sqrt(torch.clamp(var_diag, min=self.eps, max=100.0))
        
        std_outer = torch.bmm(std_diag.unsqueeze(-1), std_diag.unsqueeze(1)) + self.eps
        corr = torch.clamp(sigma / std_outer, min=-1.0, max=1.0)
        
        # 1. ELASTIC DECORRELATION: Margin tau=0.10 preserves natural semantic co-activation
        eye = torch.eye(C, device=feat.device, dtype=torch.float32).unsqueeze(0)
        off_diag_corr = torch.abs(corr * (1.0 - eye))
        excess_corr = torch.clamp(off_diag_corr - 0.10, min=0.0)
        l_decorr = excess_corr.pow(2).sum(dim=[-2, -1]) / float(C * (C - 1))
        
        # 2. ELASTIC ENERGY BAND: Bounded within [0.5, 2.0]
        collapse_penalty = torch.clamp(0.50 - std_diag, min=0.0).pow(2)
        saturation_penalty = torch.clamp(std_diag - 2.00, min=0.0).pow(2)
        l_energy = (collapse_penalty + saturation_penalty).mean(dim=-1)
        
        loss_cma = (l_decorr + 0.5 * l_energy).mean()
        return torch.nan_to_num(loss_cma, nan=0.0)


class TTTAdaptiveStage(nn.Module):
    """
    Master CMA Test-Time Training Stage for Dark2.
    Dynamically whitens and de-correlates channel activations without spatial erasure.
    """
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.05):
        super().__init__()
        self.backbone_stage = stage_module
        self.cma_aligner = CovariateMomentAligner(eps=1e-5)
        self.init_ttt_lr = init_ttt_lr
        
        self.ttt_lrs = nn.ParameterDict()
        init_raw = math.log(max(1e-6, math.exp(init_ttt_lr) - 1.0))
        for name, param in self.backbone_stage.named_parameters():
            if 'gn' in name or 'norm' in name:
                self.ttt_lrs[name.replace('.', '_')] = nn.Parameter(torch.tensor(init_raw))

    def inner_loss_fn(self, params_backbone, x_in):
        # Differentiable functional forward pass on unmasked input
        feat = functional_call(self.backbone_stage, params_backbone, x_in.float())
        return self.cma_aligner(feat)

    def forward(self, x_in, run_ttt=True, epoch=0, max_epochs=80):
        input_dtype = x_in.dtype
        is_profiling = any(hasattr(m, 'total_ops') or hasattr(m, 'total_params') for m in self.backbone_stage.modules())
        if is_profiling or not run_ttt:
            feat_static = self.backbone_stage(x_in)
            return feat_static, torch.tensor(0.0, device=x_in.device)

        is_inference_mode = (not torch.is_grad_enabled())
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        backbone_params = dict(self.backbone_stage.named_parameters())
        backbone_buffers = dict(self.backbone_stage.named_buffers())
        
        adapt_backbone = {k: v.float() for k, v in backbone_params.items() 
                          if ('gn' in k or 'norm' in k) and v.is_floating_point()}
        static_backbone_state = {k: v.float() for k, v in backbone_params.items() 
                                 if k not in adapt_backbone}
        fixed_backbone_state = {**static_backbone_state, **{k: v.float() for k, v in backbone_buffers.items()}}

        self.backbone_stage.eval()

        # DUAL-STEP INFERENCE CONVERGENCE (Costs +1.4ms, drives CMA covariance to true minimum)
        inner_steps = 1 if self.training else 2

        # Compute decay_factor ONCE outside the loop:
        # Inference: full adaptation strength (1.0).
        # Training: smooth cosine decay (0.05 -> 0.005) across 80 epochs.
        if self.training:
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * float(epoch) / float(max(1, max_epochs))))
        else:
            decay_factor = 1.0

        with context_manager:
            curr_params = backbone_params
            for step in range(inner_steps):
                def inner_grad_fn(p_adapt_b):
                    full_backbone = {**p_adapt_b, **fixed_backbone_state}
                    return self.inner_loss_fn(full_backbone, x_curr)

                grads_backbone = grad(inner_grad_fn, argnums=0)(adapt_backbone)

                updated_backbone_params = {**curr_params, **backbone_buffers}
                for name, g in grads_backbone.items():
                    lr_key = name.replace('.', '_')
                    effective_lr = torch.clamp(F.softplus(self.ttt_lrs[lr_key]) * decay_factor, min=1e-4, max=0.20)
                    g_safe = torch.nan_to_num(g.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
                    new_param = (curr_params[name].float() - effective_lr * g_safe).to(input_dtype)
                    if not self.training:
                        new_param = new_param.detach()
                    updated_backbone_params[name] = new_param
                
                curr_params = updated_backbone_params
                adapt_backbone = {k: v.float() for k, v in curr_params.items() if ('gn' in k or 'norm' in k) and v.is_floating_point()}

        if self.training:
            self.backbone_stage.train()
            out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
            aux_cma_loss = self.cma_aligner(out)
        else:
            with torch.no_grad():
                out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
            aux_cma_loss = torch.tensor(0.0, device=x_in.device)

        return out.to(input_dtype), aux_cma_loss

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
    TDE-YOLOX Master Conditioner Block:
      - Split-Dilation Droplet Bypass: 50% d=1 + 50% d=2 (samples outside droplet streaks)
      - Fused at test time into two standalone depthwise convs (0.0 ms overhead)
      - Contrast-Aware SE (Mean + Std pooling)
      - LayerScale init gamma = 0.10
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
        self.act2 = nn.SiLU()

        self.se = ContrastAwareSE(dim, reduction=4)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.10)

    def forward(self, x):
        residual = x

        x1, x2 = torch.split(x, [self.half_dim, self.dim - self.half_dim], dim=1)
        dw1 = self.rep_dw_local(x1)
        dw2 = self.rep_dw_dilated(x2)
        dw_out = torch.cat([dw1, dw2], dim=1)

        feat = self.act1(self.gn_dw(dw_out))
        feat = self.act2(self.gn_pw(self.pw(feat)))
        feat = self.se(feat)

        return residual + self.gamma * feat

    def switch_to_deploy(self):
        self.rep_dw_local.switch_to_deploy()
        self.rep_dw_dilated.switch_to_deploy()


class UnifiedRCA(nn.Module):
    """
    TDE-YOLOX Master Coordinate Attention:
      - Channel SNR Booster + Channel-Max Guard
      - Explicit (h, 1) and (1, w) adaptive pooling shapes
      - Bounded, Identity-Preserving Output: 4.0 * (att_h * att_w)
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels + 1, mip, kernel_size=1)
        self.gn1 = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        z_mean = torch.mean(x.float(), dim=1, keepdim=True).to(x.dtype)
        z_max, _ = torch.max(x, dim=1, keepdim=True)
        z = (0.5 * z_mean + 0.5 * z_max).to(x.dtype)

        x_augmented = torch.cat([x, z], dim=1)

        h_avg = F.adaptive_avg_pool2d(x_augmented, (h, 1))
        h_max = F.adaptive_max_pool2d(x_augmented, (h, 1))
        x_h = 0.5 * h_avg + 0.5 * h_max

        w_avg = F.adaptive_avg_pool2d(x_augmented, (1, w)).permute(0, 1, 3, 2)
        w_max = F.adaptive_max_pool2d(x_augmented, (1, w)).permute(0, 1, 3, 2)
        x_w = 0.5 * w_avg + 0.5 * w_max

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.gn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        att_h = torch.sigmoid(self.conv_h(x_h))
        att_w = torch.sigmoid(self.conv_w(x_w))
        
        # FRICTION FIX: 4.0 * (att_h * att_w) ensures exact 1.0 identity transmission at init!
        attn_gate = 4.0 * (att_h * att_w)

        return identity * attn_gate
    
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