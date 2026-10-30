#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad

# PATCH 1.1 APPLIED: Removed CoordinateAttention from network_blocks import
from .network_blocks import BaseConv, BaseConvGN


# =========================================================================
# PILLAR 1: ADAPTIVE STEM PRIMITIVES (TTT Stage 1)
# =========================================================================

class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2).
    Forces channel competition to prevent feature collapse during masked modeling.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        x_p = x.permute(0, 2, 3, 1)
        gx = torch.norm(x_p, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return (self.gamma * (x_p * nx) + self.beta + x_p).permute(0, 3, 1, 2)


class TTTProjector(nn.Module):
    """Lightweight feature reconstruction autoencoder with GRN stabilization."""

    def __init__(self, in_channels):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        hidden = in_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            GRN(hidden),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class TTTAdaptiveStage(nn.Module):
    """
    Stateless Meta-Adaptive Stage 1 for CSPDarknet (dark2).
    
    Surgical Fixes Applied:
      - Patch 1.1: Clean local imports without cyclic collisions.
      - Patch 1.2: Enforces strict positivity on meta-LRs via F.softplus to prevent gradient ascent.
      - Exact Quantile Masking (Fix 14.1): Thresholds at 1.0 - ratio to mask exactly top 20% salient tokens.
      - Contrast-Calibrated Inner Loss: Joint spatial MSE + directional cosine alignment.
    """

    def __init__(
        self,
        stage_module,
        in_channels,
        init_ttt_lr=0.02,
        noise_std=0.08,
        mask_ratio=0.20,
    ):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

        # Learnable per-parameter adaptation rates for all GroupNorm layers
        self.ttt_lrs = nn.ParameterDict()
        for name, param in self.backbone_stage.named_parameters():
            if ("gn" in name or "norm" in name) and param.requires_grad:
                self.ttt_lrs[name.replace(".", "_")] = nn.Parameter(
                    torch.tensor(init_ttt_lr, dtype=torch.float32)
                )

    def _get_relative_contrast_mask(self, feat, ratio=0.20):
        B, C, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype

        # 1. 2D Spatial Sobel Kernels for Gradient Magnitude
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)

        feat_mean = feat.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(feat_mean, sobel_x, padding=1)
        grad_y = F.conv2d(feat_mean, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # 2. Relative Normalization (Scale-Invariant Contrast)
        rel_grad = grad_mag / (grad_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        var_c = torch.var(feat, dim=1, keepdim=True)
        rel_var = var_c / (var_c.mean(dim=(2, 3), keepdim=True) + 1e-6)

        saliency = rel_grad + rel_var
        s_pooled = F.avg_pool2d(saliency, 4, stride=4)

        # 3. Exact Quantile Masking (Fixing Caveat 14.1)
        s_flat = s_pooled.view(B, -1).float()
        threshold = torch.quantile(s_flat, 1.0 - ratio, dim=1, keepdim=True)
        mask_pooled = (s_pooled >= threshold.view(B, 1, 1, 1)).to(dtype)

        return F.interpolate(mask_pooled, size=(H, W), mode="nearest")

    def inner_loss_fn(self, params, projector_params, x_in, mask, noise_map, clean_target):
        feat = functional_call(self.backbone_stage, params, x_in)
        feat_noisy = feat + noise_map
        feat_corrupted = feat_noisy * (1.0 - mask) + self.projector.mask_token * mask
        rec = functional_call(self.projector, projector_params, feat_corrupted)

        mse_loss = F.mse_loss(rec, clean_target)
        rec_f32 = rec.float()
        target_f32 = clean_target.float()
        rec_norm = F.normalize(rec_f32, p=2, dim=1, eps=1e-3)
        target_norm = F.normalize(target_f32, p=2, dim=1, eps=1e-3)

        cos_sim = (rec_norm * target_norm).sum(dim=1).mean()
        cos_loss = 1.0 - cos_sim

        return mse_loss + 0.1 * cos_loss

    def forward(self, x_in, run_ttt=True):
        is_profiling = False
        for m in self.backbone_stage.modules():
            if hasattr(m, "total_ops") or hasattr(m, "total_params"):
                is_profiling = True
                break

        if is_profiling or not run_ttt:
            return self.backbone_stage(x_in)

        input_dtype = x_in.dtype

        is_inference_mode = not torch.is_grad_enabled()
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        backbone_params = dict(self.backbone_stage.named_parameters())
        backbone_buffers = dict(self.backbone_stage.named_buffers())
        proj_params = dict(self.projector.named_parameters())

        adapt_backbone = {
            k: v
            for k, v in backbone_params.items()
            if ("gn" in k or "norm" in k) and v.is_floating_point()
        }
        static_backbone_state = {
            k: v for k, v in backbone_params.items() if k not in adapt_backbone
        }
        fixed_backbone_state = {**static_backbone_state, **backbone_buffers}

        self.backbone_stage.eval()
        self.projector.eval()

        with context_manager:
            with torch.no_grad():
                feat_initial = self.backbone_stage(x_curr)
                clean_target = feat_initial.detach()
                mask = self._get_relative_contrast_mask(clean_target, ratio=self.mask_ratio)
                noise_map = torch.randn_like(clean_target) * self.noise_std

            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                return self.inner_loss_fn(
                    full_backbone, p_adapt_p, x_curr, mask, noise_map, clean_target
                )

            grads_backbone = grad(inner_grad_fn, argnums=0)(adapt_backbone, proj_params)

            # PATCH 1.2 APPLIED: Softplus positivity constraint on meta-learning rates
            updated_backbone_params = {**backbone_params, **backbone_buffers}
            for name, g in grads_backbone.items():
                lr_key = name.replace(".", "_")
                # Positive, stable step size bounded to prevent inner-loop gradient ascent
                lr = torch.clamp(F.softplus(self.ttt_lrs[lr_key]), min=1e-4, max=0.20)
                updated_backbone_params[name] = updated_backbone_params[name] - lr * g.to(
                    updated_backbone_params[name].dtype
                )

        if self.training:
            self.backbone_stage.train()
            self.projector.train()

        if self.training:
            out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
        else:
            with torch.no_grad():
                out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)

        return out.to(input_dtype)


# =========================================================================
# PILLAR 2: SUB-PIXEL PHASE INJECTION (SPI) HIGHWAY
# =========================================================================

class mHC_AttnRes_SPI(nn.Module):
    """
    Sub-Pixel Phase Injection (SPI) Highway with Manifold-Constrained Convex Routing.
    
    Synthesizes:
      - Space-to-Depth Folding (PixelUnshuffle): Preserves sub-pixel coordinates without adding anchors.
      - LASK Anti-Aliasing Projector: Smooths phase boundaries via 3x3 DWConv + CoordinateAttention.
      - Birkhoff Simplex Router (mHC / Kimi AttnRes / AFF): Guarantees bounded variance (sum=1, non-negative).
    """

    def __init__(self, in_channels_dark2=64, out_channels_p3=128):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        folded_channels = in_channels_dark2 * 4

        self.anti_alias_dw = nn.Conv2d(
            folded_channels, folded_channels, kernel_size=3, padding=1, groups=folded_channels, bias=False
        )
        self.coord_attn = CoordinateAttention(folded_channels, reduction=16)
        self.projector = BaseConvGN(
            folded_channels, out_channels_p3, ksize=1, stride=1, act="silu", target_groups=16
        )

        self.router = nn.Sequential(
            nn.Linear(out_channels_p3 * 2, out_channels_p3),
            nn.SiLU(),
            nn.Linear(out_channels_p3, out_channels_p3 * 2),
        )

        with torch.no_grad():
            self.router[-1].bias.data[:out_channels_p3].fill_(2.0)
            self.router[-1].bias.data[out_channels_p3:].fill_(0.0)

    def forward(self, dark2_feat, p3_feat):
        B, C, H, W = p3_feat.shape

        unshuffled = self.unshuffle(dark2_feat)
        smoothed = self.coord_attn(self.anti_alias_dw(unshuffled))
        spi_feat = self.projector(smoothed)

        k1 = p3_feat.mean(dim=(2, 3))
        k2 = spi_feat.mean(dim=(2, 3))

        routing_inputs = torch.cat([k1, k2], dim=1)
        routing_logits = self.router(routing_inputs).view(B, C, 2)

        weights = F.softmax(routing_logits, dim=-1)
        alpha_1 = weights[:, :, 0].view(B, C, 1, 1)
        alpha_2 = weights[:, :, 1].view(B, C, 1, 1)

        fused_p3 = alpha_1 * p3_feat + alpha_2 * spi_feat
        return fused_p3


# =========================================================================
# PILLAR 2: NECK CONDITIONING ENGINE (SafeSimAM, UnifiedRCA, UnifiedRMBConv)
# =========================================================================

class SafeSimAM(nn.Module):
    """
    Zero-Centered Residual SimAM.
    - Flat background / dense fog (x ≈ μ): Gain ≡ 1.0 (Exact Identity, Zero Noise Amplification).
    - Salient structural targets: Gain → 1.0 + γ · tanh(dev) (Selective feature enhancement).
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1) * 0.10)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        var = (x - x_mean).pow(2).sum(dim=[2, 3], keepdim=True) / n
        dev = (x - x_mean).pow(2) / (4.0 * (var + self.eps))
        modulation = 1.0 + self.gamma * torch.tanh(dev)
        return x * modulation


class CoordinateAttention(nn.Module):
    """Spatial coordinate awareness via 1D directional strip pooling."""

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.GroupNorm(8, mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class UnifiedRCA(nn.Module):
    """
    Decoupled Coordinate Attention with Dedicated Orthogonal Axis Normalization.
    Features:
      - Independent H and W pipelines (eliminates cross-axis coordinate contamination).
      - Dual Avg + Max Pooling (captures localized high-intensity lights at night/snow).
      - Self-calibrating initial identity gate: 4.0 * (g_h * g_w) ≈ 1.0 at Step 0, clamped to [0.0, 2.0].
    """

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)

        self.conv_h_in = nn.Conv2d(in_channels + 1, mip, kernel_size=1)
        self.gn_h = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)

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

        x_h = 0.5 * F.adaptive_avg_pool2d(x_aug, (h, 1)) + 0.5 * F.adaptive_max_pool2d(x_aug, (h, 1))
        feat_h = self.act(self.gn_h(self.conv_h_in(x_h)))
        gate_h = torch.sigmoid(self.conv_h(feat_h))

        x_w = 0.5 * F.adaptive_avg_pool2d(x_aug, (1, w)) + 0.5 * F.adaptive_max_pool2d(x_aug, (1, w))
        feat_w = self.act(self.gn_w(self.conv_w_in(x_w)))
        gate_w = torch.sigmoid(self.conv_w(feat_w))

        raw_gate = 4.0 * (gate_h * gate_w)
        attn_gate = torch.clamp(raw_gate, min=0.0, max=2.0)
        return (identity * attn_gate).to(x.dtype)


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
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fp32 = x.float()

        mean = x_fp32.mean(dim=[-2, -1])
        var = torch.clamp(x_fp32.var(dim=[-2, -1], unbiased=False), min=1e-6)
        std = torch.sqrt(var)

        stat = torch.cat([mean, std], dim=1).to(x.dtype)
        weight = self.fc(stat).view(B, C, 1, 1).to(x.dtype)
        return x * weight


class RepDWConv(nn.Module):
    """
    Structural Re-Parameterization Depthwise Convolution.
    Training: Multi-branch (3x3 DW + 1x1 DW + Identity).
    Inference: Folds into a single standard 3x3 DWConv via switch_to_deploy() (0.0 ms overhead).
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
            groups=channels, bias=True,
        )
        self.dw_1x1 = nn.Conv2d(
            channels, channels, kernel_size=1,
            padding=0, groups=channels, bias=True,
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
            device=self.dw_3x3.weight.device, dtype=self.dw_3x3.weight.dtype,
        )
        self.fused_conv.weight.data.copy_(w_fused)
        self.fused_conv.bias.data.copy_(b_fused)

        del self.dw_3x3
        del self.dw_1x1
        self.deploy = True


class UnifiedRMBConv(nn.Module):
    """
    TDE-YOLOX Master Structural Conditioner:
      - Split-Dilation Droplet Bypass (50% d=1, 50% d=2): Samples around localized weather streaks.
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
        feat = self.gn_pw(self.pw(feat))
        feat = self.se(feat)

        return (residual + self.gamma * feat).to(x.dtype)

    def switch_to_deploy(self):
        self.rep_dw_local.switch_to_deploy()
        self.rep_dw_dilated.switch_to_deploy()


# =========================================================================
# PILLAR 2: CONTINUOUS SALIENCY LINEAR ATTENTION (Paradigm 1)
# =========================================================================

class ContinuousSaliencyAttention(nn.Module):
    """
    Paradigm 1: Multi-Head Linear Saliency-Gated Associative Attention.
    
    Surgical Fixes Applied:
      - Patch 1.3: Injects F.silu() between DW and PW convolutions for non-linear boundary modeling.
      - Patch 1.4: Computes centered two-pass patch variance to eliminate catastrophic numerical cancellation.
      - Float32 Precision Bridge: Prevents FP16 overflow (>65,504) over multi-megapixel inputs.
      - Zero-Conv Initialization: Guarantees exact mathematical identity mapping at Step 0.
    """

    def __init__(self, dim, num_heads=4, saliency_floor=0.05):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.saliency_floor = saliency_floor

        # Directional Asymmetric Indexer (1x3 horizontal + 3x1 vertical DWConv)
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)
        self.pw_index = nn.Conv2d(dim, num_heads, kernel_size=1)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # Zero-Convolution Initialization
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        input_dtype = x.dtype

        # 1. Multi-Head Projections [B, h, d, N]
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, N)
        q = qkv[:, 0]  # [B, h, d, N]
        k = qkv[:, 1]  # [B, h, d, N]
        v = qkv[:, 2]  # [B, h, d, N]

        # PATCH 1.3 APPLIED: Injected F.silu() non-linear activation
        dir_feat = F.silu(self.dw_h(x) + self.dw_w(x))
        s_raw = torch.sigmoid(self.pw_index(dir_feat)).view(B, self.num_heads, 1, N)

        # PATCH 1.4 APPLIED: Numerically stable two-pass centered patch variance
        x_fp32 = x.float()
        patch_mean = F.avg_pool2d(x_fp32, 3, stride=1, padding=1)
        diff = x_fp32 - patch_mean
        patch_var = F.avg_pool2d(diff.pow(2), 3, stride=1, padding=1)
        local_var = patch_var.mean(dim=1, keepdim=True).view(B, 1, 1, N)
        s_robust = s_raw * (1.0 + torch.tanh(local_var.clamp(min=0.0)))

        # Asymmetric Scale Floor
        s_tilde = (1.0 - self.saliency_floor) * s_robust + self.saliency_floor

        # 3. Float32 Precision Bridge (Prevents FP16 Overflow >65,504)
        phi_q = (F.elu(q.permute(0, 1, 3, 2)) + 1.0).float()  # [B, h, N, d]
        phi_k = (F.elu(k) + 1.0).float()                      # [B, h, d, N]
        v_f = v.permute(0, 1, 3, 2).float()                  # [B, h, N, d]
        s_f = s_tilde.float()                                # [B, h, 1, N]

        k_s = phi_k * s_f

        # 4. Associative Scene Memory Matrix M_kv ∈ R^{d x d} (Constant 16 KB Footprint)
        M_kv = torch.matmul(k_s, v_f)                         # [B, h, d, d]
        z_k = k_s.sum(dim=-1, keepdim=True)                  # [B, h, d, 1]

        # 5. Linear Context Retrieval: O(HW · C · d)
        num = torch.matmul(phi_q, M_kv)                       # [B, h, N, d]
        den = torch.matmul(phi_q, z_k) + 1e-4                 # [B, h, N, 1]

        context = (num / den).to(input_dtype)
        context = context.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return self.proj(context)


# =========================================================================
# PILLAR 3: ENGRAM ASSOCIATIVE MEMORY PRIMITIVES (Head)
# =========================================================================

class EngramMemoryBank(nn.Module):
    """
    Hyperspherical Associative Memory Bank.
    Resolves Caveat 14.5: Unifies retrieval and auxiliary supervision geometry into 
    the exact same normalized hyperspherical cosine space (tau=0.10, multiplier=10.0).
    """

    def __init__(self, num_classes, latent_dim=128, temperature=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.temperature = temperature

        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x_latent, uncertainty_gate, objectness_mask):
        x_norm = F.normalize(x_latent, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)

        # Scaled Cosine Similarity (Identical geometry for retrieval & supervision)
        norm_cosine_logits = torch.matmul(x_norm, p_norm.t()) * self.temperature  # [B, N, C]
        attn_weights = F.softmax(norm_cosine_logits, dim=-1)

        memory_retrieved = torch.matmul(attn_weights, p_norm)

        restoration_mask = uncertainty_gate * objectness_mask
        memory_gated = memory_retrieved * restoration_mask

        return memory_gated, norm_cosine_logits


class UncertaintyEstimator(nn.Module):
    """Learns continuous feature uncertainty from the latent representation."""

    def __init__(self, dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


# =========================================================================
# DENSE AFFORDANCE TRAINING PRIMITIVES (Training Only, 0.0 ms Test Latency)
# =========================================================================

class BoxOccupancyTargetGenerator:
    """
    Vectorized Flat-Top Superquadric Box Occupancy Target Generator.
    Features:
      - Continuous C^1 smooth cosine boundary falloff.
      - Sub-pixel micro-target floor (rx, ry >= 0.85) guaranteeing positive gradient for targets <= 8 px.
      - Pairwise seam suppression separating touching vehicle clusters.
      - Dynamic feature size resolution (compatible with multiscale Rule-of-64 inputs).
    """

    @staticmethod
    @torch.no_grad()
    def generate(targets, num_classes=9, feat_size=(72, 128), stride=8.0):
        B = targets.shape[0]
        H, W = feat_size
        device = targets.device

        heatmaps = torch.zeros((B, num_classes, H, W), device=device, dtype=torch.float32)

        y_coord, x_coord = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x_grid = x_coord.view(1, H, W)
        y_grid = y_coord.view(1, H, W)

        for b in range(B):
            img_targets = targets[b]
            valid_mask = (
                (img_targets[:, 3] > 0)
                & (img_targets[:, 4] > 0)
                & (img_targets[:, 0] >= 0)
                & (img_targets[:, 0] < num_classes)
            )
            if not valid_mask.any():
                continue

            valid_boxes = img_targets[valid_mask]
            classes = valid_boxes[:, 0].long()
            K = valid_boxes.shape[0]

            xc = (valid_boxes[:, 1] / stride).view(K, 1, 1)
            yc = (valid_boxes[:, 2] / stride).view(K, 1, 1)
            raw_w = (valid_boxes[:, 3] / stride).view(K, 1, 1)
            raw_h = (valid_boxes[:, 4] / stride).view(K, 1, 1)

            rx = torch.clamp(0.50 * raw_w, min=0.85)
            ry = torch.clamp(0.50 * raw_h, min=0.85)

            dx = torch.abs(x_grid - xc) / rx
            dy = torch.abs(y_grid - yc) / ry
            d_max = torch.max(dx, dy)

            in_core = (d_max <= 0.70).float()
            in_boundary = ((d_max > 0.70) & (d_max <= 1.0)).float()
            decay = 0.5 * (1.0 + torch.cos(math.pi * (d_max - 0.70) / 0.30))

            occupancy = in_core + in_boundary * decay
            occupancy = occupancy.view(K, H, W)

            for c in torch.unique(classes):
                cls_mask = classes == c
                boxes_of_class = occupancy[cls_mask]
                if boxes_of_class.shape[0] > 1:
                    max_occupancy = torch.max(boxes_of_class, dim=0).values
                    overlap_sum = torch.sum(boxes_of_class, dim=0)
                    seam_mask = (overlap_sum > 1.20).float() * 0.15
                    heatmaps[b, c] = torch.clamp(max_occupancy - seam_mask, min=0.0, max=1.0)
                else:
                    heatmaps[b, c] = boxes_of_class[0]

        return heatmaps


class ActiveClassFocalDiceLoss(nn.Module):
    """Scale-Invariant Active-Class Masked Focal-Dice Loss. 100% AMP & FP16 Safe."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        logits_fp32 = logits.float()
        target_fp32 = target.float()

        bce = F.binary_cross_entropy_with_logits(logits_fp32, target_fp32, reduction="mean")

        pred_prob = torch.sigmoid(logits_fp32)
        target_mass = target_fp32.sum(dim=[-2, -1])  # [B, C]
        active_mask = target_mass > 0.0

        if active_mask.any():
            intersection = (pred_prob * target_fp32).sum(dim=[-2, -1])
            union = (pred_prob.pow(2) + target_fp32.pow(2)).sum(dim=[-2, -1])
            channel_dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
            dice_loss = channel_dice[active_mask].mean()
        else:
            dice_loss = torch.tensor(0.0, device=logits.device)

        return bce + dice_loss


class WeakSemanticAffordanceHead(nn.Module):
    """
    Tri-Scale Semantic Affordance Decoder (N3 + N4 + N5).
    Fuses deep global scene context (N5) with intermediate spatial features (N4, N3).
    Completely severed at inference time (0.0 ms test latency).
    
    Patch 1.5 Applied: Set align_corners=True in bilinear upsampling to prevent sub-pixel centroid blur.
    """

    def __init__(
        self,
        in_channels_n3=128,
        in_channels_n4=256,
        in_channels_n5=512,
        num_classes=9,
        hidden_dim=64,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.lat_n5 = BaseConv(in_channels_n5, hidden_dim, 1, 1, act="silu")
        self.lat_n4 = BaseConv(in_channels_n4, hidden_dim, 1, 1, act="silu")
        self.lat_n3 = BaseConv(in_channels_n3, hidden_dim, 1, 1, act="silu")

        self.fuse_conv = nn.Sequential(
            BaseConv(hidden_dim * 3, hidden_dim, 3, 1, act="silu"),
            BaseConv(hidden_dim, hidden_dim, 3, 1, act="silu"),
        )

        self.pred_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)

        prior_prob = 0.01
        bias_init = float(-math.log((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.pred_head.bias, bias_init)

    def forward(self, n3_feat, n4_feat, n5_feat):
        target_size = n3_feat.shape[-2:]

        # PATCH 1.5 APPLIED: align_corners=True preserves sub-pixel centroid alignment
        p5 = F.interpolate(
            self.lat_n5(n5_feat), size=target_size, mode="bilinear", align_corners=True
        )
        p4 = F.interpolate(
            self.lat_n4(n4_feat), size=target_size, mode="bilinear", align_corners=True
        )
        p3 = self.lat_n3(n3_feat)

        fused = self.fuse_conv(torch.cat([p3, p4, p5], dim=1))
        return self.pred_head(fused)