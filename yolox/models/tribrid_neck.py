#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.7: Step-0 Backbone Trunk Highways & Hybrid GRC Bottom-Up Processing

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import UnifiedRCA, UnifiedRMBConv


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_fp32 = x.float()
        norm_fp32 = torch.rsqrt(x_fp32.pow(2).mean(dim=1, keepdim=True) + self.eps)
        gamma_safe = torch.clamp(self.weight.float(), min=1e-4).view(1, -1, 1, 1)
        return (x_fp32 * norm_fp32 * gamma_safe).to(x.dtype)


class RotaryEmbedding2D(nn.Module):
    """
    Continuous Aspect-Preserved 2D-RoPE.
    Calibrated with Vision Base Frequency theta=100.0 (RoPE-ViT, ECCV 2024):
      - Contracts wavelengths to fit small vision grids (18x32 at P5).
      - Eliminates the dormant frequency trap of theta=10000.
      - Maps continuous coordinates to canonical base grid (18, 32) for scale equivariance.
    """
    def __init__(self, dim, theta=100.0):
        super().__init__()
        self.dim = dim
        self.theta = float(theta)
        half_dim = dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, H, W):
        q = q.contiguous()
        k = k.contiguous()
        device = q.device
        dtype = q.dtype
        half = q.shape[-1] // 2

        norm_x = (torch.arange(W, device=device).float() / max(1.0, float(W - 1.0))) * (32.0 - 1.0)
        norm_y = (torch.arange(H, device=device).float() / max(1.0, float(H - 1.0))) * (18.0 - 1.0)

        y, x = torch.meshgrid(norm_y, norm_x, indexing="ij")
        x = x.reshape(-1)
        y = y.reshape(-1)

        inv_freq = self.inv_freq.to(device)
        freqs_x = torch.einsum("i, j -> ij", x, inv_freq)
        freqs_y = torch.einsum("i, j -> ij", y, inv_freq)

        emb_x = torch.cat([freqs_x, freqs_x], dim=-1)
        emb_y = torch.cat([freqs_y, freqs_y], dim=-1)

        cos_x, sin_x = emb_x.cos().to(dtype), emb_x.sin().to(dtype)
        cos_y, sin_y = emb_y.cos().to(dtype), emb_y.sin().to(dtype)

        qx, qy = q[..., :half], q[..., half:]
        kx, ky = k[..., :half], k[..., half:]

        qx = (qx * cos_x) + (self._rotate_half(qx) * sin_x)
        kx = (kx * cos_x) + (self._rotate_half(kx) * sin_x)
        qy = (qy * cos_y) + (self._rotate_half(qy) * sin_y)
        ky = (ky * cos_y) + (self._rotate_half(ky) * sin_y)

        return torch.cat([qx, qy], dim=-1).contiguous(), torch.cat([kx, ky], dim=-1).contiguous()


class P5ExclusiveDenseAttention(nn.Module):
    """
    P5 Exclusive Dense Global Attention with Step-0 Identity Guards & Dark Channel Prior Integration.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # STEP-0 IDENTITY GUARDS:
        self.lambda_fog = nn.Parameter(torch.tensor(0.0))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        self.rope = RotaryEmbedding2D(self.head_dim, theta=100.0)
        self.w_r1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.w_r2 = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.normal_(self.w_r1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.w_r2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.w_r1.bias, -4.0)
        if self.w_r2.bias is not None:
            nn.init.zeros_(self.w_r2.bias)

    def forward(self, x, h_atmo=None):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k, H, W)

        attn_fp32 = F.softmax(torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale, dim=-1)

        # Saliency Map Extraction for P4 Macro Spatial Prior
        diag_elements = attn_fp32.diagonal(dim1=-2, dim2=-1)
        saliency_sum = attn_fp32.sum(dim=-2) - diag_elements
        saliency = saliency_sum.mean(dim=1).reshape(B, 1, H, W) / max(1.0, float(N - 1))

        if h_atmo is not None:
            h_ratio = h_atmo.view(B, 1).to(x.dtype)
        else:
            h_ratio = torch.zeros(B, 1, device=x.device, dtype=x.dtype)

        z_spatial = self.proj(
            torch.matmul(attn_fp32, v.float()).to(v.dtype).permute(0, 1, 3, 2).reshape(B, C, H, W)
        )
        g_xsa = torch.sigmoid(
            self.w_r1(x) + self.w_r2(z_spatial) + (self.lambda_fog * h_ratio.view(B, 1, 1, 1))
        ).to(x.dtype)
        out = (x + g_xsa * z_spatial).to(x.dtype)
        return out, saliency, h_ratio


class C2f_BQSA_P4(nn.Module):
    """
    P4 HySparse2 Block-Queried Attention with Step-0 Backbone Trunk Highway (Zone 3).
    Preserves 98.2% of pre-trained Dark4 features (C4) on Step 0 while progressively
    earning multi-scale C2f representation during training.
    """
    def __init__(self, c1, c2, n=1, block_size=4, sparsity_ratio=0.35):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        self.tau = 1.0

        # PRETRAINED CSPLAYER TRUNK (Loads from COCO C3_p4!):
        self.conv1 = BaseConv(c1, self.c, 1, 1, act="silu")
        self.conv2 = BaseConv(c1, self.c, 1, 1, act="silu")
        self.conv3 = BaseConv(2 * self.c, c2, 1, 1, act="silu")
        self.m = nn.Sequential(*[Bottleneck(self.c, self.c, shortcut=False, expansion=1.0, act="silu") for _ in range(n)])

        # BQSA RESIDUAL ATTENTION BRANCH:
        self.rca = UnifiedRCA(self.c)
        self.rmbconv = UnifiedRMBConv(self.c)
        self.indexer_norm = RMSNorm(self.c, eps=1e-5)
        self.indexer = nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c),
            nn.Conv2d(self.c, 1, 1)
        )
        self.pool_alpha = nn.Parameter(torch.tensor(0.0))
        self.qkv = nn.Conv2d(self.c, self.c * 3, 1)

        # Step-0 Identity Guards (Criterion 5):
        self.proj = nn.Conv2d(self.c, self.c, 1, bias=False)
        nn.init.zeros_(self.proj.weight)  # Zero projection at initialization
        self.gr_gate = nn.Parameter(torch.full((1, self.c, 1, 1), -4.0))  # Gate ≈ 0.018

    def set_tau(self, tau):
        self.tau = tau

    def _rasterize_gt_blocks(self, targets, batch_size, num_h, num_w, device):
        """100% Vectorized GPU Rasterizer with Padded-Target Protection."""
        if targets is None or targets.shape[0] == 0:
            return torch.zeros(batch_size, num_h * num_w, device=device)

        valid = (targets[:, :, 3] > 0) & (targets[:, :, 4] > 0)
        x1 = targets[:, :, 1] - targets[:, :, 3] / 2.0
        y1 = targets[:, :, 2] - targets[:, :, 4] / 2.0
        x2 = targets[:, :, 1] + targets[:, :, 3] / 2.0
        y2 = targets[:, :, 2] + targets[:, :, 4] / 2.0

        c_min = (x1 / 64.0).floor().clamp(0, num_w - 1).long()
        c_max = ((x2 - 1e-4) / 64.0).floor().clamp(0, num_w - 1).long()
        r_min = (y1 / 64.0).floor().clamp(0, num_h - 1).long()
        r_max = ((y2 - 1e-4) / 64.0).floor().clamp(0, num_h - 1).long()

        gy, gx = torch.meshgrid(torch.arange(num_h, device=device), torch.arange(num_w, device=device), indexing="ij")
        gx = gx.view(1, 1, num_h, num_w)
        gy = gy.view(1, 1, num_h, num_w)

        in_x = (gx >= c_min.unsqueeze(-1).unsqueeze(-1)) & (gx <= c_max.unsqueeze(-1).unsqueeze(-1))
        in_y = (gy >= r_min.unsqueeze(-1).unsqueeze(-1)) & (gy <= r_max.unsqueeze(-1).unsqueeze(-1))
        
        in_box = in_x & in_y & valid.unsqueeze(-1).unsqueeze(-1)
        return in_box.any(dim=1).float().view(batch_size, -1)

    def forward(self, x, p5_saliency, h_s, targets=None):
        # 1. Pretrained CSPLayer Trunk (100% COCO feature flow!)
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)

        # 2. Residual BQSA Attention Branch (Evaluates on x_1)
        B, C, H, W = x_1.shape
        bs = min(self.block_size, H, W)
        num_h, num_w = H // bs, W // bs
        total_blocks = num_h * num_w
        tokens_per_b = bs * bs

        k_target = int(round(self.sparsity_ratio * float(total_blocks)))
        k_actual = max(24, min(48, k_target)) if total_blocks >= 24 else max(1, total_blocks)

        feat_cond = self.rca(self.rmbconv(x_1))
        s_up = F.interpolate(p5_saliency.float(), size=(H, W), mode='nearest')
        block_saliency = F.avg_pool2d(s_up, kernel_size=bs, stride=bs).view(B, -1)

        var_saliency = torch.var(block_saliency, dim=-1, keepdim=True, unbiased=False)
        p5_macro = (block_saliency - block_saliency.mean(dim=-1, keepdim=True)) / \
                   torch.sqrt(torch.clamp(var_saliency, min=1e-8)) if total_blocks > 1 else torch.zeros_like(block_saliency)

        scores = self.indexer(self.indexer_norm(feat_cond))
        scores_avg = F.avg_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        scores_max = F.max_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        alpha = torch.sigmoid(self.pool_alpha)
        block_scores = alpha * scores_max + (1.0 - alpha) * scores_avg

        var_bs = torch.var(block_scores, dim=-1, keepdim=True, unbiased=False)
        bs_norm = (block_scores - block_scores.mean(dim=-1, keepdim=True)) / \
                  torch.sqrt(torch.clamp(var_bs, min=1e-6)) if total_blocks > 1 else torch.zeros_like(block_scores)

        candidate_scores_fp32 = bs_norm.float() + 0.15 * p5_macro

        if self.training:
            if torch.is_tensor(h_s):
                h_stat = h_s.float().view(B, -1)[:, :1].clamp(0.0, 1.0) if h_s.numel() >= B else h_s.float().mean().view(1, 1).clamp(0.0, 1.0)
            else:
                h_stat = torch.tensor(float(h_s), device=x.device).view(1, 1).clamp(0.0, 1.0)

            tau_adaptive = torch.clamp(self.tau * (1.0 - 0.35 * h_stat), min=0.10)
            u1, u2 = torch.rand_like(candidate_scores_fp32), torch.rand_like(candidate_scores_fp32)
            gumbel = torch.log(u1 + 1e-7) - torch.log(u2 + 1e-7)
            soft_scores = torch.sigmoid((candidate_scores_fp32 + gumbel) / tau_adaptive).to(x_1.dtype)
            _, topk_idx = torch.topk(soft_scores, k_actual, dim=-1)
            hard_mask = torch.zeros_like(soft_scores).scatter_(-1, topk_idx, 1.0)
            ste_mask = (hard_mask - soft_scores).detach() + soft_scores
            ste_weights = torch.gather(ste_mask, -1, topk_idx).unsqueeze(-1).unsqueeze(-1)
        else:
            _, topk_idx = torch.topk(candidate_scores_fp32, k_actual, dim=-1)
            ste_weights = 1.0

        qkv = self.qkv(feat_cond).view(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        k_b = k.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        v_b = v.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)

        idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, tokens_per_b)
        k_sel = torch.gather(k_b, 1, idx_exp).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)
        v_sel = (torch.gather(v_b, 1, idx_exp) * ste_weights).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)

        k_quad = F.adaptive_avg_pool2d(k, (2, 2)).view(B, C, 4)
        v_quad = F.adaptive_avg_pool2d(v, (2, 2)).view(B, C, 4)
        rms_k_loc = torch.sqrt(k_sel.float().pow(2).mean(dim=[1, 2], keepdim=True) + 1e-5)
        rms_k_amb = torch.sqrt(k_quad.float().pow(2).mean(dim=[1, 2], keepdim=True) + 1e-5)
        k_quad_scaled = (k_quad.float() * torch.clamp(rms_k_loc / rms_k_amb, 0.1, 10.0)).to(k.dtype)

        rms_v_loc = torch.sqrt(v_sel.float().pow(2).mean(dim=[1, 2], keepdim=True) + 1e-5)
        rms_v_amb = torch.sqrt(v_quad.float().pow(2).mean(dim=[1, 2], keepdim=True) + 1e-5)
        v_quad_scaled = (v_quad.float() * torch.clamp(rms_v_loc / rms_v_amb, 0.1, 10.0)).to(v.dtype)

        k_all = torch.cat([k_sel, k_quad_scaled], dim=-1)
        v_all = torch.cat([v_sel, v_quad_scaled], dim=-1)

        q_flat = q.reshape(B, C, H * W).transpose(-2, -1).float()
        ambient_bias = torch.zeros(1, 1, k_all.shape[-1], device=x.device, dtype=torch.float32)
        ambient_bias[..., -4:] = 0.50
        logits_fp32 = torch.matmul(q_flat, k_all.float()) * (C ** -0.5) + ambient_bias
        attn_fp32 = F.softmax(logits_fp32, dim=-1)

        context = torch.matmul(v_all.float(), attn_fp32.transpose(-2, -1)).to(v_all.dtype).reshape(B, C, H, W)
        gate = torch.sigmoid(self.gr_gate)

        # Residual attention enhancement on top of pre-trained CSPLayer branch
        x_1 = x_1 + gate * self.proj(context)

        # 3. Pretrained Fusion Conv (conv3)
        p4_out = self.conv3(torch.cat([x_1, x_2], dim=1))

        indexer_loss = torch.tensor(0.0, device=x.device)
        if self.training and targets is not None:
            f_target = self._rasterize_gt_blocks(targets, B, num_h, num_w, x.device)
            p_b = torch.sigmoid(block_scores.float())
            p_t = p_b * f_target.float() + (1.0 - p_b) * (1.0 - f_target.float())
            focal_weight = ((0.85 * f_target + 0.15 * (1.0 - f_target)) * (1.0 - p_t).pow(2.0)).detach()
            indexer_loss = (focal_weight * F.binary_cross_entropy_with_logits(block_scores.float(), f_target, reduction="none")).mean()

        return p4_out, topk_idx, indexer_loss


class C2f_BQSA_P3(nn.Module):
    """
    Pretrained CSPLayer Trunk with Residual Synchronized BQSA Attention (P3).
      - Trunk: Loads pre-trained COCO C3_p3 weights (100% trained on Day 1!).
      - Attention: Inherits P4 indices, evaluates local attention, and adds context
        as a Step-0 Identity-Guarded Residual (proj=0, gate=-4.0).
      - Zero-Delta Bypass: Unselected blocks receive a delta of EXACTLY 0.00000.
    """
    def __init__(self, c1, c2, n=1, block_size=8, sparsity_ratio=0.35):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio

        # PRETRAINED CSPLAYER TRUNK (Loads directly from COCO C3_p3 weights!):
        self.conv1 = BaseConv(c1, self.c, 1, 1, act="silu")
        self.conv2 = BaseConv(c1, self.c, 1, 1, act="silu")
        self.conv3 = BaseConv(2 * self.c, c2, 1, 1, act="silu")
        self.m = nn.Sequential(*[
            Bottleneck(self.c, self.c, shortcut=False, expansion=1.0, act="silu") 
            for _ in range(n)
        ])

        # BQSA RESIDUAL ATTENTION BRANCH:
        self.rca = UnifiedRCA(self.c)
        self.rmbconv = UnifiedRMBConv(self.c)
        self.qkv = nn.Conv2d(self.c, self.c * 3, 1)

        # STEP-0 IDENTITY GUARDS (Criterion 5):
        self.proj = nn.Conv2d(self.c, self.c, 1, bias=False)
        nn.init.zeros_(self.proj.weight)  # Zero projection at initialization
        self.gr_gate = nn.Parameter(torch.full((1, self.c, 1, 1), -4.0))  # Gate ≈ 0.018

    def forward(self, x, inherited_topk_idx):
        # 1. Pretrained CSPLayer Trunk Split
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)

        # 2. Residual BQSA Attention Branch (Evaluates on x_1)
        B, C, H, W = x_1.shape
        feat_cond = self.rca(self.rmbconv(x_1))

        bs = min(self.block_size, H, W)
        num_h, num_w = H // bs, W // bs
        total_blocks = num_h * num_w
        tokens_per_b = bs * bs

        k_actual = min(inherited_topk_idx.shape[1], total_blocks)
        qkv = self.qkv(feat_cond).view(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Block-wise reshaping
        q_b = q.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        k_b = k.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        v_b = v.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)

        # Gather inherited active blocks
        valid_idx = torch.clamp(inherited_topk_idx[:, :k_actual], 0, total_blocks - 1)
        idx_exp = valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, tokens_per_b)

        q_loc = torch.gather(q_b, 1, idx_exp).permute(0, 1, 3, 2).reshape(B * k_actual, tokens_per_b, C)
        k_loc = torch.gather(k_b, 1, idx_exp).reshape(B * k_actual, C, tokens_per_b)
        v_loc = torch.gather(v_b, 1, idx_exp).permute(0, 1, 3, 2).reshape(B * k_actual, tokens_per_b, C)

        attn_fp32 = F.softmax(torch.bmm(q_loc.float(), k_loc.float()) * (C ** -0.5), dim=-1)
        ctx_loc = torch.bmm(attn_fp32, v_loc.float()).to(v_loc.dtype).reshape(B, k_actual, tokens_per_b, C)

        # ZERO-DELTA RESIDUAL BYPASS:
        # Unselected blocks receive delta = 0.00000; native trunk features pass unhindered!
        delta_flat = torch.zeros(B, total_blocks, C * tokens_per_b, device=x.device, dtype=ctx_loc.dtype)
        idx_flat = valid_idx.unsqueeze(-1).expand(-1, -1, C * tokens_per_b)
        ctx_flat = ctx_loc.permute(0, 1, 3, 2).reshape(B, k_actual, C * tokens_per_b)
        delta_flat.scatter_(1, idx_flat, ctx_flat)

        context_delta = delta_flat.reshape(B, num_h, num_w, C, bs, bs).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # Identity-Guarded Residual Addition
        gate = torch.sigmoid(self.gr_gate)
        x_1 = x_1 + gate * self.proj(context_delta)

        # 3. Pretrained Fusion Conv (conv3)
        p3_out = self.conv3(torch.cat([x_1, x_2], dim=1))
        return p3_out


class C2f(nn.Module):
    """Faster, lighter CSP Bottleneck with 2 convolutions (YOLOv8 style)."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = BaseConv(c1, 2 * self.c, 1, 1)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, expansion=1.0, depthwise=False, act="silu")
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ScaleAttnRes(nn.Module):
    """
    Hybrid GRC Highway: Multi-Scale Attention Routing + C2f Non-Linear Refinement.
      - Source 0 (Primary Trunk P_l) prior bias => Softmax([4, 0, 0]) = 96.5% on Step 0!
      - Pure Scale-Preserving Output: Zero GroupNorm mean-shift (preserves +0.40 SiLU mean).
      - C2f Refinement Block: Richest multi-level gradient paths with Step-0 Quad-Zero Guard.
    """
    def __init__(self, dim, num_sources=3, spatial_ratio=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_sources = num_sources
        self.scale = dim ** -0.5
        
        self.w = nn.Parameter(torch.zeros(dim))  # In Group 2 (No-Decay)
        self.norm = RMSNorm(dim, eps=1e-5)
        
        # Step-0 Trunk Prior: Source 0 (P_l) gets 96.5% - 98.2% weight at Step 0
        prior_logits = torch.zeros(num_sources)
        prior_logits[0] = 4.0
        self.register_buffer("prior_logits", prior_logits.view(-1, 1, 1, 1, 1))

        # C2F NON-LINEAR REFINEMENT (Richest Gradient Paths):
        self.refine = C2f(dim, dim, n=1, shortcut=True, e=0.5)
        
        # STEP-0 QUAD-ZERO IDENTITY GUARD (Criterion 5):
        # Guarantees self.refine(x) outputs EXACT 0.00000 on Day 1!
        nn.init.zeros_(self.refine.cv2.conv.weight)
        if self.refine.cv2.conv.bias is not None:
            nn.init.zeros_(self.refine.cv2.conv.bias)
        nn.init.zeros_(self.refine.cv2.bn.weight)
        nn.init.zeros_(self.refine.cv2.bn.bias)
        
        self.refine_gate = nn.Parameter(torch.full((1, dim, 1, 1), -4.0))

    def forward(self, sources):
        V = torch.stack(sources, dim=0)          # [M, B, C, H, W]
        K = torch.stack([self.norm(s) for s in sources], dim=0)
        
        w_safe = torch.clamp(self.w.to(dtype=K.dtype), min=-3.0, max=3.0)
        logits = torch.einsum('c, m b c h w -> m b h w', w_safe, K) * self.scale
        
        logits_biased = logits.unsqueeze(2) + self.prior_logits.to(logits.dtype)
        weights = F.softmax(logits_biased.float(), dim=0).to(V.dtype)
        
        # 1. Bounded Multi-Scale GRC Skip Routing
        out_routed = torch.sum(weights * V, dim=0)

        # 2. C2f Non-Linear Feature Refinement (=== 0.00000 on Step 0)
        delta_c2f = self.refine(out_routed)
        gate_refine = torch.sigmoid(self.refine_gate)
        out_final = out_routed + gate_refine * delta_c2f

        return out_final