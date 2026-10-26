#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.8: P4/P3 Dedicated Residual Assists & Hybrid GRC Highway

import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import CSPLayer
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
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

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


class BQSA_P4_Assist(nn.Module):
    """
    Sparse-DETR Aligned Block-Queried Attention Residual Assist (v3.8 Final).
    Features:
      - Kaiming Proj + Gate (-4.0) Breaks the double-zero deadlock while preserving Step-0 identity (98.2%).
      - 100% Deterministic Top-K selection in Train & Eval (Zero Gumbel/STE mismatch!).
      - Detached Scorer Input: Indexer loss trains ONLY the scorer; zero distortion of P4 features.
      - Global Context Scorer: Global average-pooled channels provide scene-aware block ranking.
    """
    def __init__(self, c, block_size=4, sparsity_ratio=0.35, stride=16, detach_scorer_input=True):
        super().__init__()
        self.c = c
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        self.block_px = block_size * stride  # 64 px for P4 (bs=4, stride=16)
        self.detach_scorer_input = detach_scorer_input

        self.rca = UnifiedRCA(c)
        self.rmbconv = UnifiedRMBConv(c)

        # Global-Context Scorer Network
        self.indexer_norm = RMSNorm(c, eps=1e-5)
        self.indexer_local = nn.Conv2d(c, c // 2, kernel_size=3, padding=1, groups=c // 2)
        self.indexer_global = nn.Conv2d(c, c // 2, kernel_size=1)
        self.indexer_fuse = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(c, 1, kernel_size=1)
        )
        self.pool_alpha = nn.Parameter(torch.tensor(0.0))

        self.qkv = nn.Conv2d(c, c * 3, 1)

        # BREAKING THE DEADLOCK:
        # Kaiming uniform on proj ensure dL/d(gr_gate) is non-zero from Iteration 0!
        # gr_gate = -4.0 alone provides 98.2% Step-0 identity protection.
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=1.0)
        self.gr_gate = nn.Parameter(torch.full((1, c, 1, 1), -4.0))

    def block_grid(self, H, W):
        bs = min(self.block_size, H, W)
        return H // bs, W // bs

    def num_selected(self, total_blocks):
        k_target = int(round(self.sparsity_ratio * float(total_blocks)))
        return max(24, min(48, k_target)) if total_blocks >= 24 else max(1, total_blocks)

    def forward(self, x):
        B, C, H, W = x.shape
        bs = min(self.block_size, H, W)
        num_h, num_w = self.block_grid(H, W)
        total_blocks = num_h * num_w
        tokens_per_b = bs * bs
        k_actual = self.num_selected(total_blocks)

        feat_cond = self.rca(self.rmbconv(x))

        # 1. SCORER (Detached from detection trunk to preserve P4 representation)
        s_in = feat_cond.detach() if self.detach_scorer_input else feat_cond
        s_in = self.indexer_norm(s_in)
        f_loc = self.indexer_local(s_in)
        f_glo = self.indexer_global(F.adaptive_avg_pool2d(s_in, (1, 1))).expand(-1, -1, H, W)
        scores = self.indexer_fuse(torch.cat([f_loc, f_glo], dim=1)).float()

        s_avg = F.avg_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        s_max = F.max_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        alpha = torch.sigmoid(self.pool_alpha.float())
        block_scores = alpha * s_max + (1.0 - alpha) * s_avg  # [B, N] fp32 logits

        # 2. 100% DETERMINISTIC TOP-K SELECTION (Train & Eval Identical)
        topk_idx = torch.topk(block_scores.detach(), k_actual, dim=-1).indices

        # 3. DENSE Q, SPARSE K/V WITH AMBIENT QUAD TOKENS
        qkv = self.qkv(feat_cond).view(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        def to_blocks(t):
            return (t.reshape(B, C, num_h, bs, num_w, bs)
                     .permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b))

        idx = topk_idx[:, :, None, None].expand(-1, -1, C, tokens_per_b)
        k_sel = torch.gather(to_blocks(k), 1, idx).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)
        v_sel = torch.gather(to_blocks(v), 1, idx).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)

        k_quad = F.adaptive_avg_pool2d(k, (2, 2)).view(B, C, 4)
        v_quad = F.adaptive_avg_pool2d(v, (2, 2)).view(B, C, 4)

        # Clamped RMS matching
        rms_k_loc = torch.sqrt(k_sel.float().pow(2).mean(dim=(1, 2), keepdim=True) + 1e-5)
        rms_k_amb = torch.sqrt(k_quad.float().pow(2).mean(dim=(1, 2), keepdim=True) + 1e-5)
        k_quad = (k_quad.float() * torch.clamp(rms_k_loc / rms_k_amb, min=0.10, max=10.0)).to(k.dtype)

        rms_v_loc = torch.sqrt(v_sel.float().pow(2).mean(dim=(1, 2), keepdim=True) + 1e-5)
        rms_v_amb = torch.sqrt(v_quad.float().pow(2).mean(dim=(1, 2), keepdim=True) + 1e-5)
        v_quad = (v_quad.float() * torch.clamp(rms_v_loc / rms_v_amb, min=0.10, max=10.0)).to(v.dtype)

        k_all = torch.cat([k_sel, k_quad], dim=-1)
        v_all = torch.cat([v_sel, v_quad], dim=-1)

        q_flat = q.reshape(B, C, H * W).transpose(-2, -1).float()
        logits = torch.matmul(q_flat, k_all.float()) * (C ** -0.5)
        bias = torch.zeros(k_all.shape[-1], device=logits.device, dtype=logits.dtype)
        bias[-4:] = 0.50
        attn = F.softmax(logits + bias, dim=-1)

        context = torch.matmul(v_all.float(), attn.transpose(-2, -1)).to(v_all.dtype).reshape(B, C, H, W)
        delta = torch.sigmoid(self.gr_gate) * self.proj(context)
        
        return delta, topk_idx, block_scores


class BQSA_P3_Assist(nn.Module):
    """
    P3 Block-Local Attention Residual Assist (v3.8 Final).
    Consumes P4's top-k indices deterministically; initialized with Kaiming proj.
    """
    def __init__(self, c, block_size=8, sparsity_ratio=0.35):
        super().__init__()
        self.c = c
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio

        self.rca = UnifiedRCA(self.c)
        self.rmbconv = UnifiedRMBConv(self.c)
        self.qkv = nn.Conv2d(self.c, self.c * 3, 1)

        self.proj = nn.Conv2d(self.c, self.c, 1, bias=False)
        nn.init.kaiming_uniform_(self.proj.weight, a=1.0)
        self.gr_gate = nn.Parameter(torch.full((1, self.c, 1, 1), -4.0))

    def forward(self, x, inherited_topk_idx):
        B, C, H, W = x.shape
        feat_cond = self.rca(self.rmbconv(x))

        bs = min(self.block_size, H, W)
        num_h, num_w = H // bs, W // bs
        total_blocks = num_h * num_w
        tokens_per_b = bs * bs

        k_actual = min(inherited_topk_idx.shape[1], total_blocks)
        qkv = self.qkv(feat_cond).view(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q_b = q.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        k_b = k.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        v_b = v.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)

        valid_idx = torch.clamp(inherited_topk_idx[:, :k_actual], 0, total_blocks - 1)
        idx_exp = valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, tokens_per_b)

        q_loc = torch.gather(q_b, 1, idx_exp).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b).transpose(-2, -1)
        k_loc = torch.gather(k_b, 1, idx_exp).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)
        v_loc = torch.gather(v_b, 1, idx_exp).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b).transpose(-2, -1)

        attn_fp32 = F.softmax(torch.bmm(q_loc.float(), k_loc.float()) * (C ** -0.5), dim=-1)
        ctx_loc = torch.bmm(attn_fp32, v_loc.float()).to(v.dtype).reshape(B, k_actual, tokens_per_b, C)

        delta_flat = torch.zeros(B, total_blocks, C * tokens_per_b, device=x.device, dtype=ctx_loc.dtype)
        idx_flat = valid_idx.unsqueeze(-1).expand(-1, -1, C * tokens_per_b)
        ctx_flat = ctx_loc.permute(0, 1, 3, 2).reshape(B, k_actual, C * tokens_per_b)
        delta_flat.scatter_(1, idx_flat, ctx_flat)

        context_delta = delta_flat.reshape(B, num_h, num_w, C, bs, bs).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        gate = torch.sigmoid(self.gr_gate)

        return gate * self.proj(context_delta), None


class ScaleAttnRes(nn.Module):
    """
    Hybrid GRC Highway: True Zero-Centered Residual Delta (Bottleneck 1 Fix).
    Eliminates the 2.0x activation explosion entering head.stems.
    """
    def __init__(self, dim, num_sources=3, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_sources = num_sources
        self.scale = dim ** -0.5
        
        self.w = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim, eps=1e-5)
        
        prior_logits = torch.zeros(num_sources)
        prior_logits[-1] = 4.0  # Base trunk is the last source
        self.register_buffer("prior_logits", prior_logits.view(-1, 1, 1, 1, 1))

        self.refine = CSPLayer(dim, dim, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu")
        # Step-0 Identity Guard: Zero the conv kernel so refine output === 0.00000 on Day 1!
        nn.init.zeros_(self.refine.conv3.conv.weight)
        if self.refine.conv3.conv.bias is not None:
            nn.init.zeros_(self.refine.conv3.conv.bias)
        # Keep BatchNorm active so backward gradients flow smoothly through the block!
        nn.init.ones_(self.refine.conv3.bn.weight)
        nn.init.zeros_(self.refine.conv3.bn.bias)
        
        self.gr_gate = nn.Parameter(torch.full((1, dim, 1, 1), -4.0))
        self.refine_gate = nn.Parameter(torch.full((1, dim, 1, 1), -4.0))

    def forward(self, sources):
        base_k = sources[-1]
        V = torch.stack(sources, dim=0)
        K = torch.stack([self.norm(s) for s in sources], dim=0)
        
        w_safe = torch.clamp(self.w.to(dtype=K.dtype), min=-3.0, max=3.0)
        logits = torch.einsum('c, m b c h w -> m b h w', w_safe, K) * self.scale
        
        logits_biased = logits.unsqueeze(2) + self.prior_logits.to(logits.dtype)
        weights = F.softmax(logits_biased.float(), dim=0).to(V.dtype)
        
        out_routed = torch.sum(weights * V, dim=0)
        
        # ZERO-CENTERED DELTA: On Step 0, out_routed ≈ base_k => delta ≈ 0.00000
        routing_delta = out_routed - base_k
        gate_routing = torch.sigmoid(self.gr_gate)
        
        delta_csp = self.refine(base_k)
        gate_refine = torch.sigmoid(self.refine_gate)
        
        total_delta = gate_routing * routing_delta + gate_refine * delta_csp
        return total_delta