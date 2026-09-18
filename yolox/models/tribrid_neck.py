#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Resolution-Adaptive Tribrid BQSA Neck & Symmetrical Scale-AttnRes

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import SimAM, CoordinateAttention, MBConvConditioner


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with FP16-safe epsilon floor (V2)"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * norm * self.weight.view(1, -1, 1, 1)


class ScaleAttnRes(nn.Module):
    """
    Kimi Attention Residuals (AttnRes) with Spatial Area Gradient Normalization.
    """
    def __init__(self, dim, num_sources=3, spatial_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.num_sources = num_sources
        self.scale = (dim ** -0.5) / math.sqrt(spatial_ratio)
        self.w = nn.Parameter(torch.zeros(dim))  # Zero-initialized
        self.norm = RMSNorm(dim, eps=1e-5)

    def forward(self, sources):
        V = torch.stack(sources, dim=0)  # [M, B, C, H, W]
        K = torch.stack([self.norm(s) for s in sources], dim=0)
        
        logits = torch.einsum('c, m b c h w -> m b h w', self.w, K) * self.scale
        weights = F.softmax(logits, dim=0)
        
        return torch.sum(weights.unsqueeze(2) * V, dim=0)


class RotaryEmbedding2D(nn.Module):
    """
    2D-GRAPE-M Dynamic Commuting Rotation on Disjoint Subspaces.
    Dynamically generates grids matching (H, W) to support arbitrary resolutions and profiler inputs.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, H, W):
        device = q.device
        dtype = q.dtype
        half = q.shape[-1] // 2

        # Dynamic 2D spatial grid matching exact input resolution
        y, x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing="ij"
        )
        x = x.reshape(-1)
        y = y.reshape(-1)

        inv_freq = self.inv_freq.to(device)
        freqs_x = torch.einsum("i, j -> ij", x.float(), inv_freq)
        freqs_y = torch.einsum("i, j -> ij", y.float(), inv_freq)

        emb_x = torch.cat([freqs_x, freqs_x], dim=-1)
        emb_y = torch.cat([freqs_y, freqs_y], dim=-1)

        cos_x = emb_x.cos().to(dtype)
        sin_x = emb_x.sin().to(dtype)
        cos_y = emb_y.cos().to(dtype)
        sin_y = emb_y.sin().to(dtype)

        qx, qy = q[..., :half], q[..., half:]
        kx, ky = k[..., :half], k[..., half:]

        qx = (qx * cos_x) + (self._rotate_half(qx) * sin_x)
        kx = (kx * cos_x) + (self._rotate_half(kx) * sin_x)
        qy = (qy * cos_y) + (self._rotate_half(qy) * sin_y)
        ky = (ky * cos_y) + (self._rotate_half(ky) * sin_y)

        return torch.cat([qx, qy], dim=-1), torch.cat([kx, ky], dim=-1)


class P5ExclusiveDenseAttention(nn.Module):
    """P5 Full Dense Attention with XSA, 2D-GRAPE-M, and Dual-Input GR Gate."""
    def __init__(self, dim, num_heads=4, lambda_fog=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.lambda_fog = lambda_fog
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.rope = RotaryEmbedding2D(self.head_dim)
        
        self.w_r1 = nn.Conv2d(dim, dim, 1)
        self.w_r2 = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.w_r1.weight)
        nn.init.zeros_(self.w_r2.weight)
        if self.w_r1.bias is not None:
            nn.init.zeros_(self.w_r1.bias)
        if self.w_r2.bias is not None:
            nn.init.zeros_(self.w_r2.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 1. Commuting 2D Position Encoding (Dynamically resolved for H, W)
        q, k = self.rope(q, k, H, W)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 2. In-Place Diagonal Zeroing for Exclusive Centrality (D4)
        exclusive_attn = attn.clone()
        exclusive_attn.diagonal(dim1=-2, dim2=-1).zero_()
        saliency = exclusive_attn.sum(dim=-2).mean(dim=1).reshape(B, 1, H, W) / max(1.0, float(N - 1))
        
        s_prob = F.softmax(saliency.flatten(1) * 10.0, dim=-1)
        h_s = -torch.sum(s_prob * torch.log(s_prob + 1e-8), dim=-1).mean()
        
        # 3. Dense Multi-Head Aggregation & XSA Subtraction
        y = torch.matmul(attn, v)
        v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-6
        proj_scalar = (y * v).sum(dim=-1, keepdim=True) / v_norm_sq
        z = y - proj_scalar * v
        z_spatial = self.proj(z.permute(0, 1, 3, 2).reshape(B, C, H, W))
        
        # 4. Dual-Input Gated Residual Highway
        h_ratio = torch.clamp(h_s / math.log(max(2, N)), 0.0, 1.0)
        g_xsa = torch.sigmoid(self.w_r1(x) + self.w_r2(z_spatial) + (self.lambda_fog * h_ratio))
        p5_feat = (1.0 - g_xsa) * x + g_xsa * z_spatial
        
        return p5_feat, saliency, h_s


class C2f_BQSA_P4(nn.Module):
    """
    P4 Reindex Block: Dynamic block partitioning supporting profiler and variable resolutions.
    """
    def __init__(self, c1, c2, n=1, block_size=4, top_k=25):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.block_size = block_size
        self.top_k = top_k
        self.tau = 1.0
        
        self.cv1 = BaseConv(c1, c2, 1, 1)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, False, 1.0) for _ in range(n))
        
        self.simam = SimAM()
        self.ca = CoordinateAttention(self.c)
        self.mbconv = MBConvConditioner(self.c)
        
        self.indexer = nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c),
            nn.Conv2d(self.c, 1, 1)
        )
        self.qkv = nn.Conv2d(self.c, self.c * 3, 1)
        self.proj = nn.Conv2d(self.c, self.c, 1)
        self.gr_gate = nn.Parameter(torch.zeros(1, self.c, 1, 1))

    def set_tau(self, tau):
        self.tau = tau

    def _rasterize_gt_blocks(self, targets, batch_size, num_h, num_w, device):
        """
        Vectorized Ground-Truth Block Target Construction with Guaranteed Minimum 1-Block Law.
        YOLOX targets shape: [B, max_labels, 5] (col 0: cls, col 1: xc, col 2: yc, col 3: w, col 4: h)
        """
        target_mask = torch.zeros(batch_size, num_h, num_w, device=device)
        if targets is None or targets.shape[0] == 0:
            return target_mask.view(batch_size, -1)

        # Iterate over each image in the batch
        for b in range(batch_size):
            img_labels = targets[b]  # [max_labels, 5]
            
            # Filter valid boxes where width and height are positive
            valid_mask = (img_labels[:, 3] > 0) & (img_labels[:, 4] > 0)
            if not valid_mask.any():
                continue
                
            valid_boxes = img_labels[valid_mask]  # [num_gt, 5]
            xc = valid_boxes[:, 1]
            yc = valid_boxes[:, 2]
            w  = valid_boxes[:, 3]
            h  = valid_boxes[:, 4]

            # Convert cxcywh to xyxy
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            # Canvas validity filter: ignore annotations outside the 640x640 frame
            on_screen = (x2 > 0) & (y2 > 0) & (x1 < 640) & (y1 < 640)
            if not on_screen.any():
                continue

            x1, y1, x2, y2 = x1[on_screen], y1[on_screen], x2[on_screen], y2[on_screen]

            # Half-open interval clamping with guaranteed minimum 1-block allocation
            c_min = torch.clamp((x1 / 64.0).floor().long(), 0, num_w - 1)
            c_max = torch.clamp(((x2 - 1e-4) / 64.0).floor().long(), 0, num_w - 1)
            c_max = torch.max(c_max, c_min)

            r_min = torch.clamp((y1 / 64.0).floor().long(), 0, num_h - 1)
            r_max = torch.clamp(((y2 - 1e-4) / 64.0).floor().long(), 0, num_h - 1)
            r_max = torch.max(r_max, r_min)

            for r1, r2, c1, c2 in zip(r_min, r_max, c_min, c_max):
                target_mask[b, r1:r2 + 1, c1:c2 + 1] = 1.0

        return target_mask.view(batch_size, -1)

    def forward(self, x, p5_saliency, h_s, targets=None):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        feat = y[-1]
        B, C, H, W = feat.shape
        
        # Dynamic block dimensions (supports profiler dummy inputs)
        bs = min(self.block_size, H, W)
        num_h = max(1, H // bs)
        num_w = max(1, W // bs)
        total_blocks = num_h * num_w
        k_actual = min(self.top_k, total_blocks)
        
        # 1. Tribrid Conditioning Stack
        feat_cond = self.mbconv(self.ca(self.simam(feat)))
        
        # 2. Dynamic Pool Gating
        s_up = F.interpolate(p5_saliency, size=(H, W), mode='nearest')
        block_saliency = F.avg_pool2d(s_up, kernel_size=bs, stride=bs).view(B, -1)
        
        mu_s = block_saliency.mean(dim=-1, keepdim=True)
        sigma_s = block_saliency.std(dim=-1, keepdim=True)
        h_ratio = torch.clamp(h_s / math.log(max(2, total_blocks)), 0.0, 1.0)
        tau_pool = torch.clamp(mu_s - 3.0 * h_ratio * sigma_s, min=0.0)
        pool_mask = (block_saliency >= tau_pool).float()
        
        # 3. Block Scoring & STE Gumbel-Softmax Top-K
        scores = self.indexer(feat_cond)
        block_scores = F.avg_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        masked_scores = block_scores * pool_mask - (1.0 - pool_mask) * 1e4
        
        if self.training:
            gumbel = -torch.empty_like(masked_scores).exponential_().log()
            soft_scores = F.softmax((masked_scores + gumbel) / self.tau, dim=-1)
            _, topk_idx = torch.topk(soft_scores, k_actual, dim=-1)
            
            hard_mask = torch.zeros_like(soft_scores).scatter_(-1, topk_idx, 1.0)
            ste_mask = (hard_mask - soft_scores).detach() + soft_scores
            ste_weights = torch.gather(ste_mask, -1, topk_idx).unsqueeze(-1).unsqueeze(-1)
        else:
            _, topk_idx = torch.topk(masked_scores, k_actual, dim=-1)
            ste_weights = 1.0

        # 4. Sparse Cross-Attention
        qkv = self.qkv(feat_cond).view(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        tokens_per_b = bs * bs
        k_b = k.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        v_b = v.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        
        idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, tokens_per_b)
        k_sel = torch.gather(k_b, 1, idx_exp).permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)
        v_sel = torch.gather(v_b, 1, idx_exp) * ste_weights
        v_sel = v_sel.permute(0, 2, 1, 3).reshape(B, C, k_actual * tokens_per_b)
        
        q_flat = q.reshape(B, C, H * W)
        attn = torch.matmul(q_flat.transpose(-2, -1), k_sel) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(v_sel, attn.transpose(-2, -1)).reshape(B, C, H, W)
        
        gate = torch.sigmoid(self.gr_gate)
        y[-1] = y[-1] + gate * self.proj(context)
        p4_out = self.cv2(torch.cat(y, 1))

        # 5. Auxiliary Focal Loss
        indexer_loss = torch.tensor(0.0, device=x.device)
        if self.training and targets is not None:
            f_target = self._rasterize_gt_blocks(targets, B, num_h, num_w, x.device)
            p_b = torch.sigmoid(block_scores)
            p_t = p_b * f_target + (1.0 - p_b) * (1.0 - f_target)
            alpha_t = 0.85 * f_target + 0.15 * (1.0 - f_target)
            focal_weight = alpha_t * (1.0 - p_t).pow(2.0)
            bce = F.binary_cross_entropy_with_logits(block_scores, f_target, reduction="none")
            indexer_loss = (focal_weight * bce).mean()

        return p4_out, topk_idx, indexer_loss


class C2f_BQSA_P3(nn.Module):
    """
    P3 Reuse Block: Resolution-Adaptive Block-Local Self-Attention with TensorRT-Safe Rank-3 Scatter.
    """
    def __init__(self, c1, c2, n=1, block_size=8, top_k=25):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.block_size = block_size
        self.top_k = top_k
        
        self.cv1 = BaseConv(c1, c2, 1, 1)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, False, 1.0) for _ in range(n))
        
        self.simam = SimAM()
        self.ca = CoordinateAttention(self.c)
        self.mbconv = MBConvConditioner(self.c)
        
        self.qkv = nn.Conv2d(self.c, self.c * 3, 1)
        self.proj = nn.Conv2d(self.c, self.c, 1)
        self.gr_gate = nn.Parameter(torch.zeros(1, self.c, 1, 1))

    def forward(self, x, inherited_topk_idx):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        feat = y[-1]
        B, C, H, W = feat.shape
        
        feat_cond = self.mbconv(self.ca(self.simam(feat)))
        
        bs = min(self.block_size, H, W)
        num_h = max(1, H // bs)
        num_w = max(1, W // bs)
        total_blocks = num_h * num_w
        tokens_per_b = bs * bs
        k_actual = min(self.top_k, inherited_topk_idx.shape[1], total_blocks)
        
        qkv = self.qkv(feat_cond).view(B, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        q_b = q.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        k_b = k.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        v_b = v.view(B, C, num_h, bs, num_w, bs).permute(0, 2, 4, 1, 3, 5).reshape(B, total_blocks, C, tokens_per_b)
        
        # Clamp inherited indices to valid total_blocks range for profiler safety
        valid_idx = torch.clamp(inherited_topk_idx[:, :k_actual], 0, total_blocks - 1)
        idx_exp = valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, tokens_per_b)
        
        q_sel = torch.gather(q_b, 1, idx_exp)
        k_sel = torch.gather(k_b, 1, idx_exp)
        v_sel = torch.gather(v_b, 1, idx_exp)
        
        # In-Block Local Self-Attention
        q_loc = q_sel.permute(0, 1, 3, 2).reshape(B * k_actual, tokens_per_b, C)
        k_loc = k_sel.reshape(B * k_actual, C, tokens_per_b)
        v_loc = v_sel.permute(0, 1, 3, 2).reshape(B * k_actual, tokens_per_b, C)
        
        attn = torch.bmm(q_loc, k_loc) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        ctx_loc = torch.bmm(attn, v_loc).reshape(B, k_actual, tokens_per_b, C)
        
        # Rank-3 TensorRT-Safe ScatterElements
        out_flat = torch.zeros(B, total_blocks, C * tokens_per_b, device=x.device, dtype=q.dtype)
        idx_flat = valid_idx.unsqueeze(-1).expand(-1, -1, C * tokens_per_b)
        ctx_flat = ctx_loc.permute(0, 1, 3, 2).reshape(B, k_actual, C * tokens_per_b)
        out_flat.scatter_(1, idx_flat, ctx_flat)
        
        context = out_flat.reshape(B, num_h, num_w, C, bs, bs).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        gate = torch.sigmoid(self.gr_gate)
        y[-1] = y[-1] + gate * self.proj(context)
        return self.cv2(torch.cat(y, 1))