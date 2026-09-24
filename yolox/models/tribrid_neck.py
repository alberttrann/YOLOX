#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.2: HySparse2-Augmented BQSA Neck & Symmetrical Scale-AttnRes

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import SimAM, CoordinateAttention, MBConvConditioner


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


class ScaleAttnRes(nn.Module):
    def __init__(self, dim, num_sources=3, spatial_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.num_sources = num_sources
        self.scale = dim ** -0.5
        self.w = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim, eps=1e-5)

    def forward(self, sources):
        V = torch.stack(sources, dim=0)
        K = torch.stack([self.norm(s) for s in sources], dim=0)

        w_safe = torch.clamp(self.w, min=-5.0, max=5.0)
        logits = torch.einsum('c, m b c h w -> m b h w', w_safe, K) * self.scale
        # Softmax computed strictly in FP32
        weights = F.softmax(logits.float(), dim=0).to(V.dtype)

        return torch.sum(weights.unsqueeze(2) * V, dim=0)


class RotaryEmbedding2D(nn.Module):
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
        q = q.contiguous()
        k = k.contiguous()
        device = q.device
        dtype = q.dtype
        half = q.shape[-1] // 2

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

        out_q = torch.cat([qx, qy], dim=-1).contiguous()
        out_k = torch.cat([kx, ky], dim=-1).contiguous()
        return out_q, out_k


class P5ExclusiveDenseAttention(nn.Module):
    def __init__(self, dim, num_heads=4, lambda_fog=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.lambda_fog = lambda_fog

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.rope = RotaryEmbedding2D(self.head_dim)

        self.w_r1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.w_r2 = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.normal_(self.w_r1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.w_r2.weight, mean=0.0, std=0.01)
        if self.w_r1.bias is not None:
            nn.init.zeros_(self.w_r1.bias)
        if self.w_r2.bias is not None:
            nn.init.zeros_(self.w_r2.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k, H, W)

        # 1. QK Matmul in FP32
        q_fp32 = q.float()
        k_fp32 = k.float()
        attn_fp32 = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * self.scale
        attn_fp32 = F.softmax(attn_fp32, dim=-1)

        # 2. Saliency Extraction in FP32 with Safe Variance
        diag_elements = attn_fp32.diagonal(dim1=-2, dim2=-1)
        saliency_sum = attn_fp32.sum(dim=-2) - diag_elements
        saliency = saliency_sum.mean(dim=1).reshape(B, 1, H, W) / max(1.0, float(N - 1))

        s_flat = saliency.flatten(1).float()
        s_mean = s_flat.mean(dim=-1, keepdim=True)
        s_var = torch.clamp(torch.var(s_flat, dim=-1, keepdim=True, unbiased=False), min=1e-8)
        s_std = torch.sqrt(s_var) + 1e-5
        s_standardized = torch.clamp((s_flat - s_mean) / s_std, min=-20.0, max=20.0)

        s_prob = F.softmax(s_standardized, dim=-1)
        s_log_prob = F.log_softmax(s_standardized, dim=-1)
        h_s_raw = (-torch.sum(s_prob * s_log_prob, dim=-1).mean()).detach()

        h_s_fallback = math.log(max(2, N))
        h_s = torch.nan_to_num(h_s_raw, nan=h_s_fallback, posinf=h_s_fallback, neginf=0.0)

        # 3. Value Aggregation strictly in FP32
        y_fp32 = torch.matmul(attn_fp32, v.float())
        v_fp32 = v.float()
        v_norm_sq = (v_fp32 * v_fp32).sum(dim=-1, keepdim=True) + 1e-5
        proj_scalar = (y_fp32 * v_fp32).sum(dim=-1, keepdim=True) / v_norm_sq
        z = (y_fp32 - proj_scalar * v_fp32).to(v.dtype)
        z_spatial = self.proj(z.permute(0, 1, 3, 2).reshape(B, C, H, W))

        # 4. Gated Residual Highway
        h_ratio = torch.clamp(h_s / math.log(max(2, N)), 0.0, 1.0)
        g_xsa = torch.sigmoid(self.w_r1(x) + self.w_r2(z_spatial) + (self.lambda_fog * h_ratio))
        p5_feat = x + g_xsa * z_spatial

        return p5_feat, saliency, h_s


class C2f_BQSA_P4(nn.Module):
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
        target_mask = torch.zeros(batch_size, num_h, num_w, device=device)
        if targets is None or targets.shape[0] == 0:
            return target_mask.view(batch_size, -1)

        for b in range(batch_size):
            img_labels = targets[b]
            valid_mask = (img_labels[:, 3] > 0) & (img_labels[:, 4] > 0)
            if not valid_mask.any():
                continue

            valid_boxes = img_labels[valid_mask]
            xc = valid_boxes[:, 1]
            yc = valid_boxes[:, 2]
            w  = valid_boxes[:, 3]
            h  = valid_boxes[:, 4]

            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0

            on_screen = (x2 > 0) & (y2 > 0) & (x1 < 640) & (y1 < 640)
            if not on_screen.any():
                continue

            x1, y1, x2, y2 = x1[on_screen], y1[on_screen], x2[on_screen], y2[on_screen]

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

        bs = min(self.block_size, H, W)
        num_h = max(1, H // bs)
        num_w = max(1, W // bs)
        total_blocks = num_h * num_w
        k_actual = min(self.top_k, total_blocks)

        # 1. Tribrid Conditioning Stack
        feat_cond = self.mbconv(self.ca(self.simam(feat)))

        # 2. Dynamic Pool Gating with Safe Variance
        s_up = F.interpolate(p5_saliency.float(), size=(H, W), mode='nearest')
        block_saliency = F.avg_pool2d(s_up, kernel_size=bs, stride=bs).view(B, -1)

        mu_s = block_saliency.mean(dim=-1, keepdim=True)
        var_s = torch.clamp(torch.var(block_saliency, dim=-1, keepdim=True, unbiased=False), min=1e-8)
        sigma_s = torch.sqrt(var_s) + 1e-5
        
        h_ratio = torch.clamp(h_s.float() / math.log(max(2, total_blocks)), 0.0, 1.0)
        tau_pool = torch.clamp(mu_s - 3.0 * h_ratio * sigma_s, min=0.0)
        pool_mask = (block_saliency >= tau_pool).to(feat.dtype)

        # Guaranteed Minimum 1-Block Law fallback
        empty_mask = (pool_mask.sum(dim=-1, keepdim=True) == 0)
        if empty_mask.any():
            top1_fallback = torch.zeros_like(pool_mask).scatter_(-1, block_saliency.argmax(dim=-1, keepdim=True), 1.0)
            pool_mask = torch.where(empty_mask, top1_fallback, pool_mask)

        # 3. HYSPARSE2-AUGMENTED BLOCK SCORING & STE GUMBEL-SOFTMAX
        scores = self.indexer(feat_cond)
        
        # Hybrid Max-Avg Saliency Pooling: Eliminates the 1/16 small object dilution bottleneck
        scores_avg = F.avg_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        scores_max = F.max_pool2d(scores, kernel_size=bs, stride=bs).view(B, -1)
        block_scores = 0.5 * scores_avg + 0.5 * scores_max

        # Guarded Masked Scores with -500.0 Floor
        masked_scores_fp32 = block_scores.float() * pool_mask.float() - (1.0 - pool_mask.float()) * 500.0

        if self.training:
            u = torch.rand_like(masked_scores_fp32, dtype=torch.float32)
            gumbel = -torch.log(-torch.log(u + 1e-7) + 1e-7)

            # HySparse2 Optical Entropy-Sharpened Temperature (Non-blocking GPU tensor)
            tau_adaptive = torch.clamp(self.tau * (1.0 - 0.35 * h_ratio), min=0.10)

            soft_scores = F.softmax(
                (masked_scores_fp32 + gumbel) / tau_adaptive, dim=-1
            ).to(feat.dtype)

            _, topk_idx = torch.topk(soft_scores, k_actual, dim=-1)

            hard_mask = torch.zeros_like(soft_scores).scatter_(-1, topk_idx, 1.0)
            ste_mask = (hard_mask - soft_scores).detach() + soft_scores
            ste_weights = torch.gather(ste_mask, -1, topk_idx).unsqueeze(-1).unsqueeze(-1)
        else:
            _, topk_idx = torch.topk(masked_scores_fp32, k_actual, dim=-1)
            ste_weights = 1.0

        # 4. Sparse Cross-Attention strictly in FP32
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

        # Full FP32 QK and Value aggregation
        q_fp32 = q_flat.transpose(-2, -1).float()
        k_fp32 = k_sel.float()
        attn_fp32 = torch.matmul(q_fp32, k_fp32) * (C ** -0.5)
        attn_fp32 = F.softmax(attn_fp32, dim=-1)

        context = torch.matmul(
            v_sel.float(), attn_fp32.transpose(-2, -1)
        ).to(v_sel.dtype).reshape(B, C, H, W)

        gate = torch.sigmoid(self.gr_gate)
        y[-1] = y[-1] + gate * self.proj(context)
        p4_out = self.cv2(torch.cat(y, 1))

        # 5. Auxiliary Focal Loss evaluated in FP32
        indexer_loss = torch.tensor(0.0, device=x.device)
        if self.training and targets is not None:
            f_target = self._rasterize_gt_blocks(targets, B, num_h, num_w, x.device)
            p_b = torch.sigmoid(block_scores.float())
            p_t = p_b * f_target.float() + (1.0 - p_b) * (1.0 - f_target.float())
            alpha_t = 0.85 * f_target.float() + 0.15 * (1.0 - f_target.float())
            focal_weight = (alpha_t * (1.0 - p_t).pow(2.0)).detach()
            bce = F.binary_cross_entropy_with_logits(block_scores.float(), f_target.float(), reduction="none")
            indexer_loss = (focal_weight * bce).mean()

        return p4_out, topk_idx, indexer_loss


class C2f_BQSA_P3(nn.Module):
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

        valid_idx = torch.clamp(inherited_topk_idx[:, :k_actual], 0, total_blocks - 1)
        idx_exp = valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, tokens_per_b)

        q_sel = torch.gather(q_b, 1, idx_exp)
        k_sel = torch.gather(k_b, 1, idx_exp)
        v_sel = torch.gather(v_b, 1, idx_exp)

        q_loc = q_sel.permute(0, 1, 3, 2).reshape(B * k_actual, tokens_per_b, C)
        k_loc = k_sel.reshape(B * k_actual, C, tokens_per_b)
        v_loc = v_sel.permute(0, 1, 3, 2).reshape(B * k_actual, tokens_per_b, C)

        # Full FP32 QK and Value aggregation
        q_fp32 = q_loc.float()
        k_fp32 = k_loc.float()
        attn_fp32 = torch.bmm(q_fp32, k_fp32) * (C ** -0.5)
        attn_fp32 = F.softmax(attn_fp32, dim=-1)

        ctx_loc = torch.bmm(
            attn_fp32, v_loc.float()
        ).to(v_loc.dtype).reshape(B, k_actual, tokens_per_b, C)

        out_flat = torch.zeros(B, total_blocks, C * tokens_per_b, device=x.device, dtype=q.dtype)
        idx_flat = valid_idx.unsqueeze(-1).expand(-1, -1, C * tokens_per_b)
        ctx_flat = ctx_loc.permute(0, 1, 3, 2).reshape(B, k_actual, C * tokens_per_b)
        out_flat.scatter_(1, idx_flat, ctx_flat)

        context = out_flat.reshape(B, num_h, num_w, C, bs, bs).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        gate = torch.sigmoid(self.gr_gate)
        y[-1] = y[-1] + gate * self.proj(context)
        return self.cv2(torch.cat(y, 1))


class YOLOPAFPN(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        ttt_lr=0.05,
        ttt_noise_std=0.08
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels

        from .darknet import CSPDarknet
        self.backbone = CSPDarknet(
            depth, width, out_features=in_features,
            depthwise=depthwise, act=act,
            ttt_lr=ttt_lr, ttt_noise_std=ttt_noise_std
        )

        c3, c4, c5 = [int(x * width) for x in in_channels]
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_c5 = BaseConv(c5, c4, 1, 1, act=act)
        self.p5_dense_attn = P5ExclusiveDenseAttention(c5)
        self.p4_bqsa = C2f_BQSA_P4(c4 + c4, c4, n=round(3 * depth))
        self.reduce_p4 = BaseConv(c4, c3, 1, 1, act=act)
        self.p3_bqsa = C2f_BQSA_P3(c3 + c3, c3, n=round(3 * depth))

        self.down_n3 = BaseConv(c3, c4, 3, 2, act=act)
        self.down_n4 = BaseConv(c4, c5, 3, 2, act=act)

        self.attnres_n3 = ScaleAttnRes(dim=c3, num_sources=2, spatial_ratio=16.0)
        self.attnres_n4 = ScaleAttnRes(dim=c4, num_sources=3, spatial_ratio=4.0)
        self.attnres_n5 = ScaleAttnRes(dim=c5, num_sources=3, spatial_ratio=1.0)

    def forward(self, input, targets=None, ttt_prob=None):
        out_features, proj_loss, phase_loss = self.backbone(input, ttt_prob=ttt_prob)
        c3 = out_features["dark3"]
        c4 = out_features["dark4"]
        c5 = out_features["dark5"]

        # 2. TOP-DOWN BQSA CASCADE
        p5_feat, p5_saliency, h_s = self.p5_dense_attn(c5)

        _p5_h, _p5_w = p5_feat.shape[-2], p5_feat.shape[-1]
        _h_s_fallback = math.log(max(2, _p5_h * _p5_w))
        h_s = torch.nan_to_num(h_s, nan=_h_s_fallback, posinf=_h_s_fallback, neginf=0.0)

        p4_in = torch.cat([self.upsample(self.lateral_c5(p5_feat)), c4], dim=1)
        p4_out, p4_indices, indexer_loss = self.p4_bqsa(p4_in, p5_saliency, h_s, targets=targets)

        p3_in = torch.cat([self.upsample(self.reduce_p4(p4_out)), c3], dim=1)
        p3_out = self.p3_bqsa(p3_in, p4_indices)

        # 3. BOTTOM-UP SCALE-ATTNRES HIGHWAY
        n3_out = self.attnres_n3([p3_out, c3])
        n4_out = self.attnres_n4([p4_out, self.down_n3(n3_out), c4])
        n5_out = self.attnres_n5([p5_feat, self.down_n4(n4_out), c5])

        return ((n3_out, n4_out, n5_out), (p3_out, p4_out, p5_feat), h_s), proj_loss, phase_loss, indexer_loss