#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.4: Production-Grade Training-Only Weak Semantic Affordance

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv


class GaussianOccupancyTargetGenerator:
    """
    Vectorized CUDA-Native Anisotropic Gaussian Heatmap Generator.
    Operates strictly in GPU VRAM from existing YOLOX bounding box tensors.
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
            indexing="ij"
        )
        x_grid = x_coord.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        y_grid = y_coord.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        for b in range(B):
            img_targets = targets[b]
            valid_mask = (img_targets[:, 3] > 0) & (img_targets[:, 4] > 0)
            if not valid_mask.any():
                continue
            
            valid_boxes = img_targets[valid_mask]
            classes = valid_boxes[:, 0].long()
            
            # FRICTION FIX: Snap to integer grid cell coordinates and clamp within boundaries!
            # Guarantees (x_grid - xc)^2 == 0 at peak, producing exact 1.0 targets!
            xc = torch.clamp(torch.round(valid_boxes[:, 1] / stride), 0, W - 1).view(-1, 1, 1)
            yc = torch.clamp(torch.round(valid_boxes[:, 2] / stride), 0, H - 1).view(-1, 1, 1)
            w = (valid_boxes[:, 3] / stride).view(-1, 1, 1)
            h = (valid_boxes[:, 4] / stride).view(-1, 1, 1)

            # Minimum 0.75 sigma guarantees smooth 3-cell spatial dispersion on small objects
            sig_x = torch.clamp(w / 6.0, min=0.75)
            sig_y = torch.clamp(h / 6.0, min=0.75)

            exp_term = -( (x_grid - xc).pow(2) / (2.0 * sig_x.pow(2)) + 
                          (y_grid - yc).pow(2) / (2.0 * sig_y.pow(2)) )
            gaussian = torch.exp(torch.clamp(exp_term, min=-20.0))

            # FRICTION FIX: Vectorized Class-Batched Reduction (9 dispatches instead of 80+)
            for c in range(num_classes):
                cls_mask = (classes == c)
                if cls_mask.any():
                    heatmaps[b, c] = torch.max(gaussian[cls_mask], dim=0).values

        return heatmaps


class GaussianFocalLoss(nn.Module):
    """
    Penalty-Reduced Gaussian Focal Loss (CornerNet / CenterNet formulation).
    """
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred.float(), min=self.eps, max=1.0 - self.eps)
        target = target.float()

        # Tolerance threshold accounts for FP16 precision
        pos_mask = target.ge(0.99)
        neg_mask = target.lt(0.99)

        pos_loss = (1.0 - pred).pow(self.alpha) * torch.log(pred) * pos_mask.float()
        neg_loss = (1.0 - target).pow(self.beta) * pred.pow(self.alpha) * torch.log(1.0 - pred) * neg_mask.float()

        num_pos = pos_mask.float().sum()
        total_loss = -(pos_loss.sum() + neg_loss.sum())
        
        return total_loss / torch.clamp(num_pos, min=1.0)


class WeakSemanticAffordanceHead(nn.Module):
    """
    Lightweight Training-Only Semantic Affordance Decoder.
    Connects to Neck N3 (Stride 8) and N4 (Stride 16).
    Completely severed at inference time (0.0 ms test latency).
    """
    def __init__(self, in_channels_n3=128, in_channels_n4=256, num_classes=9, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        
        self.lateral_n4 = BaseConv(in_channels_n4, hidden_dim, 1, 1, act="silu")
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        self.fuse_conv = nn.Sequential(
            BaseConv(in_channels_n3 + hidden_dim, hidden_dim, 3, 1, act="silu"),
            BaseConv(hidden_dim, hidden_dim, 3, 1, act="silu")
        )
        
        self.pred_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)
        
        prior_prob = 0.01
        bias_init = float(-math.log((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.pred_head.bias, bias_init)

    def forward(self, n3_feat, n4_feat):
        # Dynamically match n3 spatial dimensions under any multiscale resolution
        n4_lat = self.lateral_n4(n4_feat)
        n4_proj = F.interpolate(n4_lat, size=n3_feat.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse_conv(torch.cat([n3_feat, n4_proj], dim=1))
        logits = self.pred_head(fused)
        return torch.sigmoid(logits)