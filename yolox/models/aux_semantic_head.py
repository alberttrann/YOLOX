#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.4: Training-Only Weak Semantic Affordance (T-WSA) Engine

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv


class GaussianOccupancyTargetGenerator:
    """
    Vectorized CUDA-Native Anisotropic Gaussian Heatmap Generator.
    Operates strictly in GPU VRAM from existing YOLOX bounding box tensors.
    Zero Disk I/O. Zero CPU worker stalls.
    """
    @staticmethod
    @torch.no_grad()
    def generate(targets, num_classes=9, feat_size=(80, 80), stride=8.0):
        """
        Args:
            targets: Tensor [B, 120, 5] with [class_id, xc, yc, w, h] in image space.
            num_classes: Number of foreground categories (9).
            feat_size: Spatial dimensions of the Stride 8 feature canvas (H, W).
            stride: Spatial stride ratio (8.0).
        Returns:
            heatmaps: Tensor [B, num_classes, H, W] in [0.0, 1.0].
        """
        B = targets.shape[0]
        H, W = feat_size
        device = targets.device
        
        heatmaps = torch.zeros((B, num_classes, H, W), device=device, dtype=torch.float32)
        
        # Coordinate grid for feature map
        y_coord, x_coord = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        x_grid = x_coord.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        y_grid = y_coord.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        for b in range(B):
            img_targets = targets[b]
            # Valid boxes have positive width and height
            valid_mask = (img_targets[:, 3] > 0) & (img_targets[:, 4] > 0)
            if not valid_mask.any():
                continue
            
            valid_boxes = img_targets[valid_mask]
            classes = valid_boxes[:, 0].long()
            
            # Map box coordinates to feature grid units
            xc = (valid_boxes[:, 1] / stride).view(-1, 1, 1)  # [K, 1, 1]
            yc = (valid_boxes[:, 2] / stride).view(-1, 1, 1)
            w = (valid_boxes[:, 3] / stride).view(-1, 1, 1)
            h = (valid_boxes[:, 4] / stride).view(-1, 1, 1)

            # Anisotropic 3-sigma standard deviation: 3 * sigma = radius = dimension / 2
            sig_x = torch.clamp(w / 6.0, min=0.5)
            sig_y = torch.clamp(h / 6.0, min=0.5)

            # Evaluate Anisotropic 2D Gaussian: [K, H, W]
            exp_term = -( (x_grid - xc).pow(2) / (2.0 * sig_x.pow(2)) + 
                          (y_grid - yc).pow(2) / (2.0 * sig_y.pow(2)) )
            gaussian = torch.exp(torch.clamp(exp_term, min=-20.0))

            # Scatter into target heatmaps using max-aggregation across instances
            for k in range(valid_boxes.shape[0]):
                cls_idx = classes[k]
                if 0 <= cls_idx < num_classes:
                    heatmaps[b, cls_idx] = torch.maximum(heatmaps[b, cls_idx], gaussian[k])

        return heatmaps


class GaussianFocalLoss(nn.Module):
    """
    Penalty-Reduced Gaussian Focal Loss (CornerNet / CenterNet formulation).
    Prevents over-penalizing soft boundary falloff under adverse weather blur.
    """
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities in (0, 1), shape [B, C, H, W].
            target: Ground truth Gaussian heatmaps in [0, 1], shape [B, C, H, W].
        """
        pred = torch.clamp(pred.float(), min=self.eps, max=1.0 - self.eps)
        target = target.float()

        pos_mask = target.eq(1.0)
        neg_mask = target.lt(1.0)

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
        
        # Lateral reduction for N4
        self.lateral_n4 = BaseConv(in_channels_n4, hidden_dim, 1, 1, act="silu")
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # Fusion block on Stride 8: [N3 (128) + Upsample(N4) (128)] -> 256
        self.fuse_conv = nn.Sequential(
            BaseConv(in_channels_n3 + hidden_dim, hidden_dim, 3, 1, act="silu"),
            BaseConv(hidden_dim, hidden_dim, 3, 1, act="silu")
        )
        
        # Heatmap projection head
        self.pred_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Initialize bias to prior probability (0.01) for focal stability
        prior_prob = 0.01
        bias_init = float(-math.log((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.pred_head.bias, bias_init)

    def forward(self, n3_feat, n4_feat):
        """
        Computes predicted semantic occupancy heatmaps: [B, num_classes, H_n3, W_n3].
        """
        n4_proj = self.upsample(self.lateral_n4(n4_feat))
        fused = self.fuse_conv(torch.cat([n3_feat, n4_proj], dim=1))
        logits = self.pred_head(fused)
        return torch.sigmoid(logits)