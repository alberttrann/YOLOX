#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
import torch
import torch.nn as nn


class IOUloss(nn.Module):
    """
    Unified Bounding Box Regression Loss with OIoU Support.
    
    Supported loss_type:
      - "iou": Standard Intersection over Union loss.
      - "giou": Generalized IoU loss (Rezatofighi et al.).
      - "oiou": Optimal Oriented IoU loss (OD-YOLO, Sensors 2024).
                Incorporates angle disparity (δ), exponential distance (ε), 
                exponential shape penalty (ζ, θ=4), and centroid variance (η).
                Eliminates vanishing IoU gradients on sub-16px targets.
    """

    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "pred and target must have matching batch dimension."

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # 1. Coordinate Decomposition: (cx, cy, w, h)
        pred_cx, pred_cy = pred[:, 0], pred[:, 1]
        pred_w, pred_h = pred[:, 2], pred[:, 3]
        target_cx, target_cy = target[:, 0], target[:, 1]
        target_w, target_h = target[:, 2], target[:, 3]

        # 2. Corner Points for Intersection and Enclosing Box
        tl_p = torch.stack([pred_cx - pred_w / 2.0, pred_cy - pred_h / 2.0], dim=1)
        br_p = torch.stack([pred_cx + pred_w / 2.0, pred_cy + pred_h / 2.0], dim=1)
        tl_g = torch.stack([target_cx - target_w / 2.0, target_cy - target_h / 2.0], dim=1)
        br_g = torch.stack([target_cx + target_w / 2.0, target_cy + target_h / 2.0], dim=1)

        # Intersection bounds
        tl_i = torch.max(tl_p, tl_g)
        br_i = torch.min(br_p, br_g)

        area_p = pred_w * pred_h
        area_g = target_w * target_h

        en = (tl_i < br_i).type(tl_p.type()).prod(dim=1)
        area_i = torch.prod(br_i - tl_i, dim=1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1.0 - iou ** 2

        elif self.loss_type == "giou":
            c_tl = torch.min(tl_p, tl_g)
            c_br = torch.max(br_p, br_g)
            area_c = torch.prod(c_br - c_tl, dim=1)
            giou = iou - (area_c - area_u) / area_c.clamp(min=1e-16)
            loss = 1.0 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "oiou":
            # --- OD-YOLO (Sensors 2024) OIoU Loss Implementation ---
            # Enclosing box dimensions (Cw, Ch)
            c_tl = torch.min(tl_p, tl_g)
            c_br = torch.max(br_p, br_g)
            c_dims = (c_br - c_tl).clamp(min=1e-8)
            cw, ch = c_dims[:, 0], c_dims[:, 1]
            c_diag_sq = cw.pow(2) + ch.pow(2) + 1e-16

            # Center Euclidean distance (gamma) and height disparity
            gamma_sq = (target_cx - pred_cx).pow(2) + (target_cy - pred_cy).pow(2)
            gamma = torch.sqrt(gamma_sq + 1e-16)
            dy = torch.abs(target_cy - pred_cy)

            # Angle Loss (delta): Measures orientation discrepancy between centers
            ratio_angle = torch.clamp(dy / (gamma + 1e-16), min=0.0, max=1.0 - 1e-7)
            alpha_angle = torch.asin(ratio_angle)
            # When gamma -> 0, delta evaluates smoothly to cos(-pi/4) = sqrt(2)/2
            delta = torch.cos(2.0 * alpha_angle - math.pi / 4.0)

            # Angle-Modulated Distance Loss (epsilon):
            mu = 2.0 - delta
            rho_x = (target_cx - pred_cx).pow(2) / cw.pow(2)
            rho_y = (target_cy - pred_cy).pow(2) / ch.pow(2)
            eps_dist = 2.0 - torch.exp(-mu * rho_x) - torch.exp(-mu * rho_y)

            # Exponential Shape Loss (zeta) with power theta = 4 (OD-YOLO):
            # Eliminates vanishing gradients on sub-16px targets
            w_denom = torch.max(pred_w, target_w).clamp(min=1e-8)
            h_denom = torch.max(pred_h, target_h).clamp(min=1e-8)
            w_w = torch.abs(pred_w - target_w) / w_denom
            w_h = torch.abs(pred_h - target_h) / h_denom
            theta = 4.0
            zeta = (1.0 - torch.exp(-w_w)).pow(theta) + (1.0 - torch.exp(-w_h)).pow(theta)

            # Centroid and Dimension Variance Loss (eta):
            var_b = gamma_sq / c_diag_sq
            var_w = (pred_w - target_w).pow(2) / (cw.pow(2) + 1e-16)
            var_h = (pred_h - target_h).pow(2) / (ch.pow(2) + 1e-16)
            eta = var_b + var_w + var_h

            # Unified OIoU Objective (Eq. 14 of OD-YOLO):
            loss = 1.0 - iou + (eps_dist + zeta + eta) / 3.0

        else:
            raise NotImplementedError(f"Unsupported loss_type: {self.loss_type}")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss