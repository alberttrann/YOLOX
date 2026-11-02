#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
import torch
import torch.nn as nn


class IOUloss(nn.Module):
    """
    Unified Bounding Box Regression Loss with Hardened Quadratic OIoU Support.

    Supported loss_type:
      - "iou": Standard Intersection over Union loss.
      - "giou": Generalized IoU loss (Rezatofighi et al.).
      - "oiou": Optimal Oriented IoU loss (OD-YOLO, Sensors 2024, Refined vFinal).
                Incorporates angle disparity (delta), exponential distance (epsilon),
                quadratic smooth shape penalty (zeta, theta=2.0), and centroid variance (eta).
                Eliminates the theta=4.0 vanishing gradient dead zone while avoiding
                theta=1.0 L1 sign-flip oscillations near the optimum.
    """

    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0], "pred and target must have matching batch dimension."

        # Execute strictly in Float32 (underflow/overflow limit: 3.4e38)
        # Prevents coordinates squared (800^2 = 640,000) from overflowing FP16 (65,504) to +Inf
        with torch.cuda.amp.autocast(enabled=False):
            pred = pred.view(-1, 4).float()
            target = target.view(-1, 4).float()

            pred_cx, pred_cy = pred[:, 0], pred[:, 1]
            pred_w, pred_h = pred[:, 2], pred[:, 3]
            target_cx, target_cy = target[:, 0], target[:, 1]
            target_w, target_h = target[:, 2], target[:, 3]

            tl_p = torch.stack([pred_cx - pred_w / 2.0, pred_cy - pred_h / 2.0], dim=1)
            br_p = torch.stack([pred_cx + pred_w / 2.0, pred_cy + pred_h / 2.0], dim=1)
            tl_g = torch.stack([target_cx - target_w / 2.0, target_cy - target_h / 2.0], dim=1)
            br_g = torch.stack([target_cx + target_w / 2.0, target_cy + target_h / 2.0], dim=1)

            tl_i = torch.max(tl_p, tl_g)
            br_i = torch.min(br_p, br_g)

            area_p = torch.clamp(pred_w * pred_h, min=0.0)
            area_g = torch.clamp(target_w * target_h, min=0.0)

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
                c_tl = torch.min(tl_p, tl_g)
                c_br = torch.max(br_p, br_g)
                c_dims = (c_br - c_tl).clamp(min=1e-8)
                cw, ch = c_dims[:, 0], c_dims[:, 1]
                c_diag_sq = cw.pow(2) + ch.pow(2) + 1e-16

                gamma_sq = (target_cx - pred_cx).pow(2) + (target_cy - pred_cy).pow(2)
                gamma = torch.sqrt(gamma_sq + 1e-16)
                dy = torch.abs(target_cy - pred_cy)

                ratio_angle = torch.clamp(dy / (gamma + 1e-16), min=0.0, max=1.0 - 1e-7)
                alpha_angle = torch.asin(ratio_angle)
                delta = torch.cos(2.0 * alpha_angle - math.pi / 4.0)

                mu = 2.0 - delta
                rho_x = (target_cx - pred_cx).pow(2) / (cw.pow(2) + 1e-16)
                rho_y = (target_cy - pred_cy).pow(2) / (ch.pow(2) + 1e-16)
                eps_dist = 2.0 - torch.exp(-mu * rho_x) - torch.exp(-mu * rho_y)

                w_denom = torch.max(pred_w, target_w).clamp(min=1e-8)
                h_denom = torch.max(pred_h, target_h).clamp(min=1e-8)
                w_w = torch.abs(pred_w - target_w) / w_denom
                w_h = torch.abs(pred_h - target_h) / h_denom
                theta = 2.0
                zeta = (1.0 - torch.exp(-w_w)).pow(theta) + (1.0 - torch.exp(-w_h)).pow(theta)

                var_b = gamma_sq / c_diag_sq
                var_w = (pred_w - target_w).pow(2) / (cw.pow(2) + 1e-16)
                var_h = (pred_h - target_h).pow(2) / (ch.pow(2) + 1e-16)
                eta = var_b + var_w + var_h

                loss = 1.0 - iou + (eps_dist + zeta + eta) / 3.0

            else:
                raise NotImplementedError(f"Unsupported loss_type: {self.loss_type}")

            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()

            return loss