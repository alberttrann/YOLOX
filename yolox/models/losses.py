#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.7: Combinatorial Capacity Repulsion & Directed Category Orthogonality

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    """
    Generalized Intersection over Union (GIoU) Loss.
    Executes area products strictly in FP32 to eliminate half-precision overflow.
    """
    def __init__(self, reduction="none", loss_type="giou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4).float()
        target = target.view(-1, 4).float()

        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2.0), (target[:, :2] - target[:, 2:] / 2.0)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2.0), (target[:, :2] + target[:, 2:] / 2.0)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1.0 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2.0), (target[:, :2] - target[:, 2:] / 2.0)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2.0), (target[:, :2] + target[:, 2:] / 2.0)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(min=1e-16)
            loss = 1.0 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise NotImplementedError(f"Unsupported loss_type: {self.loss_type}")

        if self.reduction == "mean":
            return loss.mean().float()
        elif self.reduction == "sum":
            return loss.sum().float()

        return loss.float()


class AdaptiveNWDloss(nn.Module):
    """
    Scale-Adaptive Normalized Gaussian Wasserstein Distance (NWD) Loss.
    Models boundary blur under atmospheric scattering with DFA-YOLO scale weighting.
    """
    def __init__(self, kappa=2.0, reduction="none", eps=1e-5):
        super(AdaptiveNWDloss, self).__init__()
        self.kappa = kappa
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        p_x, p_y, p_w, p_h = pred[:, 0].float(), pred[:, 1].float(), pred[:, 2].float(), pred[:, 3].float()
        g_x, g_y, g_w, g_h = target[:, 0].float(), target[:, 1].float(), target[:, 2].float(), target[:, 3].float()

        center_dist_sq = (p_x - g_x).pow(2) + (p_y - g_y).pow(2)
        scale_dist_sq = ((p_w - g_w).pow(2) + (p_h - g_h).pow(2)) / 4.0
        w2_sq = center_dist_sq + scale_dist_sq + 1e-7

        c_adaptive = torch.sqrt(g_w.pow(2) + g_h.pow(2) + 1e-7) + self.eps
        c_adaptive_safe = torch.clamp(c_adaptive, min=8.0)

        area_gt = torch.clamp(g_w * g_h, min=1.0)
        gamma_scale = torch.exp(-area_gt / 1000.0)

        nwd = torch.exp(-self.kappa * (torch.sqrt(w2_sq) / c_adaptive_safe))
        loss = (1.0 + gamma_scale) * (1.0 - nwd)

        if self.reduction == "mean":
            return loss.mean().float()
        elif self.reduction == "sum":
            return loss.sum().float()

        return loss.float()


class PrototypeRepulsionLoss(nn.Module):
    """
    Hyperspherical Prototype Margin Repulsion Loss with Capacity Normalization & Directed Orthogonality.
      - Intra-Class Dispersion: m_intra = 0.25 (Allows natural sub-modal variance).
      - Inter-Class Orthogonality: m_inter = 0.15 (~81.4 deg separation, Welch-compliant).
      - Directed Pairwise Separation: m_directed = 0.05 between Rider (ID 4) and Motor/Bike (IDs 6, 5).
      - Capacity Normalization: Normalizes by C * K so loss scales monotonically with collision count.
    """
    def __init__(self, num_classes=9, num_modes=64, m_intra=0.25, m_inter=0.15, eps=1e-5):
        super(PrototypeRepulsionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_modes = num_modes
        self.m_intra = float(m_intra)
        self.m_inter = float(m_inter)
        self.m_directed = 0.05  # ~87.1 deg orthogonality for co-occurring classes
        self.eps = eps

        # Pre-registered persistent GPU index masks
        class_ids = torch.arange(num_classes).repeat_interleave(num_modes)
        diff_class_mask = class_ids.unsqueeze(0) != class_ids.unsqueeze(1)
        self.register_buffer("diff_class_mask", diff_class_mask, persistent=False)

        eye_mask = torch.eye(num_modes, dtype=torch.bool).unsqueeze(0)
        self.register_buffer("eye_mask", eye_mask, persistent=False)

    def forward(self, prototypes):
        if prototypes.dim() == 2:
            prototypes = prototypes.view(self.num_classes, self.num_modes, -1)

        C, K, D = prototypes.shape
        p_norm = F.normalize(prototypes.float(), p=2, dim=-1, eps=self.eps)

        # 1. Intra-Class Dispersion (Penalizes mode collapse within same class)
        intra_sim = torch.bmm(p_norm, p_norm.transpose(1, 2))
        intra_sim_non_diag = intra_sim.masked_fill(self.eye_mask, -1.0)
        intra_violations = torch.clamp(intra_sim_non_diag - self.m_intra, min=0.0)
        # CAPACITY NORMALIZATION (Fix 10): Normalizes by total mode capacity C * K
        l_intra = intra_violations.pow(2).sum() / float(C * K)

        # 2. Inter-Class Orthogonality (Welch-compliant categorical separation)
        flat_p = p_norm.view(C * K, D)
        inter_sim = torch.matmul(flat_p, flat_p.t())
        inter_violations = torch.clamp(inter_sim[self.diff_class_mask] - self.m_inter, min=0.0)
        l_inter = inter_violations.pow(2).sum() / float(C * K * (C - 1))

        # 3. DIRECTED PAIRWISE SEPARATION (Fix 11: Rider vs. Motor & Rider vs. Bike)
        # In 0-based indexing: Person=3, Rider=4, Bike=5, Motor=6
        p_rider = p_norm[4]  # [K, D]
        p_motor = p_norm[6]  # [K, D]
        p_bike  = p_norm[5]  # [K, D]

        sim_rider_motor = torch.matmul(p_rider, p_motor.t())  # [K, K]
        sim_rider_bike  = torch.matmul(p_rider, p_bike.t())   # [K, K]

        viol_rider_motor = torch.clamp(sim_rider_motor - self.m_directed, min=0.0)
        viol_rider_bike  = torch.clamp(sim_rider_bike - self.m_directed, min=0.0)

        l_directed = (viol_rider_motor.pow(2).sum() + viol_rider_bike.pow(2).sum()) / float(K * K)

        total_rep = l_intra + l_inter + (2.0 * l_directed)
        return total_rep.to(prototypes.dtype)