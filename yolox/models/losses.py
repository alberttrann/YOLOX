#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Adaptive NWD & Tuned Prototype Margin Repulsion

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class AdaptiveNWDloss(nn.Module):
    """
    Scale-Adaptive Normalized Gaussian Wasserstein Distance (NWD) Loss.
    Modeled for diffuse optical boundary blur under atmospheric scattering.
    """
    def __init__(self, kappa=2.0, reduction="none", eps=1e-5):
        super(AdaptiveNWDloss, self).__init__()
        self.kappa = kappa
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        # Execute in FP32 to eliminate half-precision underflow in distance calculation
        p_x, p_y, p_w, p_h = pred[:, 0].float(), pred[:, 1].float(), pred[:, 2].float(), pred[:, 3].float()
        g_x, g_y, g_w, g_h = target[:, 0].float(), target[:, 1].float(), target[:, 2].float(), target[:, 3].float()

        # 1. 2D Gaussian Wasserstein Distance
        center_dist_sq = (p_x - g_x).pow(2) + (p_y - g_y).pow(2)
        scale_dist_sq = ((p_w - g_w).pow(2) + (p_h - g_h).pow(2)) / 4.0
        w2_sq = center_dist_sq + scale_dist_sq + 1e-7

        # 2. Scale-Adaptive Diagonal Normalization
        c_adaptive = torch.sqrt(g_w.pow(2) + g_h.pow(2) + 1e-7) + self.eps

        # 3. Dimensionless Normalized Wasserstein Distance
        nwd = torch.exp(-self.kappa * (torch.sqrt(w2_sq) / c_adaptive))
        loss = 1.0 - nwd

        loss = loss.to(pred.dtype)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PrototypeRepulsionLoss(nn.Module):
    """
    Closed-Form Hyperspherical Prototype Margin Repulsion Loss.
    Calibration 4: m_intra calibrated to 0.10 (~84.3 deg) to prevent over-repelling
    valid sub-class modes (front vs side of cars), with strict m_inter = 0.05.
    """
    def __init__(self, num_classes=9, num_modes=64, m_intra=0.10, m_inter=0.05, eps=1e-5):
        super(PrototypeRepulsionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_modes = num_modes
        self.m_intra = m_intra  # Calibrated gentle intra-class dispersion
        self.m_inter = m_inter  # Strict inter-class categorical orthogonality
        self.eps = eps

    def forward(self, prototypes):
        if prototypes.dim() == 2:
            prototypes = prototypes.view(self.num_classes, self.num_modes, -1)

        C, K, D = prototypes.shape

        # Enforce unit hypersphere normalization strictly in FP32
        p_norm = F.normalize(prototypes.float(), p=2, dim=-1, eps=self.eps)

        # 1. Intra-Class Sub-Class Dispersion
        # [C, K, D] @ [C, D, K] -> [C, K, K]
        intra_sim = torch.bmm(p_norm, p_norm.transpose(1, 2))
        eye_mask = torch.eye(K, device=prototypes.device, dtype=torch.bool).unsqueeze(0)
        intra_sim_non_diag = intra_sim.masked_fill(eye_mask, -1.0)
        
        intra_violations = torch.clamp(intra_sim_non_diag - self.m_intra, min=0.0)
        l_intra = intra_violations.pow(2).sum() / max(1.0, float(C * K * (K - 1)))

        # 2. Inter-Class Categorical Orthogonality
        flat_p = p_norm.view(C * K, D)
        inter_sim = torch.matmul(flat_p, flat_p.t())

        class_ids = torch.arange(C, device=prototypes.device).repeat_interleave(K)
        diff_class_mask = class_ids.unsqueeze(0) != class_ids.unsqueeze(1)

        inter_violations = torch.clamp(inter_sim[diff_class_mask] - self.m_inter, min=0.0)
        l_inter = inter_violations.pow(2).sum() / max(1.0, float(C * (C - 1) * K * K))

        return (l_intra + l_inter).to(prototypes.dtype)