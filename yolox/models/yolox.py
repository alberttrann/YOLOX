#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    TDE-YOLOX Unified Meta-Learning Architecture.
    
    Fix 2 Applied: Includes Orthogonal Prototype Contrastive Loss (_calculate_orthogonal_prototype_loss)
    to force prototype vectors apart in latent space and eliminate Rider/Motor/Bike/Bus collisions.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.num_classes = head.num_classes

        self.current_epoch = 0
        self.max_epochs = 80
        self.warmup_epochs = 10
        self.ramp_epochs = 50
        self.prob_start = 0.10
        self.prob_end = 0.60

    def set_meta_training_state(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def _get_ttt_probability(self):
        if not self.training:
            return 1.0

        if self.current_epoch < self.warmup_epochs:
            return self.prob_start
        if self.current_epoch >= self.ramp_epochs:
            return self.prob_end

        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, (self.ramp_epochs - self.warmup_epochs)
        )
        return self.prob_start + (self.prob_end - self.prob_start) * progress

    def _calculate_orthogonal_prototype_loss(self):
        """
        FIX 2: Forces class prototypes apart in latent space to prevent collisions.
        Penalizes any pairwise cosine similarity above margin m = 0.10.
        """
        all_prototypes = []
        if hasattr(self.head, "memory_banks"):
            for bank in self.head.memory_banks:
                p_norm = F.normalize(bank.prototypes, p=2, dim=-1)
                all_prototypes.append(p_norm)

        if not all_prototypes:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        proto_loss = 0.0
        for p in all_prototypes:
            # Pairwise cosine similarity matrix: [num_classes, num_classes]
            sim_matrix = torch.matmul(p, p.t())
            eye = torch.eye(self.num_classes, device=p.device)
            off_diag_sim = sim_matrix * (1.0 - eye)
            # Quadratic margin penalty for similarity above 0.10
            penalty = F.relu(off_diag_sim - 0.10) ** 2
            proto_loss += penalty.mean()

        return proto_loss / len(all_prototypes)

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob)

        if self.training:
            assert targets is not None
            (
                det_loss,
                iou_l,
                conf_l,
                cls_l,
                l1_l,
                num_fg,
                aux_mem_logits_list,
                cls_targets,
                fg_masks,
            ) = self.head(fpn_outs, targets, x)

            # CAVEAT 14.7 RESOLUTION: Reorder auxiliary logits across pyramid levels 
            # along the spatial dimension (dim 1) inside each image FIRST.
            batch_ordered_logits = torch.cat(aux_mem_logits_list, dim=1)  # [B, Total_Anchors, num_classes]
            all_logits = batch_ordered_logits.view(-1, self.num_classes)  # [B * Total_Anchors, num_classes]

            fg_logits = all_logits[fg_masks.view(-1)]

            # Class-Balanced Memory Anchor Loss
            if fg_logits.shape[0] > 0:
                bce_per_element = F.binary_cross_entropy_with_logits(
                    fg_logits, cls_targets, reduction="none"
                )
                class_counts = cls_targets.sum(dim=0)
                class_weights = 1.0 / (class_counts + 1.0)
                class_weights = class_weights / class_weights.mean()

                mem_loss = (bce_per_element * class_weights.unsqueeze(0)).mean()
            else:
                mem_loss = all_logits.sum() * 0.0

            # FIX 2: Calculate Orthogonal Prototype Separation Loss
            proto_loss = self._calculate_orthogonal_prototype_loss()

            # Total Loss = Det + Memory Anchor + Orthogonal Separation
            total_loss = det_loss + 0.1 * mem_loss + 0.1 * proto_loss

            return {
                "total_loss": total_loss,
                "iou_loss": iou_l,
                "l1_loss": l1_l,
                "conf_loss": conf_l,
                "cls_loss": cls_l,
                "mem_loss": mem_loss,
                "proto_loss": proto_loss,
                "num_fg": num_fg,
                "ttt_prob": current_ttt_prob,
            }
        else:
            return self.head(fpn_outs)

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x, ttt_prob=1.0)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)