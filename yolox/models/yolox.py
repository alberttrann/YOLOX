#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .ttt_modules import (
    ActiveClassFocalDiceLoss,
    BoxOccupancyTargetGenerator,
    WeakSemanticAffordanceHead,
)


class YOLOX(nn.Module):
    """
    TDE-YOLOX Master Model Architecture (Surgical vNext).
    
    Surgical Fixes Applied:
      - Caveat 14.7 Resolution: Reorders auxiliary memory logits across levels along dim 1 
        (spatial anchors) inside each image FIRST, eliminating cross-image target scrambling.
      - Patch 4.1: Square-root damped class balancing (1 / sqrt(N_c + 1)) prevents Car gradient starvation.
      - Orthogonal Prototype Loss: Enforces angular separation on memory prototypes to prevent class collapse.
      - Dense Affordance Supervision: WeakSemanticAffordanceHead with dynamic spatial resolution binding.
      - Caveat 14.2 Resolution: Curriculum scheduler dynamically pulls attributes from Exp.
    """

    def __init__(self, backbone=None, head=None, afford_loss_weight=0.25):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.num_classes = head.num_classes
        self.afford_loss_weight = afford_loss_weight

        # Multi-scale channel dimensions for YOLOX-S (width=0.5: N3=128, N4=256, N5=512)
        head_width = getattr(head, "width", 1.0)
        c_n3 = int(256 * head_width)
        c_n4 = int(512 * head_width)
        c_n5 = int(1024 * head_width)

        # Training-Only Semantics-Enriched Affordance Modules (0.0 ms Test Latency)
        self.affordance_head = WeakSemanticAffordanceHead(
            in_channels_n3=c_n3,
            in_channels_n4=c_n4,
            in_channels_n5=c_n5,
            num_classes=self.num_classes,
            hidden_dim=64,
        )
        self.afford_criterion = ActiveClassFocalDiceLoss()
        self.target_generator = BoxOccupancyTargetGenerator()

        # Config-Driven Curriculum Scheduler (Caveat 14.2 Resolution)
        self.current_epoch = 0
        self.max_epochs = 80
        self.warmup_epochs = 10
        self.ramp_epochs = 50
        self.prob_start = 0.10
        self.prob_end = 0.60

    def set_meta_training_state(self, epoch, max_epochs):
        """Synchronized exactly once per epoch by Trainer.before_epoch (Fixes Caveat 14.3)."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def _get_ttt_probability(self):
        """Unified curriculum adaptation scheduler driven by experiment configuration."""
        if not self.training:
            return 1.0  # Test-time adaptation is deterministic and active

        if self.current_epoch < self.warmup_epochs:
            return self.prob_start
        if self.current_epoch >= self.ramp_epochs:
            return self.prob_end

        # Linear Curriculum Interpolation
        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, (self.ramp_epochs - self.warmup_epochs)
        )
        return self.prob_start + (self.prob_end - self.prob_start) * progress

    def _calculate_orthogonal_prototype_loss(self):
        """Enforces angular separation on memory prototypes to prevent cluster overlap."""
        all_prototypes = []
        if hasattr(self.head, "memory_banks"):
            for bank in self.head.memory_banks:
                p_norm = F.normalize(bank.prototypes, p=2, dim=-1, eps=1e-3)
                all_prototypes.append(p_norm)

        if not all_prototypes:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        proto_loss = 0.0
        for p in all_prototypes:
            sim_matrix = torch.matmul(p, p.t()).clamp(-1.0, 1.0)
            eye = torch.eye(self.num_classes, device=p.device)
            off_diag_sim = sim_matrix * (1.0 - eye)
            penalty = F.relu(off_diag_sim - 0.10) ** 2
            proto_loss += penalty.mean()

        return proto_loss / len(all_prototypes)

    def _calculate_class_balanced_memory_loss(self, aux_mem_logits_list, cls_targets, fg_masks):
        """
        CAVEAT 14.7 RESOLUTION & PATCH 4.1:
        Reorders auxiliary logits across pyramid levels along dim 1 inside each image FIRST.
        Applies square-root damped class balancing (1 / sqrt(N_c + 1)) to prevent Car starvation.
        """
        # aux_mem_logits_list contains 3 tensors: [B, H3*W3, num_classes], [B, H4*W4, C], [B, H5*W5, C]
        batch_ordered_logits = torch.cat(aux_mem_logits_list, dim=1)  # [B, Total_Anchors, num_classes]
        all_logits = batch_ordered_logits.view(-1, self.num_classes)  # [B * Total_Anchors, num_classes]

        # Extract foreground targets directly aligned with SimOTA's fg_masks
        fg_logits = all_logits[fg_masks.view(-1)]

        if fg_logits.shape[0] == 0:
            return all_logits.sum() * 0.0

        bce_per_element = F.binary_cross_entropy_with_logits(
            fg_logits, cls_targets, reduction="none"
        )

        # PATCH 4.1 APPLIED: Square-root class volume damping (Cui et al., CVPR 2019)
        class_counts = cls_targets.sum(dim=0)  # [num_classes]
        class_weights = 1.0 / torch.sqrt(class_counts + 1.0)
        class_weights = class_weights / class_weights.mean()

        mem_loss = (bce_per_element * class_weights.unsqueeze(0)).mean()
        return mem_loss

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob)

        if self.training:
            assert targets is not None
            pan_out2, pan_out1, pan_out0 = fpn_outs

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

            # 1. Class-Balanced Memory Anchor Loss (Fixing Caveats 14.5 and 14.7)
            mem_loss = self._calculate_class_balanced_memory_loss(
                aux_mem_logits_list, cls_targets, fg_masks
            )

            # 2. Orthogonal Prototype Separation Loss
            proto_loss = self._calculate_orthogonal_prototype_loss()

            # 3. Dense Weak Semantic Affordance Supervision (Training Only)
            afford_logits = self.affordance_head(pan_out2, pan_out1, pan_out0)
            target_heatmaps = self.target_generator.generate(
                targets,
                num_classes=self.num_classes,
                feat_size=pan_out2.shape[-2:],  # Dynamically resolved to N3 spatial shape
                stride=8.0,
            )
            afford_loss = self.afford_criterion(afford_logits, target_heatmaps)

            # Total Unified Multi-Task Objective
            total_loss = (
                det_loss
                + 0.10 * mem_loss
                + 0.10 * proto_loss
                + self.afford_loss_weight * afford_loss
            )

            return {
                "total_loss": total_loss,
                "iou_loss": iou_l,
                "l1_loss": l1_l,
                "conf_loss": conf_l,
                "cls_loss": cls_l,
                "mem_loss": mem_loss,
                "proto_loss": proto_loss,
                "afford_loss": afford_loss,
                "num_fg": num_fg,
                "ttt_prob": current_ttt_prob,
            }
        else:
            # During evaluation/inference: Affordance head is completely severed (0.0 ms latency)
            return self.head(fpn_outs)

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x, ttt_prob=1.0)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)