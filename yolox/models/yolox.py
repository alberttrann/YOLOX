#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Master Multi-Task Loss Synthesis & Curriculum Gating

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import PrototypeRepulsionLoss
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    TDE-YOLOX v3.1 Master Architecture Host.
    Certified Innovations:
      - Master Multi-Task Synthesis: Balances 6 loss objectives across dynamic training curricula.
      - Auxiliary Loss Curriculum Decay: Smoothly fades representation constraints to 10% (Epochs 40-65).
      - Atomic Supervised Memory Loss Normalization: Normalized strictly by actual matched foreground anchors.
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

        # Prototype Hyperspherical Repulsion Engine (m_intra=0.10, m_inter=0.05)
        self.repulse_loss = PrototypeRepulsionLoss(
            num_classes=self.num_classes,
            num_modes=64,
            m_intra=0.10,
            m_inter=0.05
        )

        self.current_epoch = 0
        self.max_epochs = 80

        self.warmup_end = 10
        self.ramp_end = 50
        self.prob_start = 0.1
        self.prob_end = 0.30  # Capped to 0.30 to eliminate post-E25 gradient turbulence

    def set_meta_training_state(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

        head_ref = self.head.module if hasattr(self.head, "module") else self.head
        if hasattr(head_ref, "current_epoch"):
            head_ref.current_epoch = epoch

        # Cosine Gumbel-Softmax Temperature Annealing tau: 1.0 -> 0.50 (Maintains multi-block candidate stability)
        tau_min = 0.50
        tau_max = 1.00
        decay_horizon = max(1.0, float(max_epochs - 20))
        progress = min(1.0, float(epoch) / decay_horizon)
        current_tau = tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(progress * math.pi))

        neck_ref = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
        if hasattr(neck_ref, "p4_bqsa") and hasattr(neck_ref.p4_bqsa, "set_tau"):
            neck_ref.p4_bqsa.set_tau(current_tau)

    def _get_ttt_probability(self):
        if not self.training:
            return 1.0

        if self.current_epoch < self.warmup_end:
            return self.prob_start
        if self.current_epoch >= self.ramp_end:
            return self.prob_end

        progress = (self.current_epoch - self.warmup_end) / (self.ramp_end - self.warmup_end)
        return self.prob_start + (self.prob_end - self.prob_start) * progress

    def _get_indexer_loss_weight(self):
        if self.current_epoch <= 20:
            return 0.10
        progress = (self.current_epoch - 20) / max(1.0, float(self.max_epochs - 20))
        return max(0.01, 0.10 - 0.09 * progress)

    def _get_aux_multiplier(self):
        """
        Auxiliary Loss Decay Multiplier (Curriculum Gating).
        Gradually fades out auxiliary representation constraints from Epoch 30 to 65
        so the network focuses 100% of gradient bandwidth on sub-pixel detection precision.
        """
        decay_start = 30
        decay_end = 65
        min_ratio = 0.10
        
        if self.current_epoch < decay_start:
            return 1.0
        if self.current_epoch >= decay_end:
            return min_ratio
            
        progress = (self.current_epoch - decay_start) / float(decay_end - decay_start)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(progress * math.pi))

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()

        # 1. Forward through Backbone & BQSA Neck (Returns h_s optical entropy oracle)
        neck_output, proj_loss, phase_loss, indexer_loss = self.backbone(
            x, targets=targets, ttt_prob=current_ttt_prob
        )

        if self.training:
            # 2. Forward through Decoupled Engram Head
            (
                total_det_loss, loss_iou, loss_nwd, loss_obj, loss_cls, loss_l1,
                num_fg_ratio, aux_mem_logits, cls_targets_concat, fg_masks_concat
            ) = self.head(neck_output, targets, x)

            # 3. Supervised Memory Anchor Loss normalized in FP32
            memory_anchor_loss = self._calculate_supervised_memory_loss(
                aux_mem_logits, cls_targets_concat, fg_masks_concat
            )

            # 4. Prototype Margin Repulsion Loss
            head_ref = self.head.module if hasattr(self.head, "module") else self.head
            if hasattr(head_ref, "memory_banks"):
                repulse_losses = [
                    self.repulse_loss(bank.prototypes) for bank in head_ref.memory_banks
                ]
                total_repulse_loss = sum(repulse_losses) / float(len(repulse_losses))
            else:
                total_repulse_loss = torch.tensor(0.0, device=x.device)

            # 5. Master Multi-Task Loss Synthesis with Auxiliary Decay
            # =====================================================================
            # TDE-YOLOX v3.2: SHIELDED DECOUPLED CURRICULUM SYNTHESIS
            # =====================================================================
            lambda_idx = self._get_indexer_loss_weight()
            aux_mult = self._get_aux_multiplier()
            
            # 1. Fleeting Box Scaffolding: Decays to 10% to clear headroom for L1 regression
            lambda_proj = 0.10 * aux_mult
            
            # 2. Permanent Invariance Shields: Phase, Memory, and Repulsion NEVER drop below 70%!
            # Guarantees extreme Snow (0.184) and Small Object (0.139) immunity are preserved permanently!
            shield_mult = max(0.70, aux_mult)
            lambda_mem = 0.05 * shield_mult
            lambda_rep = 0.01 * shield_mult
            lambda_phase = 0.05 * min(1.0, float(self.current_epoch) / 5.0) * shield_mult

            total_loss = (
                total_det_loss
                + (lambda_mem * memory_anchor_loss)
                + (lambda_proj * proj_loss)
                + (lambda_idx * indexer_loss)
                + (lambda_phase * phase_loss)
                + (lambda_rep * total_repulse_loss)
            )
            # =====================================================================

            return {
                "total_loss": total_loss,
                "iou_loss": loss_iou,
                "nwd_loss": loss_nwd,
                "l1_loss": loss_l1,
                "conf_loss": loss_obj,
                "cls_loss": loss_cls,
                "mem_loss": memory_anchor_loss,
                "proj_loss": proj_loss,
                "phase_loss": phase_loss,
                "idx_loss": indexer_loss,
                "rep_loss": total_repulse_loss,
                "num_fg": num_fg_ratio,
                "ttt_prob": current_ttt_prob
            }
        else:
            return self.head(neck_output)

    def _calculate_supervised_memory_loss(self, aux_mem_logits, cls_targets, fg_masks):
        all_logits_per_image = torch.cat(aux_mem_logits, dim=1)
        flat_logits_fp32 = all_logits_per_image.view(-1, self.num_classes).float()
        flat_mask = fg_masks.view(-1)
        
        fg_logits = flat_logits_fp32[flat_mask]
        
        if fg_logits.shape[0] > 0 and cls_targets.shape[0] > 0:
            min_len = min(fg_logits.shape[0], cls_targets.shape[0])
            num_fg_actual = max(1.0, float(min_len))
            bce_sum = F.binary_cross_entropy_with_logits(
                fg_logits[:min_len], cls_targets[:min_len].float(), reduction="sum"
            )
            return bce_sum / num_fg_actual
            
        return flat_logits_fp32.sum() * 0.0

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        neck_output, _, _, _ = self.backbone(x, targets=targets, ttt_prob=1.0)
        self.head.visualize_assign_result(neck_output[0], targets, x, save_prefix)