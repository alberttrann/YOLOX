#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.7: Master Three-Phase Curriculum Synthesis & Level-Normalized Memory Supervision

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import PrototypeRepulsionLoss
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .aux_semantic_head import BoxOccupancyTargetGenerator, ActiveClassFocalDiceLoss

@torch.no_grad()
def build_dam_block_targets_untruncated(fg_masks, hw_list, strides, num_h, num_w, block_px):
    """
    Untruncated Binary DAM Target Generator (Solution A).
    - Zero label noise: Every block containing a positive anchor is labeled 1.0 deterministically.
    - Scorer learns true probability P(object present | features).
    - Selection is TopK(s, k) based on learned confidence s_b.
    """
    B = fg_masks.shape[0]
    N = num_h * num_w

    expected_anchors = sum(h * w for h, w in hw_list)
    assert fg_masks.dim() == 2, f"fg_masks must be [B, A], got shape {fg_masks.shape}"
    assert fg_masks.shape[1] == expected_anchors, (
        f"fg_masks anchor count mismatch: {fg_masks.shape[1]} vs expected {expected_anchors}"
    )

    level_presences = []
    off = 0
    for (h, w), s in zip(hw_list, strides):
        n = h * w
        m = fg_masks[:, off:off + n].float().view(B, 1, h, w)
        off += n
        r = block_px // s  # 8 for P3, 4 for P4, 2 for P5

        has_pos = (F.max_pool2d(m, kernel_size=r, stride=r) > 0).float()
        level_presences.append(has_pos.flatten(1))

    # Binary presence across ANY level: shape [B, N], values strictly in {0.0, 1.0}
    target = torch.stack(level_presences, dim=1).max(dim=1).values
    return target


def dam_indexer_loss(block_scores, fg_masks, hw_list, bqsa, strides=(8, 16, 32)):
    """Computes BCE loss against the Untruncated DAM target and tracks diagnostic recall."""
    H4, W4 = hw_list[1]
    num_h, num_w = bqsa.block_grid(H4, W4)
    k = bqsa.num_selected(num_h * num_w)
    
    target = build_dam_block_targets_untruncated(fg_masks, hw_list, strides, num_h, num_w, bqsa.block_px)
    loss = F.binary_cross_entropy_with_logits(block_scores.float(), target)
    
    with torch.no_grad():
        sel = torch.zeros_like(target).scatter_(1, torch.topk(block_scores.detach(), k, dim=1).indices, 1.0)
        recall = (sel * target).sum() / target.sum().clamp(min=1.0)
        
    return loss, recall
class YOLOX(nn.Module):
    """
    TDE-YOLOX v3.7 Master Architecture Host.
    Features:
      - Three-Phase "Earn-Your-Influence" Curriculum: Auxiliary losses phase in smoothly over Epochs 4-13.
      - Level-Normalized Supervised Memory Loss: P3, P4, and P5 prototypes receive equal supervision.
      - Continuous Auxiliary Multiplier Fade: Fades smoothly to 0.15 floor between Epochs 45 and 65.
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

        

        self.sem_loss_fn = ActiveClassFocalDiceLoss()

        self.current_epoch = 0
        self.max_epochs = 80

    def set_meta_training_state(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

        head_ref = self.head.module if hasattr(self.head, "module") else self.head
        if hasattr(head_ref, "current_epoch"):
            head_ref.current_epoch = epoch
        if hasattr(head_ref, "max_epochs"):
            head_ref.max_epochs = max_epochs

        # Cosine Gumbel-Softmax Temperature Annealing: tau: 1.0 -> 0.50
        tau_min = 0.50
        tau_max = 1.00
        decay_horizon = max(1.0, float(max_epochs - 20))
        progress = min(1.0, float(epoch) / decay_horizon)
        current_tau = tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(progress * math.pi))

        neck_ref = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
        if hasattr(neck_ref, "bqsa_p4") and hasattr(neck_ref.bqsa_p4, "set_tau"):
            neck_ref.bqsa_p4.set_tau(current_tau)

    def _get_ttt_probability(self):
        return 1.0

    def _get_indexer_loss_weight(self):
        if self.current_epoch <= 20:
            return 0.10
        progress = (self.current_epoch - 20) / max(1.0, float(self.max_epochs - 20))
        return max(0.01, 0.10 - 0.09 * progress)

    def _get_aux_multiplier(self):
        # Cosine fade between Epochs 45 and 65 down to a 0.15 floor
        decay_start, decay_end, min_ratio = 45, 65, 0.15
        if self.current_epoch < decay_start:
            return 1.0
        if self.current_epoch >= decay_end:
            return min_ratio
        progress = (self.current_epoch - decay_start) / float(decay_end - decay_start)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(progress * math.pi))

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()

        # 1. Forward through Backbone & BQSA Neck
        neck_output, cma_loss, phase_loss, p4_block_scores, sem_pred = self.backbone(
            x, targets=targets, ttt_prob=current_ttt_prob,
            epoch=self.current_epoch, max_epochs=self.max_epochs
        )

        if self.training:
            # 2. Forward through Decoupled Engram Head with Unified Single Dispatch
            (
                total_det_loss, loss_iou, loss_nwd, loss_obj, loss_cls, loss_l1,
                num_fg_ratio, aux_mem_logits, cls_targets_concat, fg_masks_concat, expanded_strides
            ) = self.head(neck_output, targets, x)

            # 3. Supervised Memory Anchor Loss Level-Normalized across P3, P4, P5
            memory_anchor_loss = self._calculate_supervised_memory_loss(
                aux_mem_logits, cls_targets_concat, fg_masks_concat, expanded_strides
            )

            # 4. Canonical Prototype Orthogonal Separation on S^127 (9 Classes)
            head_ref = self.head.module if hasattr(self.head, "module") else self.head
            if hasattr(head_ref, "memory_banks"):
                rep_losses = []
                for bank in head_ref.memory_banks:
                    p = F.normalize(bank.prototypes.float(), p=2, dim=-1)
                    sim = torch.matmul(p, p.t())  # Shape [9, 9]
                    # Penalize positive dot-product overlap between different classes
                    off_diag = sim * (1.0 - torch.eye(self.num_classes, device=x.device))
                    rep_losses.append(torch.clamp(off_diag, min=0.0).pow(2).mean())
                total_repulse_loss = sum(rep_losses) / float(len(rep_losses))
            else:
                total_repulse_loss = torch.tensor(0.0, device=x.device)

            # 5. Evaluate Weak Semantic Affordance Loss
            if sem_pred is not None and targets is not None:
                feat_size = (sem_pred.shape[-2], sem_pred.shape[-1])
                target_heatmaps = BoxOccupancyTargetGenerator.generate(
                    targets, num_classes=self.num_classes, feat_size=feat_size, stride=8.0
                )
                raw_sem_loss = self.sem_loss_fn(sem_pred, target_heatmaps)
            else:
                raw_sem_loss = torch.tensor(0.0, device=x.device)

            # 6. DOWNSTREAM SIMOTA DAM INDEXER LOSS (Sparse-DETR) - PLACED HERE IN FORWARD!
            neck_ref = self.backbone.module if hasattr(self.backbone, "module") else self.backbone
            hw_list = [tuple(f.shape[-2:]) for f in neck_output[0]]
            indexer_loss, idx_recall = dam_indexer_loss(
                p4_block_scores, fg_masks_concat, hw_list, neck_ref.bqsa_p4
            )

            # -------------------------------------------------------------
            # MASTER "EARN-YOUR-INFLUENCE" CURRICULUM SYNTHESIS
            # -------------------------------------------------------------
            aux_mult = self._get_aux_multiplier()
            current_ep = float(self.current_epoch) if self.current_epoch > 0 else 1.0

            aux_ramp_10 = min(1.0, max(0.0, (current_ep - 3.0) / 10.0))
            cma_ramp = min(1.0, current_ep / 10.0)
            phase_ramp = min(1.0, current_ep / 5.0)

            lambda_sem   = 0.015 * aux_ramp_10 * aux_mult
            lambda_mem   = 0.040 * aux_ramp_10 * aux_mult
            lambda_cma   = 0.050 * cma_ramp * aux_mult
            lambda_phase = 0.030 * phase_ramp * aux_mult
            lambda_rep   = 0.030 * aux_mult
            lambda_idx   = 1.000  # Fixed weight independent of aux_mult so scorer trains continuously

            total_loss = (
                total_det_loss
                + (lambda_mem * memory_anchor_loss)
                + (lambda_cma * cma_loss)
                + (lambda_idx * indexer_loss)
                + (lambda_phase * phase_loss)
                + (lambda_rep * total_repulse_loss)
                + (lambda_sem * raw_sem_loss)
            )

            return {
                "total_loss": total_loss,
                "iou_loss": loss_iou,
                "nwd_loss": loss_nwd,
                "l1_loss": loss_l1,
                "conf_loss": loss_obj,
                "cls_loss": loss_cls,
                "mem_loss": memory_anchor_loss,
                "cma_loss": cma_loss,
                "proj_loss": cma_loss,
                "phase_loss": phase_loss,
                "idx_loss": indexer_loss,
                "idx_recall": idx_recall.detach(),
                "rep_loss": total_repulse_loss,
                "sem_loss": raw_sem_loss,
                "num_fg": num_fg_ratio,
                "ttt_prob": current_ttt_prob
            }
        else:
            return self.head(neck_output)

    def _calculate_supervised_memory_loss(self, aux_mem_logits, cls_targets, fg_masks, expanded_strides=None):
        """
        LEVEL-NORMALIZED SUPERVISED MEMORY LOSS (Fix 8):
        Supervises P3, P4, and P5 prototypes with equal task gradient strength.
        Safe against tensor/list variations with zero memory allocation.
        """
        if len(aux_mem_logits) == 0 or cls_targets.shape[0] == 0:
            return torch.tensor(0.0, device=cls_targets.device if cls_targets.is_cuda else "cpu")

        strides = [8.0, 16.0, 32.0]
        level_losses = []

        flat_mask = fg_masks.view(-1)
        flat_targets = cls_targets.float()

        if expanded_strides is not None:
            # Dimension-safe concatenation across list of stride tensors
            if isinstance(expanded_strides, (list, tuple)):
                cat_strides = torch.cat(expanded_strides, dim=1)
            else:
                cat_strides = expanded_strides.view(1, -1)

            B = aux_mem_logits[0].shape[0]
            strides_all = cat_strides.expand(B, -1).reshape(-1)
            fg_strides = strides_all[flat_mask]

            all_logits = torch.cat(aux_mem_logits, dim=1).view(-1, self.num_classes).float()
            fg_logits = all_logits[flat_mask]

            min_len = min(fg_logits.shape[0], flat_targets.shape[0], fg_strides.shape[0])
            fg_logits = fg_logits[:min_len]
            flat_targets = flat_targets[:min_len]
            fg_strides = fg_strides[:min_len]

            for s in strides:
                s_mask = (fg_strides == s)
                if s_mask.any():
                    logits_s = fg_logits[s_mask]
                    targets_s = flat_targets[s_mask]
                    bce_s = F.binary_cross_entropy_with_logits(logits_s, targets_s, reduction="sum")
                    n_fg_s = max(1.0, float(s_mask.sum().item()))
                    level_losses.append(bce_s / (n_fg_s * float(self.num_classes)))
        else:
            all_logits = torch.cat(aux_mem_logits, dim=1).view(-1, self.num_classes).float()
            fg_logits = all_logits[flat_mask]
            min_len = min(fg_logits.shape[0], flat_targets.shape[0])
            bce_sum = F.binary_cross_entropy_with_logits(
                fg_logits[:min_len], flat_targets[:min_len], reduction="sum"
            )
            return bce_sum / (max(1.0, float(min_len)) * float(self.num_classes))

        if len(level_losses) > 0:
            return sum(level_losses) / float(len(level_losses))
        return torch.tensor(0.0, device=flat_targets.device)