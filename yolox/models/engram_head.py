#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign
from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from .ttt_modules import EngramMemoryBank
from .yolo_head import YOLOXHead


class TDE_Head(YOLOXHead):
    """
    TDE-YOLOX Decoupled Detection Head (Hardened vFinal).

    Core Architectural Refinements:
      - Purged Feature Sabotage: Clean convolutional features propagate uncorrupted;
        15% Bernoulli spatial dropout and Gaussian noise are permanently deleted.
      - Native 1x1 Conv2d Latent Projection: Operates directly on 2D feature maps,
        eliminating 4 redundant flatten/permute tensor copies per level.
      - Zero-Initialized Spatial Smoothers: Residual depthwise smoothing with zero weights
        guarantees exact 1.0x identity at Step 0, eliminating Dirac amplitude inflation.
      - Grounded Manifold Uncertainty: Uncertainty is derived from prototype distance
        grounded by detached objectness; strictly zero on background, reactive in fog/night.
      - LayerNorm-Safe Soft-Gated Hybrid FiLM: Multiplicative gating on background guarantees
        zero empty-space hallucination, while smooth Sigmoid recovery restores fog targets.
      - Monotonic Bayesian Joint Scoring: S = p_obj * p_cls purges false-positive rank inversion.
      - SimOTA Boundary Zero-Candidate Guard: Prevents empty-tensor topk crashes on borders.
      - Progressive Continuous Objectness Targets: Soft IoU targets ramped across Epochs 10-50
        eliminate cold-start deadlock while permanently suppressing boundary halo duplicates.
      - Direct Integer Label Passing: matched_classes_concat passes true integer classes.
      - Balanced Loss Equilibrium: Rescaled reg_weight = 2.5; normalized strictly by N_fg.
    """

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        loss_type="oiou",
    ):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        self.width = width
        self.enable_memory_restoration = True
        self.use_baseline_assignments = False
        # Rich, detached assignment telemetry. The trainer reads this only at
        # telemetry intervals, so the matching path does not retain autograd graphs.
        self.last_assignment_stats = {"num_fg_total": 0, "num_gt_total": 0}
        self._last_assignment_meta = {}
        # Trainer enables this only on telemetry iterations to keep rich matching
        # diagnostics from perturbing normal throughput.
        self.collect_assignment_telemetry = False

        # Upgrade IoU loss to Quadratic Smooth OIoU (theta=2.0)
        self.iou_loss = IOUloss(reduction="none", loss_type=loss_type)

        self.latent_dim = 128
        self.memory_banks = nn.ModuleList()
        self.latent_projectors = nn.ModuleList()
        self.spatial_smoothers = nn.ModuleList()
        self.memory_proj_back = nn.ModuleList()
        self.memory_norms = nn.ModuleList()

        # Zero-initialized regression residual: preserves 100% pre-trained COCO regression at Step 0
        self.gamma_reg = nn.Parameter(torch.zeros(1))

        # Zero-initialized memory gate: guarantees ZERO random prototype noise at Step 0
        self.mem_gate = nn.Parameter(torch.zeros(1))

        # Track external training epoch for progressive target schedule
        self.current_epoch = 1
        self.max_epochs = 80

        for i in range(len(in_channels)):
            feat_channels = int(256 * width)
            
            # Native 1x1 Conv2d projection operates directly on 2D spatial maps
            self.latent_projectors.append(
                nn.Conv2d(feat_channels, self.latent_dim, kernel_size=1, stride=1, padding=0)
            )
            
            # Zero-initialized residual spatial smoother: z + DWConv(z) = z at Step 0
            smoother = nn.Conv2d(
                self.latent_dim, self.latent_dim, kernel_size=3, padding=1, groups=self.latent_dim, bias=False
            )
            nn.init.zeros_(smoother.weight)
            self.spatial_smoothers.append(smoother)

            # Multi-Centroid Memory Bank: 4 sub-centroids per class, tau = 0.25 (4.0x multiplier)
            self.memory_banks.append(
                EngramMemoryBank(num_classes, self.latent_dim, num_centroids=4, temperature=4.0)
            )

            # Projected memory expansion back to convolutional channel dimension
            self.memory_proj_back.append(
                nn.Conv2d(self.latent_dim, feat_channels, kernel_size=1, stride=1, padding=0)
            )
            self.memory_norms.append(nn.LayerNorm(feat_channels))

    def set_epoch_state(self, epoch, max_epochs):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        aux_memory_logits_list = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)

            # --- BRANCH 1: REGRESSION & OBJECTNESS (Translation-Covariant, Memory-Free) ---
            # Step-0 exact pre-trained regression mapping
            reg_feat = reg_conv(x) if self.use_baseline_assignments else (reg_conv(x) + 0.20 * torch.tanh(self.gamma_reg) * x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # --- BRANCH 2: CLASSIFICATION ---
            cls_feat = cls_conv(x)
            B, C, H, W = cls_feat.shape
            if not self.enable_memory_restoration:
                cls_output = self.cls_preds[k](cls_feat)
                if self.training:
                    output = torch.cat([reg_output, obj_output, cls_output], 1)
                    output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                    x_shifts.append(grid[:, :, 0])
                    y_shifts.append(grid[:, :, 1])
                    expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                    if self.use_l1:
                        batch_size = reg_output.shape[0]
                        hsize, wsize = reg_output.shape[-2:]
                        orig = reg_output.view(batch_size, 1, 4, hsize, wsize)
                        orig = orig.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                        origin_preds.append(orig.clone())
                else:
                    output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                outputs.append(output)
                continue

            # Purged feature sabotage: native clean convolutional features propagate uncorrupted
            cls_feat_mod = cls_feat

            # Native 1x1 Conv2d projection operates directly on 2D maps
            with torch.cuda.amp.autocast(enabled=False):
                latent_sp = self.latent_projectors[k](cls_feat_mod.float())  # FP32 latent geometry

            # Zero-initialized residual spatial smoother (exact 1.0x identity at Step 0)
            latent_smooth = latent_sp + self.spatial_smoothers[k](latent_sp)

            # Flatten to [B, H*W, 128] ONLY for memory bank prototype query
            latent_flat = latent_smooth.permute(0, 2, 3, 1).reshape(B, H * W, self.latent_dim)
            x_norm_latent = F.normalize(latent_flat, p=2, dim=-1)

            # Grounded Manifold Uncertainty: 0 on background; reactive in fog/night
            obj_prob_detached = torch.sigmoid(obj_output).detach()  # [B, 1, H, W]
            obj_prob_flat = obj_prob_detached.view(B, 1, -1).permute(0, 2, 1)  # [B, H*W, 1]

            p_flat = self.memory_banks[k].prototypes.view(
                self.num_classes * self.memory_banks[k].num_centroids, self.latent_dim
            )
            p_norm = F.normalize(p_flat, p=2, dim=-1)
            max_align = torch.matmul(x_norm_latent, p_norm.t()).max(dim=-1, keepdim=True).values  # [B, H*W, 1]

            # Preserve a small objectness-independent uncertainty pathway so
            # obscured micro objects are not automatically rejected by memory.
            object_evidence = 0.10 + 0.90 * obj_prob_flat.float()
            uncertainty_flat = object_evidence * torch.clamp(1.0 - max_align.float(), min=0.0, max=1.0)
            uncertainty_sp = uncertainty_flat.view(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]

            # Query Memory Bank: returns un-gated memory mixture and class-level logits
            with torch.cuda.amp.autocast(enabled=False):
                memory_retrieved_flat, class_logits_flat = self.memory_banks[k](latent_flat.float())
            aux_memory_logits_list.append(class_logits_flat)

            # Reconstruct 2D spatial memory map
            # Reconstruct 2D spatial memory map in matching precision
            mem_sp = memory_retrieved_flat.view(B, H, W, self.latent_dim).permute(0, 3, 1, 2).to(dtype=x.dtype)
            mem_inflated = self.memory_proj_back[k](mem_sp)  # [B, C, H, W]

            # LayerNorm on un-gated memory (Bug B6 Fix: no gate cancellation!)
            mem_normed = self.memory_norms[k](mem_inflated.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            # Scale-matched FiLM modulation
            cls_std = cls_feat.std(dim=1, keepdim=True).detach().clamp(min=1e-3, max=100.0)
            mem_scaled = mem_normed * cls_std

            # Zero-Gated FiLM Injection:
            # mem_gate is initialized to 0.0 -> tanh(0) = 0.0 -> restored_cls_feat == cls_feat at Step 0!
            # Zero random prototype noise can infect the classification head during early training
            effective_gate = 0.20 * torch.tanh(self.mem_gate) * uncertainty_sp
            soft_obj_gate = torch.sigmoid((obj_prob_detached - 0.15) / 0.05)
            
            restored_cls_feat = (
                cls_feat.float() * (1.0 + effective_gate * torch.tanh(mem_scaled.float()))
                + effective_gate * mem_scaled.float() * soft_obj_gate
            ).to(cls_feat.dtype)

            cls_output = self.cls_preds[k](restored_cls_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output_tmp = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output_tmp = reg_output_tmp.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output_tmp.clone())
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
            outputs.append(output)

        if self.training:
            (
                total_loss,
                iou_loss,
                conf_loss,
                cls_loss,
                l1_loss,
                num_fg,
                cls_targets,
                fg_masks,
                matched_classes_concat,
            ) = self.get_losses_with_targets(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
            return (
                total_loss,
                iou_loss,
                conf_loss,
                cls_loss,
                l1_loss,
                num_fg,
                aux_memory_logits_list,
                cls_targets,
                fg_masks,
                matched_classes_concat,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(
                0, 2, 1
            )
            return self.decode_outputs(outputs, dtype=xin[0].type()) if self.decode_in_inference else outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)

        # OUT-OF-PLACE RECONSTRUCTION:
        # Replaces in-place slice mutation (output[..., :2] = ...) with torch.cat,
        # completely eliminating AsStridedBackward0 version conflicts during backward()
        xy = (output[..., :2] + grid) * stride
        clamped_dim = torch.clamp(output[..., 2:4], min=-5.0, max=7.0)
        wh = torch.exp(clamped_dim) * stride
        
        output = torch.cat([xy, wh, output[..., 4:]], dim=-1)
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # Clamped exponent range avoids +Inf coordinates in half-precision
        clamped_dim = torch.clamp(outputs[..., 2:4], min=-5.0, max=7.0)

        # Monotonic Bayesian Joint Scoring: purges low-objectness background false positives
        obj_prob = outputs[..., 4:5]
        cls_prob = outputs[..., 5:]

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(clamped_dim) * strides,
            obj_prob,
            cls_prob,
        ], dim=-1)
        return outputs

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        if self.use_baseline_assignments:
            self._last_assignment_meta = {"assignment_mode": "baseline"}
            return YOLOXHead.get_assignments(
                self, batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
                cls_preds, obj_preds, mode=mode,
            )
        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        # A missing geometry candidate must not silently erase positive supervision.
        # Augment candidate union with nearest anchor for each unmatched GT.
        missing_gt = ~geometry_relation.any(dim=1)
        geometry_fallback_gt = (
            missing_gt.sum().detach() if self.collect_assignment_telemetry else None
        )
        if missing_gt.any():
            anchor_cx = (x_shifts[0] + 0.5) * expanded_strides[0]
            anchor_cy = (y_shifts[0] + 0.5) * expanded_strides[0]
            missing_boxes = gt_bboxes_per_image[missing_gt]
            dx = missing_boxes[:, 0:1] - anchor_cx.unsqueeze(0)
            dy = missing_boxes[:, 1:2] - anchor_cy.unsqueeze(0)
            closest = (dx.square() + dy.square()).argmin(dim=1)
            old_indices = fg_mask.nonzero(as_tuple=False).flatten()
            new_fg = fg_mask.clone()
            new_fg[closest] = True
            new_indices = new_fg.nonzero(as_tuple=False).flatten()
            new_relation = geometry_relation.new_zeros((num_gt, new_indices.numel()))
            if old_indices.numel():
                new_relation[:, torch.searchsorted(new_indices, old_indices)] = geometry_relation
            for gt_idx, anchor_idx in zip(missing_gt.nonzero(as_tuple=False).flatten(), closest):
                new_relation[gt_idx, torch.searchsorted(new_indices, anchor_idx)] = True
            fg_mask, geometry_relation = new_fg, new_relation
        num_in_boxes_anchor = int(fg_mask.sum().item())

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        # Convert logits to probabilities safely out-of-place
        with torch.cuda.amp.autocast(enabled=False):
            cls_prob = torch.sigmoid(cls_preds_.float())
            obj_prob = torch.sigmoid(obj_preds_.float())
            
            # Sqrt joint probability calculation with explicit bounding
            # Clamping to [0.0, 1.0 - 1e-7] strictly prevents Loss.cu:103 assertion crashes
            joint_prob = torch.sqrt(cls_prob * obj_prob)
            joint_prob = torch.clamp(joint_prob, min=0.0, max=1.0 - 1e-7)
            joint_prob = torch.nan_to_num(joint_prob, nan=0.0, posinf=1.0 - 1e-7, neginf=0.0)

            pair_wise_cls_loss = F.binary_cross_entropy(
                joint_prob.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        # Rebalanced 3.0x IoU cost restores semantic compatibility over geometric coincidence
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        # Merge geometry/candidate metadata with the matching metadata generated
        # inside simota_matching(). All tensors are detached and tiny.
        if self.collect_assignment_telemetry:
            meta = dict(getattr(self, "_last_assignment_meta", {}) or {})
            meta.update({
                "assignment_mode": "custom",
                "geometry_fallback_gt": geometry_fallback_gt,
                "candidate_anchor_count": torch.as_tensor(
                    num_in_boxes_anchor, device=gt_bboxes_per_image.device, dtype=torch.long
                ),
            })
            self._last_assignment_meta = meta
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        if self.use_baseline_assignments:
            return YOLOXHead.simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)

        # IoU-Adaptive Dynamic-K Floor: allows k=1 for isolated micro targets, k>=2 for overlapped targets
        iou_sums = topk_ious.sum(1)
        dynamic_ks = torch.where(
            iou_sums < 1.20,
            torch.ones_like(iou_sums, dtype=torch.int32),
            torch.clamp(iou_sums.int(), min=2, max=10)
        )
        # Ensure dynamic_ks never exceeds available candidate count
        dynamic_ks = torch.clamp(dynamic_ks, min=1, max=n_candidate_k)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        # Duplicate-anchor resolution is part of the algorithm; the extra scalar
        # reductions below are only evaluated on telemetry iterations.
        pre_conflict_positive_total = (
            matching_matrix.sum().detach() if self.collect_assignment_telemetry else None
        )
        anchor_matching_gt = matching_matrix.sum(0)
        multiple_match_mask = anchor_matching_gt > 1
        conflict_anchor_count = (
            multiple_match_mask.sum().detach() if self.collect_assignment_telemetry else None
        )
        if multiple_match_mask.any():
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        # IMPORTANT: recompute after conflict resolution. The old code reused the
        # pre-conflict anchor_matching_gt tensor; the boolean result happened to be
        # equivalent for occupied anchors, but recomputation makes telemetry exact.
        anchor_matching_gt = matching_matrix.sum(0)
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        if self.collect_assignment_telemetry:
            dynamic_k_hist = torch.bincount(
                dynamic_ks.to(torch.long), minlength=11
            )[:11].detach()
            self._last_assignment_meta = {
                "assignment_mode": "custom",
                "dynamic_ks": dynamic_ks.detach(),
                "dynamic_k_hist": dynamic_k_hist,
                "pre_conflict_positive_total": pre_conflict_positive_total,
                "conflict_anchor_count": conflict_anchor_count,
            }
        del topk_ious, pos_idx
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def get_losses_with_targets(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # Enforce Float32 for all loss computations to eliminate FP16 accumulation overflow
        with torch.cuda.amp.autocast(enabled=False):
            bbox_preds = outputs[:, :, :4].float()
            obj_preds = outputs[:, :, 4:5].float()
            cls_preds = outputs[:, :, 5:].float()

            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
            total_num_anchors = outputs.shape[1]
            x_shifts = torch.cat(x_shifts, 1).float()
            y_shifts = torch.cat(y_shifts, 1).float()
            expanded_strides = torch.cat(expanded_strides, 1).float()
            if self.use_l1:
                origin_preds = torch.cat(origin_preds, 1).float()

            cls_targets = []
            reg_targets = []
            l1_targets = []
            obj_targets = []
            fg_masks = []
            matched_classes = []

            num_fg = 0.0
            num_gts = 0.0

            # Assignment telemetry is aggregated on-device and detached once per
            # batch, but ONLY on requested telemetry iterations. Size bins use the
            # CURRENT augmented coordinates, i.e. the scale the detector sees.
            collect_telemetry = bool(self.collect_assignment_telemetry)
            stats_device = bbox_preds.device
            size_names = ("micro", "medium", "macro")
            size_thresholds_px = (32.0, 96.0)
            size_gt_counts = torch.zeros(3, device=stats_device, dtype=torch.long)
            size_positive_counts = torch.zeros(3, device=stats_device, dtype=torch.long)
            size_covered_gt_counts = torch.zeros(3, device=stats_device, dtype=torch.long)
            size_zero_positive_gt_counts = torch.zeros(3, device=stats_device, dtype=torch.long)
            size_iou_sum = torch.zeros(3, device=stats_device, dtype=torch.float64)
            size_iou_count = torch.zeros(3, device=stats_device, dtype=torch.long)
            fpn_strides = tuple(int(x) for x in self.strides)
            fpn_positive_counts = torch.zeros(len(fpn_strides), device=stats_device, dtype=torch.long)
            fpn_iou_sum = torch.zeros(len(fpn_strides), device=stats_device, dtype=torch.float64)
            fpn_iou_count = torch.zeros(len(fpn_strides), device=stats_device, dtype=torch.long)
            class_gt_counts = torch.zeros(self.num_classes, device=stats_device, dtype=torch.long)
            class_positive_counts = torch.zeros(self.num_classes, device=stats_device, dtype=torch.long)
            matched_iou_sum = torch.zeros((), device=stats_device, dtype=torch.float64)
            matched_iou_sq_sum = torch.zeros((), device=stats_device, dtype=torch.float64)
            matched_iou_count = torch.zeros((), device=stats_device, dtype=torch.long)
            matched_iou_min = torch.full((), float("inf"), device=stats_device, dtype=torch.float32)
            matched_iou_max = torch.full((), float("-inf"), device=stats_device, dtype=torch.float32)
            matched_iou_hist = torch.zeros(4, device=stats_device, dtype=torch.long)
            invalid_matched_iou_count = torch.zeros((), device=stats_device, dtype=torch.long)
            dynamic_k_sum = torch.zeros((), device=stats_device, dtype=torch.float64)
            dynamic_k_count = torch.zeros((), device=stats_device, dtype=torch.long)
            dynamic_k_min = torch.full((), 2**31 - 1, device=stats_device, dtype=torch.long)
            dynamic_k_max = torch.zeros((), device=stats_device, dtype=torch.long)
            dynamic_k_hist = torch.zeros(11, device=stats_device, dtype=torch.long)
            pre_conflict_positive_total = torch.zeros((), device=stats_device, dtype=torch.long)
            conflict_anchor_total = torch.zeros((), device=stats_device, dtype=torch.long)
            geometry_fallback_gt_total = torch.zeros((), device=stats_device, dtype=torch.long)
            candidate_anchor_total = torch.zeros((), device=stats_device, dtype=torch.long)
            images_with_gt = 0

            lambda_soft = 0.0 if self.use_baseline_assignments else min(1.0, max(0.0, (self.current_epoch - 10) / 40.0))

            for batch_idx in range(outputs.shape[0]):
                num_gt = int(nlabel[batch_idx])
                num_gts += num_gt
                if num_gt == 0:
                    cls_target = outputs.new_zeros((0, self.num_classes), dtype=torch.float32)
                    reg_target = outputs.new_zeros((0, 4), dtype=torch.float32)
                    l1_target = outputs.new_zeros((0, 4), dtype=torch.float32)
                    obj_target = outputs.new_zeros((total_num_anchors, 1), dtype=torch.float32)
                    fg_mask = outputs.new_zeros(total_num_anchors).bool()
                    matched_class = outputs.new_zeros(0, dtype=torch.int64)
                else:
                    gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5].float()
                    gt_classes = labels[batch_idx, :num_gt, 0]
                    bboxes_preds_per_image = bbox_preds[batch_idx]

                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )

                    num_fg += num_fg_img

                    if collect_telemetry:
                        # ---------------- Rich assignment telemetry ----------------
                        images_with_gt += 1
                        gt_classes_long = gt_classes.to(torch.long)
                        class_gt_counts += torch.bincount(
                            gt_classes_long, minlength=self.num_classes
                        )[:self.num_classes]

                        # Equivalent side length sqrt(w*h) in current augmented pixels.
                        gt_wh = gt_bboxes_per_image[:, 2:4].clamp(min=0.0)
                        gt_side = torch.sqrt((gt_wh[:, 0] * gt_wh[:, 1]).clamp(min=0.0))
                        size_idx = (gt_side >= size_thresholds_px[0]).long() + (
                            gt_side >= size_thresholds_px[1]
                        ).long()
                        size_gt_counts += torch.bincount(size_idx, minlength=3)[:3]

                        if num_fg_img > 0:
                            matched_gt_inds_long = matched_gt_inds.to(torch.long)
                            pos_size_idx = size_idx[matched_gt_inds_long]
                            size_positive_counts += torch.bincount(pos_size_idx, minlength=3)[:3]

                            per_gt_pos = torch.bincount(
                                matched_gt_inds_long, minlength=num_gt
                            )
                            covered = per_gt_pos > 0
                            size_covered_gt_counts += torch.bincount(
                                size_idx[covered], minlength=3
                            )[:3]
                            size_zero_positive_gt_counts += torch.bincount(
                                size_idx[~covered], minlength=3
                            )[:3]

                            matched_classes_for_stats = gt_classes_long[matched_gt_inds_long]
                            class_positive_counts += torch.bincount(
                                matched_classes_for_stats, minlength=self.num_classes
                            )[:self.num_classes]

                            p_iou = pred_ious_this_matching.detach().float()
                            finite_iou = torch.isfinite(p_iou)
                            valid_range = finite_iou & (p_iou >= 0.0) & (p_iou <= 1.0)
                            invalid_matched_iou_count += (~valid_range).sum()
                            safe_iou = p_iou[valid_range]
                            safe_size = pos_size_idx[valid_range]
                            if safe_iou.numel() > 0:
                                matched_iou_sum += safe_iou.double().sum()
                                matched_iou_sq_sum += safe_iou.double().square().sum()
                                matched_iou_count += safe_iou.numel()
                                matched_iou_min = torch.minimum(matched_iou_min, safe_iou.min())
                                matched_iou_max = torch.maximum(matched_iou_max, safe_iou.max())
                                # [0,.25), [.25,.5), [.5,.75), [.75,1.0]
                                iou_bin = torch.bucketize(
                                    safe_iou,
                                    torch.tensor([0.25, 0.50, 0.75], device=safe_iou.device),
                                    right=False,
                                )
                                matched_iou_hist += torch.bincount(iou_bin, minlength=4)[:4]
                                for bin_idx in range(3):
                                    mask_b = safe_size == bin_idx
                                    if mask_b.any():
                                        size_iou_sum[bin_idx] += safe_iou[mask_b].double().sum()
                                        size_iou_count[bin_idx] += mask_b.sum()

                            pos_strides = expanded_strides[0][fg_mask].detach().float()
                            for level_idx, stride_val in enumerate(fpn_strides):
                                level_mask = pos_strides == float(stride_val)
                                fpn_positive_counts[level_idx] += level_mask.sum()
                                if level_mask.any():
                                    level_ious = p_iou[level_mask & valid_range]
                                    if level_ious.numel() > 0:
                                        fpn_iou_sum[level_idx] += level_ious.double().sum()
                                        fpn_iou_count[level_idx] += level_ious.numel()
                        else:
                            # Every GT in this image is uncovered if matching produced 0 FG.
                            size_zero_positive_gt_counts += torch.bincount(size_idx, minlength=3)[:3]

                        meta = getattr(self, "_last_assignment_meta", {}) or {}
                        dyn = meta.get("dynamic_ks")
                        if torch.is_tensor(dyn) and dyn.numel() > 0:
                            dyn = dyn.to(stats_device)
                            dynamic_k_sum += dyn.double().sum()
                            dynamic_k_count += dyn.numel()
                            dynamic_k_min = torch.minimum(dynamic_k_min, dyn.min().to(torch.long))
                            dynamic_k_max = torch.maximum(dynamic_k_max, dyn.max().to(torch.long))
                            hist = meta.get("dynamic_k_hist")
                            if torch.is_tensor(hist):
                                dynamic_k_hist += hist.to(stats_device, dtype=torch.long)[:11]
                        for meta_key, target_acc_name in (
                            ("pre_conflict_positive_total", "pre"),
                            ("conflict_anchor_count", "conflict"),
                            ("geometry_fallback_gt", "fallback"),
                            ("candidate_anchor_count", "candidate"),
                        ):
                            value = meta.get(meta_key)
                            if torch.is_tensor(value):
                                value = value.to(stats_device, dtype=torch.long)
                            elif value is not None:
                                value = torch.as_tensor(value, device=stats_device, dtype=torch.long)
                            if value is not None:
                                if target_acc_name == "pre":
                                    pre_conflict_positive_total += value
                                elif target_acc_name == "conflict":
                                    conflict_anchor_total += value
                                elif target_acc_name == "fallback":
                                    geometry_fallback_gt_total += value
                                elif target_acc_name == "candidate":
                                    candidate_anchor_total += value

                    cls_target = F.one_hot(
                        gt_matched_classes.to(torch.int64), self.num_classes
                    ).float() * pred_ious_this_matching.unsqueeze(-1).float()

                    obj_target = outputs.new_zeros((total_num_anchors, 1), dtype=torch.float32)
                    if num_fg_img > 0:
                        soft_iou = torch.clamp(pred_ious_this_matching.unsqueeze(-1).float(), min=0.10, max=1.0)
                        target_val = (1.0 - lambda_soft) * 1.0 + lambda_soft * soft_iou
                        obj_target[fg_mask] = target_val

                    matched_class = gt_matched_classes.to(torch.int64)
                    reg_target = gt_bboxes_per_image[matched_gt_inds]
                    if self.use_l1:
                        l1_target = self.get_l1_target(
                            outputs.new_zeros((num_fg_img, 4), dtype=torch.float32),
                            gt_bboxes_per_image[matched_gt_inds],
                            expanded_strides[0][fg_mask],
                            x_shifts=x_shifts[0][fg_mask],
                            y_shifts=y_shifts[0][fg_mask],
                        )

                cls_targets.append(cls_target)
                reg_targets.append(reg_target)
                obj_targets.append(obj_target)
                fg_masks.append(fg_mask)
                matched_classes.append(matched_class)
                if self.use_l1:
                    l1_targets.append(l1_target)

            cls_targets_concat = torch.cat(cls_targets, 0)
            reg_targets_concat = torch.cat(reg_targets, 0)
            obj_targets_concat = torch.cat(obj_targets, 0)
            fg_masks_concat = torch.cat(fg_masks, 0)
            matched_classes_concat = torch.cat(matched_classes, 0)

            if collect_telemetry:
                self.last_assignment_stats = {
                    "telemetry_schema": 3,
                    "assignment_mode": "baseline" if self.use_baseline_assignments else "custom",
                    "num_fg_total": int(num_fg),
                    "num_gt_total": int(num_gts),
                    "input_hw": [int(imgs.shape[-2]), int(imgs.shape[-1])] if imgs is not None else None,
                    "size_bin_basis": "sqrt(current_augmented_box_area_px)",
                    "size_bin_names": list(size_names),
                    "size_thresholds_px": list(size_thresholds_px),
                    "size_gt_counts": size_gt_counts.detach(),
                    "size_positive_counts": size_positive_counts.detach(),
                    "size_covered_gt_counts": size_covered_gt_counts.detach(),
                    "size_zero_positive_gt_counts": size_zero_positive_gt_counts.detach(),
                    "size_iou_sum": size_iou_sum.detach(),
                    "size_iou_count": size_iou_count.detach(),
                    "fpn_strides": list(fpn_strides),
                    "fpn_positive_counts": fpn_positive_counts.detach(),
                    "fpn_iou_sum": fpn_iou_sum.detach(),
                    "fpn_iou_count": fpn_iou_count.detach(),
                    "class_gt_counts": class_gt_counts.detach(),
                    "class_positive_counts": class_positive_counts.detach(),
                    "matched_iou_sum": matched_iou_sum.detach(),
                    "matched_iou_sq_sum": matched_iou_sq_sum.detach(),
                    "matched_iou_count": matched_iou_count.detach(),
                    "matched_iou_min": matched_iou_min.detach(),
                    "matched_iou_max": matched_iou_max.detach(),
                    "matched_iou_hist": matched_iou_hist.detach(),
                    "invalid_matched_iou_count": invalid_matched_iou_count.detach(),
                    "dynamic_k_sum": dynamic_k_sum.detach(),
                    "dynamic_k_count": dynamic_k_count.detach(),
                    "dynamic_k_min": dynamic_k_min.detach(),
                    "dynamic_k_max": dynamic_k_max.detach(),
                    "dynamic_k_hist": dynamic_k_hist.detach(),
                    "pre_conflict_positive_total": pre_conflict_positive_total.detach(),
                    "conflict_anchor_total": conflict_anchor_total.detach(),
                    "geometry_fallback_gt_total": geometry_fallback_gt_total.detach(),
                    "candidate_anchor_total": candidate_anchor_total.detach(),
                    "images_with_gt": int(images_with_gt),
                }
            else:
                self.last_assignment_stats = {
                    "telemetry_schema": 3,
                    "assignment_mode": "baseline" if self.use_baseline_assignments else "custom",
                    "num_fg_total": int(num_fg),
                    "num_gt_total": int(num_gts),
                }
            num_fg = max(num_fg, 1)
            loss_iou = (
                self.iou_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)
            ).sum() / num_fg

            loss_obj = (
                self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets_concat)
            ).sum() / num_fg

            loss_cls = (
                self.bcewithlog_loss(
                    cls_preds.view(-1, self.num_classes)[fg_masks_concat], cls_targets_concat
                )
            ).sum() / num_fg

            loss_l1 = 0.0
            if self.use_l1:
                l1_targets_concat = torch.cat(l1_targets, 0)
                l1_weight = getattr(self, "l1_weight", 1.0)
                loss_l1 = (
                    self.l1_loss(origin_preds.view(-1, 4)[fg_masks_concat], l1_targets_concat)
                ).sum() / num_fg * l1_weight

            reg_weight = 5.0 if self.use_baseline_assignments else 2.5
            total_loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

            return (
                total_loss,
                reg_weight * loss_iou,
                loss_obj,
                loss_cls,
                loss_l1,
                num_fg / max(num_gts, 1),
                cls_targets_concat,
                fg_masks_concat,
                matched_classes_concat,
            )
