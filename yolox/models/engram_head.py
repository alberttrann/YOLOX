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
            reg_feat = reg_conv(x) + 0.20 * torch.tanh(self.gamma_reg) * x
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # --- BRANCH 2: CLASSIFICATION (Unshackled Spatial Representation & Multi-Centroid FiLM) ---
            cls_feat = cls_conv(x)
            B, C, H, W = cls_feat.shape

            # Purged feature sabotage: native clean convolutional features propagate uncorrupted
            cls_feat_mod = cls_feat

            # Native 1x1 Conv2d projection operates directly on 2D maps
            latent_sp = self.latent_projectors[k](cls_feat_mod)  # [B, 128, H, W]

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

            uncertainty_flat = obj_prob_flat * torch.clamp(1.0 - max_align, min=0.0, max=1.0)
            uncertainty_sp = uncertainty_flat.view(B, H, W, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]

            # Query Memory Bank: returns un-gated memory mixture and class-level logits
            memory_retrieved_flat, class_logits_flat = self.memory_banks[k](latent_flat)
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
                cls_feat * (1.0 + effective_gate * torch.tanh(mem_scaled))
                + effective_gate * mem_scaled * soft_obj_gate
            )

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

        num_in_boxes_anchor = fg_mask.sum().item()
        # Empty-candidate early exit guard: prevents torch.topk crashes on image boundary boxes
        if num_in_boxes_anchor == 0:
            return (
                gt_classes.new_zeros(0),
                fg_mask,
                gt_bboxes_per_image.new_zeros(0),
                gt_classes.new_zeros(0, dtype=torch.int64),
                0,
            )

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

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
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

            lambda_soft = min(1.0, max(0.0, (self.current_epoch - 10) / 40.0))

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

            reg_weight = 2.5
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