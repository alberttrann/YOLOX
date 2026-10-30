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
from .ttt_modules import EngramMemoryBank, UncertaintyEstimator
from .yolo_head import YOLOXHead


class TDE_Head(YOLOXHead):
    """
    TDE-YOLOX Decoupled Detection Head (Surgical vNext).
    
    Surgical Fixes Applied:
      - Patch 3.1: SimOTA IoU matching cost weight increased from 3.0x to 5.0x (fixes halo drift).
      - Patch 3.2: Micro-target dynamic-K floor set to min=2 (prevents micro-target K=1 anchor cliff).
      - Patch 3.3: Zero-cost identity skip connection: reg_feat = reg_conv(x) + x (preserves SPI phase).
      - Patch 3.4: Rescales conf_loss by resolution factor r_scale = max(1.0, HW / (640*640)) to prevent over-suppression.
      - Patch 3.5: FP16 exponent bounding clamp([-5.0, 7.0]) & Task-aligned NMS score modulation.
      - Caveat 14.5: Hyperspherical cosine matching with tau=0.10 (multiplier=10.0).
      - Caveat 14.6: Detached presence gate obj_mask = sigmoid(obj_output).detach().
      - Non-Destructive FiLM: Scale-matched additive residual modulation preserves macro-objects in fog.
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

        # Upgrade IoU loss to OIoU (Exponential shape-calibrated regression)
        self.iou_loss = IOUloss(reduction="none", loss_type=loss_type)

        self.latent_dim = 128
        self.memory_banks = nn.ModuleList()
        self.uncertainty_gates = nn.ModuleList()
        self.latent_projectors = nn.ModuleList()
        self.memory_norms = nn.ModuleList()

        for i in range(len(in_channels)):
            feat_channels = int(256 * width)
            self.latent_projectors.append(nn.Linear(feat_channels, self.latent_dim))
            # Temperature = 10.0 enforces stable contrastive matching (tau = 0.10)
            self.memory_banks.append(
                EngramMemoryBank(num_classes, self.latent_dim, temperature=10.0)
            )
            self.uncertainty_gates.append(UncertaintyEstimator(self.latent_dim))
            self.memory_norms.append(nn.LayerNorm(feat_channels))

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
            # PATCH 3.3 APPLIED: Zero-cost identity skip preserves high-frequency SPI phase channels
            reg_feat = reg_conv(x) + x
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # --- BRANCH 2: CLASSIFICATION (Engram-Augmented) ---
            cls_feat = cls_conv(x)
            B, C, H, W = cls_feat.shape

            # Adversarial Regularization (Forces Memory Utilization on Clean Data)
            if self.training:
                f_mask = (torch.rand(B, 1, H, W, device=x.device) > 0.15).float()
                cls_feat_mod = (cls_feat + torch.randn_like(cls_feat) * 0.05) * f_mask
            else:
                cls_feat_mod = cls_feat

            cls_feat_flat = cls_feat_mod.permute(0, 2, 3, 1).reshape(B, H * W, C)
            latent_vec = self.latent_projectors[k](cls_feat_flat)

            # Continuous Uncertainty Gate
            uncertainty = self.uncertainty_gates[k](latent_vec)  # [B, H*W, 1]

            # CAVEAT 14.6 RESOLUTION: Explicitly detach objectness mask to prevent gradient leakage
            obj_mask = torch.sigmoid(obj_output.view(B, 1, -1).permute(0, 2, 1)).detach()

            # Hyperspherical Memory Retrieval (CAVEAT 14.5 RESOLUTION)
            memory_gated, norm_cosine_logits = self.memory_banks[k](
                latent_vec, uncertainty, obj_mask
            )
            aux_memory_logits_list.append(norm_cosine_logits)

            # DYNAMIC SCALE-MATCHED FiLM INJECTION (Preserves Macro Buses and Trucks)
            memory_inflated = F.linear(memory_gated, self.latent_projectors[k].weight.t())
            memory_normed = self.memory_norms[k](memory_inflated)

            cls_std = cls_feat_flat.std(dim=-1, keepdim=True).detach().clamp(min=1e-3, max=100.0)
            memory_scaled = memory_normed * cls_std

            gate_spatial = uncertainty.view(B, H, W, 1).permute(0, 3, 1, 2)
            mem_spatial = memory_scaled.view(B, H, W, C).permute(0, 3, 1, 2)

            # Non-destructive FiLM residual: Base features are NEVER multiplied by (1 - G)
            restored_cls_feat = (
                cls_feat * (1.0 + torch.tanh(gate_spatial * mem_spatial))
                + gate_spatial * mem_spatial
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
        output[..., :2] = (output[..., :2] + grid) * stride

        # PATCH 3.5 APPLIED: Clamped log-space box dimension pre-exp to eliminate FP16 overflow (>65,504)
        clamped_dim = torch.clamp(output[..., 2:4], min=-5.0, max=7.0)
        output[..., 2:4] = torch.exp(clamped_dim) * stride
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

        # PATCH 3.5 APPLIED: Clamped exponent range avoids +Inf coordinates in half-precision
        clamped_dim = torch.clamp(outputs[..., 2:4], min=-5.0, max=7.0)

        # PATCH 3.5 APPLIED: Task-aligned NMS score modulation: Score = obj^0.2 * cls^0.8
        obj_aligned = outputs[..., 4:5].clamp(min=1e-6).pow(0.2)
        cls_aligned = outputs[..., 5:].clamp(min=1e-6).pow(0.8)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(clamped_dim) * strides,
            obj_aligned,
            cls_aligned,
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

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        # PATCH 3.1 APPLIED: Upgraded IoU matching cost multiplier from 3.0 to 5.0
        # Prevents loose halo/reflection anchors from winning dynamic target assignment
        cost = (
            pair_wise_cls_loss
            + 5.0 * pair_wise_ious_loss
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

        # PATCH 3.2 APPLIED: Minimum anchor floor min=2 prevents micro-target gradient instability
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=2, max=10)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
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
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
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
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets_concat = torch.cat(cls_targets, 0)
        reg_targets_concat = torch.cat(reg_targets, 0)
        obj_targets_concat = torch.cat(obj_targets, 0)
        fg_masks_concat = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)
        ).sum() / num_fg

        # PATCH 3.4 APPLIED: High-resolution objectness loss normalization factor
        # Eliminates the +44% background over-suppression penalty at 576x1024
        r_scale = max(1.0, float(imgs.shape[-2] * imgs.shape[-1]) / (640.0 * 640.0))
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets_concat)
        ).sum() / (num_fg * r_scale)

        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks_concat], cls_targets_concat
            )
        ).sum() / num_fg

        loss_l1 = 0.0
        if self.use_l1:
            l1_targets_concat = torch.cat(l1_targets, 0)
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks_concat], l1_targets_concat)
            ).sum() / num_fg

        reg_weight = 5.0
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
        )