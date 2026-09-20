#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Decoupled Anti-Collision Engram Head & Calibrated SimOTA

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid
from .losses import IOUloss, AdaptiveNWDloss
from .network_blocks import BaseConv
from .yolo_head import YOLOXHead

class AntiCollisionEngramBank(nn.Module):
    """
    Hyperspherical Engram Associative Memory Bank.
    """
    def __init__(self, num_classes=9, num_modes=64, latent_dim=128, temperature=16.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # 576 Canonical Prototypes on unit hypersphere S^127 [576, 128]
        self.prototypes = nn.Parameter(torch.randn(num_classes * num_modes, latent_dim))
        nn.init.orthogonal_(self.prototypes)
        self.prototypes.is_hypersphere = True
        
        self.dw_hf = nn.Conv2d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim)
        self.w_hf = nn.Parameter(torch.ones(1) * 0.1)  # Initialized to 0.1
        
        self.v_q = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.v_k = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, anchor_feat, h_local, objectness_mask, z_conv_logits, epoch=0, h_s=0.0):
        B, C, H, W = h_local.shape
        D = self.latent_dim
        
        p_interp = F.interpolate(anchor_feat, size=(H, W), mode='bilinear', align_corners=False)
        # Tanh bounds the high-freq injection strictly between [-1.0, 1.0]
        q = p_interp + torch.tanh(self.w_hf) * self.dw_hf(h_local)
        
        # Hyperspherical Hard Attention strictly in single-precision FP32 (Z1, L1, W3)
        q_fp32 = q.float()
        p_fp32 = self.prototypes.float()
        
        q_norm = F.normalize(q_fp32.permute(0, 2, 3, 1).reshape(B, H * W, D), p=2, dim=-1, eps=1e-5)
        p_norm = F.normalize(p_fp32, p=2, dim=-1, eps=1e-5)
        
        scores_fp32 = torch.matmul(q_norm, p_norm.t()) * self.temperature
        attn_weights = F.softmax(scores_fp32, dim=-1)
        
        retrieved_memory_fp32 = torch.matmul(attn_weights, p_norm)
        retrieved_memory = retrieved_memory_fp32.to(q.dtype)
        
        scores_by_class = scores_fp32.view(B, H * W, self.num_classes, self.num_modes)
        raw_class_scores = torch.logsumexp(scores_by_class, dim=-1)  # [B, HW, 9] in FP32
        
        # Bayesian Predictive Confusion (Detached Stop-Gradient)
        z_conv_flat = z_conv_logits.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        p_conv = torch.sigmoid(z_conv_flat.detach())
        max_p, _ = torch.max(p_conv, dim=-1, keepdim=True)
        uncertainty = 1.0 - max_p
        
        # 4. GRAPE-AP SPATIAL DISTANCE PENALTY 
        h_flat = h_local.permute(0, 2, 3, 1).reshape(B, H * W, C)
        slope_q = F.softplus(torch.matmul(h_flat, self.v_q))
        slope_k = F.softplus(torch.matmul(retrieved_memory, self.v_k))
        # Clamp omega to prevent it from permanently overriding the Bayesian gate
        dist_penalty = torch.clamp(self.omega, min=0.0, max=2.0) * (slope_q.unsqueeze(-1) + slope_k.unsqueeze(-1))
        
        # Gate Logit with negative baseline offset
        gate_logit = 5.0 * uncertainty - dist_penalty - 2.5
        obj_flat = objectness_mask.permute(0, 2, 3, 1).reshape(B, H * W, 1)
        g_raw = torch.sigmoid(gate_logit) * obj_flat
        
        # Curriculum Gate Floor (Epochs 1-15)
        curriculum_floor = 0.20 * max(0.0, 1.0 - float(epoch) / 15.0)
        g_effective = torch.max(g_raw, torch.tensor(curriculum_floor, device=q.device, dtype=q.dtype))
        
        # Blizzard Optical Circuit Breaker
        if h_s >= 5.69:
            g_final = torch.zeros_like(g_effective)
        else:
            g_final = g_effective
            
        mem_spatial = retrieved_memory.reshape(B, H, W, D).permute(0, 3, 1, 2)
        gate_spatial = g_final.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        
        return mem_spatial, gate_spatial, raw_class_scores


class TDE_Head(YOLOXHead):
    def __init__(
        self,
        num_classes=9,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False
    ):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        self.latent_dim = 128
        self.current_epoch = 0
        self.nwd_loss = AdaptiveNWDloss(kappa=2.0, reduction="none")
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        
        c4 = int(in_channels[1] * width)  # 256
        c5 = int(in_channels[2] * width)  # 512
        feat_c = int(256 * width)         # 128
        
        self.anchor_projectors = nn.ModuleList([
            nn.Conv2d(c4, self.latent_dim, kernel_size=1),  # Proj 0: 256 -> 128 (P4 -> P3)
            nn.Conv2d(c5, self.latent_dim, kernel_size=1),  # Proj 1: 512 -> 128 (P5 -> P4)
            nn.Conv2d(c5, self.latent_dim, kernel_size=1)   # Proj 2: 512 -> 128 (P5 -> P5)
        ])
        
        self.local_projectors = nn.ModuleList()
        self.memory_banks = nn.ModuleList()
        self.manifold_decoders = nn.ModuleList()
        self.post_engram_dw = nn.ModuleList()

        for i in range(len(in_channels)):
            in_c = int(in_channels[i] * width)
            self.stems.append(BaseConv(in_c, feat_c, 1, 1, act=act))
            
            self.reg_convs.append(nn.Sequential(
                BaseConv(feat_c, feat_c, 3, 1, act=act),
                BaseConv(feat_c, feat_c, 3, 1, act=act)
            ))
            self.cls_convs.append(nn.Sequential(
                BaseConv(feat_c, feat_c, 3, 1, act=act),
                BaseConv(feat_c, feat_c, 3, 1, act=act)
            ))
            
            self.local_projectors.append(nn.Conv2d(feat_c, self.latent_dim, kernel_size=1))
            self.memory_banks.append(AntiCollisionEngramBank(num_classes, num_modes=64, latent_dim=self.latent_dim))
            
            dec = nn.Conv2d(self.latent_dim, feat_c, kernel_size=1)
            nn.init.normal_(dec.weight, mean=0.0, std=0.01)
            if dec.bias is not None:
                nn.init.zeros_(dec.bias)
            self.manifold_decoders.append(dec)
            
            dw_smooth = nn.Conv2d(feat_c, feat_c, kernel_size=3, padding=1, groups=feat_c)
            nn.init.zeros_(dw_smooth.weight)
            if dw_smooth.bias is not None:
                nn.init.zeros_(dw_smooth.bias)
            self.post_engram_dw.append(dw_smooth)
            
            self.cls_preds.append(nn.Conv2d(feat_c, num_classes, kernel_size=1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(feat_c, 4, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(feat_c, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, inputs, labels=None, imgs=None):
        xin, p_anchors = inputs
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        aux_memory_logits = []

        scale_anchors = [p_anchors[1], p_anchors[2], p_anchors[2]]

        for k, (cls_conv, reg_conv, stride_l, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            
            # Regression: Memory-free
            reg_feat = reg_conv(x)
            reg_out = self.reg_preds[k](reg_feat)
            obj_out = self.obj_preds[k](reg_feat)
            obj_mask = torch.sigmoid(obj_out)

            # Classification: Engram-restored
            cls_feat = cls_conv(x)
            z_conv = self.cls_preds[k](cls_feat)
            
            anchor_lat = self.anchor_projectors[k](scale_anchors[k])
            local_lat = self.local_projectors[k](cls_feat)
            
            mem_restored, gate, raw_scores = self.memory_banks[k](
                anchor_lat, local_lat, obj_mask, z_conv, epoch=self.current_epoch
            )
            aux_memory_logits.append(raw_scores)
            
            mem_conv = self.manifold_decoders[k](mem_restored)
            # Additive Gated Memory (cls_feat is ALWAYS preserved at 100%)
            restored_feat = cls_feat + gate * mem_conv
            restored_feat = restored_feat + self.post_engram_dw[k](restored_feat)
            
            cls_out = self.cls_preds[k](restored_feat)

            if self.training:
                output = torch.cat([reg_out, obj_out, cls_out], 1)
                output, grid = self.get_output_and_grid(output, k, stride_l, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_l).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_out.shape[0]
                    hsize, wsize = reg_out.shape[-2:]
                    reg_tmp = reg_out.view(batch_size, 1, 4, hsize, wsize).permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_tmp.clone())
            else:
                output = torch.cat([reg_out, obj_out.sigmoid(), cls_out.sigmoid()], 1)
            outputs.append(output)

        if self.training:
            (
                total_det_loss, loss_iou, loss_nwd, loss_obj, loss_cls, loss_l1,
                num_fg_ratio, cls_targets_concat, fg_masks_concat
            ) = self.get_losses_with_targets(
                imgs, x_shifts, y_shifts, expanded_strides, labels,
                torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype
            )
            return (
                total_det_loss, loss_iou, loss_nwd, loss_obj, loss_cls, loss_l1,
                num_fg_ratio, aux_memory_logits, cls_targets_concat, fg_masks_concat
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            return self.decode_outputs(outputs, dtype=xin[0].type()) if self.decode_in_inference else outputs

    def get_losses_with_targets(
        self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype
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
                    gt_matched_classes, fg_mask, pred_ious_this_matching,
                    matched_gt_inds, num_fg_img
                ) = self.get_assignments_with_nwd(
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds
                )
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask]
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

        num_fg = max(num_fg, 1.0)
        
        # Hybrid 3.0 GIoU + 2.0 Adaptive NWD
        loss_giou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg
        loss_nwd = (self.nwd_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg
        reg_weight = 3.0
        nwd_weight = 2.0

        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets_concat)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks_concat], cls_targets_concat)).sum() / num_fg
        
        loss_l1 = 0.0
        if self.use_l1:
            l1_targets_concat = torch.cat(l1_targets, 0)
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks_concat], l1_targets_concat)).sum() / num_fg

        total_det_loss = (reg_weight * loss_giou) + (nwd_weight * loss_nwd) + loss_obj + loss_cls + loss_l1
        
        return (
            total_det_loss,
            reg_weight * loss_giou,
            nwd_weight * loss_nwd,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1.0),
            cls_targets_concat,
            fg_masks_concat.view(outputs.shape[0], -1)
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, center_radius=2.5
    ):
        """
        Calibration 2: Loosened center_radius from 1.5 -> 2.5 strides.
        Prevents SimOTA geometry constraint from discarding fog-blurred near-misses.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    @torch.no_grad()
    def get_assignments_with_nwd(
        self, batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
        cls_preds, obj_preds
    ):
        # 1. Geometry Filter with Loosened Center Radius = 2.5 (Calibration 2)
        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, center_radius=2.5
        )

        bboxes_preds = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds.shape[0]

        # 2. Pairwise IoU and Pairwise Scale-Adaptive NWD
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds, False)

        p_c = bboxes_preds[:, :2]
        g_c = gt_bboxes_per_image[:, :2]
        c_dist_sq = (p_c.unsqueeze(0) - g_c.unsqueeze(1)).pow(2).sum(-1)

        p_s = bboxes_preds[:, 2:4]
        g_s = gt_bboxes_per_image[:, 2:4]
        s_dist_sq = (p_s.unsqueeze(0) - g_s.unsqueeze(1)).pow(2).sum(-1) / 4.0
        w2_sq = c_dist_sq + s_dist_sq + 1e-7

        diag_gt = torch.sqrt(g_s.pow(2).sum(-1) + 1e-7).unsqueeze(1) + 1e-5
        pair_wise_nwd = torch.exp(-2.0 * (torch.sqrt(w2_sq) / diag_gt))

        # 3. Dynamic-K Allocation Clamped to max=15 (Calibration 3)
        hybrid_overlap = 0.5 * (pair_wise_ious + pair_wise_nwd)
        n_candidate_k = min(15, hybrid_overlap.size(1))
        topk_ious, _ = torch.topk(hybrid_overlap, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1, max=15)

        # 4. Classification Matching Cost (Guarded against CUDA Loss.cu >= 0 and <= 1 assertion)
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        with torch.cuda.amp.autocast(enabled=False):
            cls_prob = (cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()).sqrt()
            
            # NUMERICAL GUARD: Clamp to [0.0, 1.0 - 1e-7] to prevent floating point rounding > 1.0
            cls_prob = torch.clamp(cls_prob, min=0.0, max=1.0 - 1e-7)
            cls_prob = torch.nan_to_num(cls_prob, nan=0.0)
            
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_prob.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)

        # 5. Hybrid Cost
        cost = (
            pair_wise_cls_loss
            + 2.0 * (1.0 - pair_wise_ious)
            + 2.0 * (1.0 - pair_wise_nwd)
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg