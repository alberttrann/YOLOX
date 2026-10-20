#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.6: Hardened Anti-Collision Memory Bank with True S^127 Chordal Distance

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid
from .losses import IOUloss, AdaptiveNWDloss
from .network_blocks import BaseConv
from .yolo_head import YOLOXHead


class AntiCollisionEngramBank(nn.Module):
    def __init__(self, num_classes=9, num_modes=64, latent_dim=128, temperature=16.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.temperature = float(temperature)

        self.prototypes = nn.Parameter(torch.randn(num_classes * num_modes, latent_dim))
        nn.init.orthogonal_(self.prototypes)
        self.prototypes.is_hypersphere = True

        self.dw_hf = nn.Conv2d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim)
        self.w_hf = nn.Parameter(torch.ones(1) * 0.20)  # In Group 2 (No-Decay)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, anchor_feat, h_local, objectness_mask, epoch=0, h_s=0.0, is_p3=False):
        B, C, H, W = h_local.shape
        D = self.latent_dim

        # P3 LOCAL-DOMINATED QUERY RE-WEIGHTING (v3.6):
        p_interp = F.interpolate(anchor_feat, size=(H, W), mode='bilinear', align_corners=False)
        dw_local = self.dw_hf(h_local)
        p_scale = p_interp.float() * torch.rsqrt(p_interp.float().pow(2).mean(dim=1, keepdim=True) + 1e-5)
        dw_scale = dw_local.float() * torch.rsqrt(dw_local.float().pow(2).mean(dim=1, keepdim=True) + 1e-5)

        if is_p3:
            # 80% Local Texture, 20% Coarse Anchor for fine-grained micro targets
            q = (0.25 * p_scale + dw_scale).to(h_local.dtype)
        else:
            q = (p_scale + torch.clamp(self.w_hf.float(), 0.05, 1.0) * dw_scale).to(h_local.dtype)

        q_norm = F.normalize(q.float().permute(0, 2, 3, 1).reshape(B, H * W, D), p=2, dim=-1, eps=1e-5)
        p_norm = F.normalize(self.prototypes.float(), p=2, dim=-1, eps=1e-5)

        # 1. Raw cosine similarity on S^127: shape [B, H*W, 576] in [-1, 1]
        cos_sim_all = torch.matmul(q_norm, p_norm.t())

        # 2. Scale-Dependent Temperature for Memory Retrieval Attention:
        # (T=32 for P3 sharp attention; T=16 for P5 soft density)
        scores_fp32 = cos_sim_all * self.temperature
        attn_weights = F.softmax(scores_fp32, dim=-1)
        retrieved_memory = torch.matmul(attn_weights, p_norm).to(q.dtype)

        # 3. Unified Mode Aggregation for Categorical Class Logits (v3.6):
        # Uses constant T_mode = 16.0 across all scales to guarantee exact machine parity!
        T_mode = 16.0
        log_k = math.log(float(self.num_modes))
        cos_by_class = cos_sim_all.view(B, H * W, self.num_classes, self.num_modes)
        raw_lse = torch.logsumexp(cos_by_class * T_mode, dim=-1) - log_k
        raw_class_scores = (raw_lse / T_mode) * 16.0

        # CORRECTION OF THE INVERTED BILATERAL GATE (v3.6):
        # High Certainty (+4.0) opens; Ambiguity / Distance Penalty (-2.0) shuts
        p_mem = F.softmax(raw_class_scores.detach().float(), dim=-1)
        top2_p, _ = torch.topk(p_mem, k=2, dim=-1)
        certainty = torch.clamp(top2_p[..., 0:1] - top2_p[..., 1:2], min=0.0, max=1.0).to(q.dtype)

        mem_dir = F.normalize(retrieved_memory.float(), p=2, dim=-1, eps=1e-5)
        cos_sim = torch.sum(q_norm * mem_dir, dim=-1, keepdim=True)
        chordal_dist = torch.clamp(1.0 - cos_sim, min=0.0, max=2.0).to(q.dtype)
        dist_penalty = torch.clamp(self.omega, min=0.0, max=2.0) * chordal_dist

        gate_logit = 4.0 * certainty - 2.0 * dist_penalty - 1.0

        # OBJECTNESS BYPASS FOR FAINT OBJECTS:
        obj_flat = objectness_mask.permute(0, 2, 3, 1).reshape(B, H * W, 1)
        gate_prob = torch.sigmoid(gate_logit)
        effective_obj = torch.max(obj_flat.detach(), gate_prob * 0.10)
        obj_envelope = torch.clamp(effective_obj / 0.10, min=0.0, max=1.0)
        g_effective = gate_prob * obj_envelope

        # DEFENSIVE DIMENSION GUARD (v3.6):
        # Force h_ratio to shape [B, 1, 1] (or [1, 1, 1]) to cleanly match 3D g_effective [B, H*W, 1]
        if torch.is_tensor(h_s):
            h_val = h_s.float()
            if h_val.numel() == B:
                h_ratio = h_val.view(B, 1, 1).clamp(0.0, 1.0)
            elif h_val.numel() == 1:
                h_ratio = h_val.view(1, 1, 1).clamp(0.0, 1.0)
            else:
                h_ratio = h_val.mean().view(1, 1, 1).clamp(0.0, 1.0)
        else:
            h_ratio = torch.tensor(float(h_s), device=q.device, dtype=torch.float32).view(1, 1, 1).clamp(0.0, 1.0)

        atmo_modulator = 0.75 + 0.50 * torch.sigmoid(4.0 * (h_ratio - 0.50)).to(q.dtype)
        
        # [B, H*W, 1] * [B, 1, 1] -> strictly [B, H*W, 1] (7,680 elements!)
        g_final = torch.clamp(g_effective * atmo_modulator, min=0.0, max=0.70)

        mem_spatial = retrieved_memory.reshape(B, H, W, D).permute(0, 3, 1, 2).to(h_local.dtype)
        gate_spatial = g_final.reshape(B, H, W, 1).permute(0, 3, 1, 2).to(h_local.dtype)
        return mem_spatial, gate_spatial, raw_class_scores


class TDE_Head(YOLOXHead):
    """
    TDE-YOLOX v3.6 Master Head:
      - Memory-Free High-Fidelity Regression Tower (GIoU + Adaptive NWD Loss).
      - Decoupled Classification Tower with RMS Scale-Matched Convex Switching.
      - Level-Aware Dynamic ATSS-SimOTA Assigner.
    """
    def __init__(self, num_classes=9, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        self.latent_dim = 128
        self.current_epoch = 0
        self.max_epochs = 80

        self.iou_loss = IOUloss(reduction="none", loss_type="giou")
        self.nwd_loss = AdaptiveNWDloss(kappa=2.0, reduction="none")

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        c4 = int(in_channels[1] * width)
        c5 = int(in_channels[2] * width)
        feat_c = int(256 * width)

        self.anchor_projectors = nn.ModuleList([
            nn.Conv2d(c4, self.latent_dim, kernel_size=1),
            nn.Conv2d(c5, self.latent_dim, kernel_size=1),
            nn.Conv2d(c5, self.latent_dim, kernel_size=1)
        ])

        self.local_projectors = nn.ModuleList()
        self.memory_banks = nn.ModuleList()
        self.manifold_decoders = nn.ModuleList()
        self.post_engram_dw = nn.ModuleList()

        temperatures = [32.0, 24.0, 16.0]

        for i in range(len(in_channels)):
            in_c = int(in_channels[i] * width)
            self.stems.append(BaseConv(in_c, feat_c, 1, 1, act=act))

            # Regression Tower (Memory-Free)
            self.reg_convs.append(nn.Sequential(
                BaseConv(feat_c, feat_c, 3, 1, act=act),
                BaseConv(feat_c, feat_c, 3, 1, act=act)
            ))
            # Classification Tower
            self.cls_convs.append(nn.Sequential(
                BaseConv(feat_c, feat_c, 3, 1, act=act),
                BaseConv(feat_c, feat_c, 3, 1, act=act)
            ))

            self.local_projectors.append(nn.Conv2d(feat_c, self.latent_dim, kernel_size=1))
            self.memory_banks.append(
                AntiCollisionEngramBank(
                    num_classes, num_modes=64, latent_dim=self.latent_dim, temperature=temperatures[i]
                )
            )

            dec = nn.Conv2d(self.latent_dim, feat_c, kernel_size=1)
            nn.init.kaiming_uniform_(dec.weight, a=1.0)
            if dec.bias is not None:
                nn.init.zeros_(dec.bias)
            self.manifold_decoders.append(dec)

            dw_smooth = nn.Conv2d(feat_c, feat_c, kernel_size=3, padding=1, groups=feat_c)
            nn.init.zeros_(dw_smooth.weight)
            if dw_smooth.bias is not None:
                nn.init.zeros_(dw_smooth.bias)
            self.post_engram_dw.append(dw_smooth)

            self.cls_preds.append(
                nn.Conv2d(feat_c, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.reg_preds.append(nn.Conv2d(feat_c, 4, kernel_size=1, stride=1, padding=0))
            # Evaluates directly on stem feature x (128 channels) to decouple from reg_conv:
            self.obj_preds.append(nn.Conv2d(feat_c, 1, kernel_size=1, stride=1, padding=0))

    def initialize_biases(self, prior_prob):
        super().initialize_biases(prior_prob)
        for conv in self.reg_preds:
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, inputs, labels=None, imgs=None):
        xin, p_anchors, h_s = inputs
        if torch.is_tensor(h_s):
            h_s = torch.nan_to_num(h_s, nan=0.50, posinf=1.0, neginf=0.10)
        else:
            h_s = torch.tensor(float(h_s), device=xin[0].device, dtype=torch.float32)

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        aux_memory_logits = []

        scale_anchors = [p_anchors[1], p_anchors[2], p_anchors[2]]

        for k, (cls_conv, reg_conv, stride_l, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)

            # 1. REGRESSION TOWER (High Fidelity, Memory-Free)
            reg_feat = reg_conv(x)
            reg_out = self.reg_preds[k](reg_feat)
            # Evaluated directly on stem feature x to stop negative anchors from poisoning reg_conv:
            obj_out = self.obj_preds[k](x)
            obj_mask = torch.sigmoid(obj_out)

            # 2. CLASSIFICATION TOWER: Unified Associative Restoration
            cls_feat = cls_conv(x)
            B, _, H, W = cls_feat.shape

            anchor_lat = self.anchor_projectors[k](scale_anchors[k])
            local_lat = self.local_projectors[k](cls_feat)

            mem_restored, gate, raw_scores = self.memory_banks[k](
                anchor_lat, local_lat, obj_mask, epoch=self.current_epoch, h_s=h_s, is_p3=(k == 0)
            )
            aux_memory_logits.append(raw_scores)

            mem_conv = self.manifold_decoders[k](mem_restored)

            # UNCONSTRAINED LOCAL ENERGY MATCHING (Removes unit-floor inflation):
            rms_cls = torch.sqrt(torch.mean(cls_feat.float().pow(2), dim=1, keepdim=True) + 1e-5)
            rms_mem = torch.sqrt(torch.mean(mem_conv.float().pow(2), dim=1, keepdim=True) + 1e-5)
            scale_ratio = torch.clamp(rms_cls / (rms_mem + 1e-5), min=0.10, max=3.0)
            mem_conv_scaled = (mem_conv.float() * scale_ratio).to(cls_feat.dtype)

            restored_feat = ((1.0 - 0.60 * gate) * cls_feat + gate * mem_conv_scaled).to(cls_feat.dtype)
            smooth_weight = torch.tensor(0.05 + 0.10 * float(k), device=restored_feat.device, dtype=cls_feat.dtype)
            restored_feat = (restored_feat + smooth_weight * torch.tanh(self.post_engram_dw[k](restored_feat))).to(cls_feat.dtype)

            # UNIFORM 0.250 EFFECTIVE INJECTION WEIGHT ACROSS ALL SCALES:
            proto_scale_map = [0.500, 0.375, 0.250]
            raw_scores_spatial = raw_scores.permute(0, 2, 1).reshape(B, self.num_classes, H, W).to(cls_feat.dtype)
            s_proto_centered = raw_scores_spatial - raw_scores_spatial.mean(dim=1, keepdim=True)
            cls_out = self.cls_preds[k](restored_feat) + (gate * s_proto_centered * proto_scale_map[k])

            if self.training:
                output = torch.cat([reg_out, obj_out, cls_out], 1)
                output, grid = self.get_output_and_grid(output, k, stride_l, xin[0].dtype)
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1], device=xin[0].device, dtype=xin[0].dtype).fill_(stride_l)
                )
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
            return self.decode_outputs(outputs, dtype=xin[0].dtype) if self.decode_in_inference else outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]

        grid = self.grids[k]
        if grid.shape[2:4] != (hsize, wsize) or grid.device != output.device or grid.dtype != dtype:
            yv, xv = meshgrid([torch.arange(hsize, device=output.device), torch.arange(wsize, device=output.device)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(dtype=dtype, device=output.device)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)

        xy = (output[..., :2] + grid) * stride
        wh = torch.exp(torch.clamp(output[..., 2:4], min=-10.0, max=10.0)) * stride
        rest = output[..., 4:]

        output_decoded = torch.cat([xy, wh, rest], dim=-1)
        return output_decoded, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize, device=outputs.device), torch.arange(wsize, device=outputs.device)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, device=outputs.device))

        grids = torch.cat(grids, dim=1).to(dtype=dtype, device=outputs.device)
        strides = torch.cat(strides, dim=1).to(dtype=dtype, device=outputs.device)

        raw_scales = torch.exp(torch.clamp(outputs[..., 2:4], min=-10.0, max=10.0)) * strides
        box_scales = torch.clamp(raw_scales, min=1.0)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            box_scales,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

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

        # 1. CONSTANT 5.0 GIOU WEIGHT + SMOOTH NWD DECAY (Paced for 80 Epochs):
        reg_weight = 5.0
        if self.current_epoch < 50:
            nwd_weight, loss_l1_weight = 1.25, 0.00
        elif self.current_epoch < 65:
            prog = (self.current_epoch - 50) / 15.0
            nwd_weight = 1.25 - (1.25 - 0.25) * prog
            loss_l1_weight = (self.current_epoch - 58) / 7.0 if self.current_epoch >= 58 else 0.00
        else:
            nwd_weight, loss_l1_weight = 0.25, 1.00

        loss_giou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg
        loss_nwd = (self.nwd_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg

        # 2. SCALE-NORMALIZED OBJECTNESS LOSS (Cancels 576x1024 grid inflation):
        obj_preds_fp32 = obj_preds.view(-1, 1).float()
        pos_mask = obj_targets_concat.bool()
        loss_obj_pos = F.binary_cross_entropy_with_logits(obj_preds_fp32[pos_mask], obj_targets_concat[pos_mask].float(), reduction="sum") / num_fg
        loss_obj_neg = F.binary_cross_entropy_with_logits(obj_preds_fp32[~pos_mask], obj_targets_concat[~pos_mask].float(), reduction="sum") / float(obj_preds_fp32.shape[0])
        loss_obj = loss_obj_pos + (loss_obj_neg * 12.0)

        # 3. 50/50 BALANCED HYBRID CLASSIFICATION LOSS (Protects rare classes):
        fg_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks_concat].float()
        raw_bce = F.binary_cross_entropy_with_logits(fg_cls_preds, cls_targets_concat.float(), reduction="none")
        pos_mask_cls = (cls_targets_concat > 0.0).float()
        neg_mask_cls = 1.0 - pos_mask_cls

        loss_cls_neg = (raw_bce * neg_mask_cls).sum() / num_fg
        loss_cls_pos_global = (raw_bce * pos_mask_cls).sum() / num_fg
        class_fg_counts = cls_targets_concat.sum(dim=0)
        active_mask = class_fg_counts > 0.0
        pos_bce_per_class = (raw_bce * pos_mask_cls).sum(dim=0)
        loss_cls_pos_equal = (pos_bce_per_class[active_mask] / torch.clamp(class_fg_counts[active_mask], min=1.0)).mean() if active_mask.any() else 0.0

        loss_cls = (0.50 * loss_cls_pos_global + 0.50 * loss_cls_pos_equal) + loss_cls_neg

        loss_l1 = 0.0
        if self.use_l1:
            l1_targets_concat = torch.cat(l1_targets, 0)
            raw_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks_concat], l1_targets_concat)).sum() / num_fg
            loss_l1 = raw_l1 * loss_l1_weight

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
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        stride_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_w = gt_bboxes_per_image[:, 2:3]
        gt_h = gt_bboxes_per_image[:, 3:4]

        center_dist_x = torch.max(stride_dist, gt_w * 0.5)
        center_dist_y = torch.max(stride_dist, gt_h * 0.5)

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist_x
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist_x
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist_y
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist_y

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
        total_anchors = expanded_strides.shape[1]
        if num_gt == 0:
            return (
                gt_classes.new_zeros((0,)),
                torch.zeros(total_anchors, device=gt_classes.device, dtype=torch.bool),
                gt_classes.new_zeros((0,)),
                gt_classes.new_zeros((0,), dtype=torch.int64),
                0.0
            )

        strides_1d = expanded_strides[0]
        level_masks = [(strides_1d == s) for s in self.strides]
        level_strides = [float(s) for s in self.strides]

        gt_cx = gt_bboxes_per_image[:, 0:1]
        gt_cy = gt_bboxes_per_image[:, 1:2]
        gt_w = gt_bboxes_per_image[:, 2:3]
        gt_h = gt_bboxes_per_image[:, 3:4]
        gt_scale = torch.sqrt(gt_w * gt_h + 1e-5)

        anchor_cx = (x_shifts[0] + 0.5) * strides_1d
        anchor_cy = (y_shifts[0] + 0.5) * strides_1d

        delta_x = torch.max(strides_1d.unsqueeze(0) * 2.5, gt_w * 0.50)
        delta_y = torch.max(strides_1d.unsqueeze(0) * 2.5, gt_h * 0.50)

        x_centers = anchor_cx.unsqueeze(0)
        y_centers = anchor_cy.unsqueeze(0)

        c_l = x_centers - (gt_cx - delta_x)
        c_r = (gt_cx + delta_x) - x_centers
        c_t = y_centers - (gt_cy - delta_y)
        c_b = (gt_cy + delta_y) - y_centers
        is_in_centers = torch.stack([c_l, c_r, c_t, c_b], 2).min(dim=-1).values > 0.0

        # ELEVATED CANDIDATE POOL (24) to match 576x1024 resolution:
        k_cand_per_level = 24
        candidate_indices_list = []

        for l_mask, s_l in zip(level_masks, level_strides):
            l_indices = torch.nonzero(l_mask).squeeze(-1)
            l_cx = anchor_cx[l_mask]
            l_cy = anchor_cy[l_mask]

            # Aspect-ratio bounded Mahalanobis distance (max 1.6x distortion)
            den_x = torch.clamp(gt_w * 0.50, min=s_l)
            den_y = torch.clamp(gt_h * 0.50, min=s_l)
            den_x = torch.max(den_x, gt_scale * 0.35)
            den_y = torch.max(den_y, gt_scale * 0.35)

            norm_dx = (gt_cx - l_cx.unsqueeze(0)) / den_x
            norm_dy = (gt_cy - l_cy.unsqueeze(0)) / den_y
            dist_mahalanobis = norm_dx.pow(2) + norm_dy.pow(2)

            _, topk_local_idx = torch.topk(
                dist_mahalanobis, min(k_cand_per_level, l_cx.shape[0]), dim=-1, largest=False
            )
            candidate_indices_list.append(l_indices[topk_local_idx])

        candidate_indices = torch.cat(candidate_indices_list, dim=-1)

        is_in_candidates = torch.zeros(num_gt, total_anchors, device=gt_classes.device, dtype=torch.bool)
        is_in_candidates.scatter_(1, candidate_indices, True)

        is_active = is_in_centers & is_in_candidates
        fg_mask = is_active.sum(dim=0) > 0
        num_in_boxes_anchor = int(fg_mask.sum().item())

        if num_in_boxes_anchor == 0:
            fg_mask[candidate_indices[:, 0]] = True
            is_active = is_in_candidates & fg_mask.unsqueeze(0)
            num_in_boxes_anchor = int(fg_mask.sum().item())

        geometry_relation = is_active[:, fg_mask]

        bboxes_preds = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        cand_strides = strides_1d[fg_mask].unsqueeze(0)

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds, False)

        canonical_scale = cand_strides * 4.0
        scale_ratio = gt_scale / (canonical_scale + 1e-5)
        c_scale_prior = torch.clamp(torch.log(scale_ratio + 1e-5).pow(2) * 0.03, max=0.05)

        p_c = bboxes_preds[:, :2]
        g_c = gt_bboxes_per_image[:, :2]
        c_dist_sq = (p_c.unsqueeze(0) - g_c.unsqueeze(1)).pow(2).sum(-1)
        p_s = bboxes_preds[:, 2:4]
        g_s = gt_bboxes_per_image[:, 2:4]
        s_dist_sq = (p_s.unsqueeze(0) - g_s.unsqueeze(1)).pow(2).sum(-1) / 4.0
        w2_sq = c_dist_sq + s_dist_sq + 1e-7
        diag_gt_safe = torch.clamp(torch.sqrt(g_s.pow(2).sum(-1) + 1e-7).unsqueeze(1) + 1e-5, min=8.0)
        pair_wise_nwd = torch.exp(-2.0 * (torch.sqrt(w2_sq) / diag_gt_safe))

        hybrid_overlap = 0.5 * (pair_wise_ious + pair_wise_nwd)

        is_macro = (gt_scale.view(-1, 1) >= 96.0)
        n_cand = min(14, hybrid_overlap.size(1))

        topk_overlap, _ = torch.topk(hybrid_overlap, n_cand, dim=1)

        # Full candidate access across all scales (No micro truncation):
        cand_mask = torch.ones_like(topk_overlap)
        dynamic_ks = (topk_overlap * cand_mask).sum(dim=1).round().int()
        min_k = torch.where(is_macro.squeeze(-1), 3, 2)
        max_k = torch.where(is_macro.squeeze(-1), 12, 10) # Raised from 8 to 10 for micro targets
        dynamic_ks = torch.clamp(dynamic_ks, min=min_k, max=max_k)

        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        with torch.cuda.amp.autocast(enabled=False):
            cls_prob = (cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()).sqrt()
            cls_prob = torch.nan_to_num(cls_prob, nan=0.0, posinf=1.0 - 1e-7, neginf=0.0)
            cls_prob = torch.clamp(cls_prob, min=0.0, max=1.0 - 1e-7)

            gt_target = gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1).float()
            gt_target = torch.clamp(gt_target, min=0.0, max=1.0)

            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_prob.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_target,
                reduction="none"
            ).sum(-1)

        max_epochs = getattr(self, "max_epochs", 80)
        no_aug_start = max_epochs - 15
        if self.current_epoch < no_aug_start:
            cost_iou_w = 2.0
            cost_nwd_w = 2.0
        elif self.current_epoch >= (no_aug_start + 3):
            cost_iou_w = 3.0
            cost_nwd_w = 0.50
        else:
            t = float(self.current_epoch - no_aug_start + 1) / 4.0
            cost_iou_w = 2.0 + 1.0 * t
            cost_nwd_w = 2.0 - 1.50 * t

        cost = (
            pair_wise_cls_loss
            + cost_iou_w * (1.0 - pair_wise_ious)
            + cost_nwd_w * (1.0 - pair_wise_nwd)
            + c_scale_prior
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        ) = self.simota_matching_guarded(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask, dynamic_ks, num_in_boxes_anchor
        )

        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def simota_matching_guarded(
        self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask, dynamic_ks, num_in_boxes_anchor
    ):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        for gt_idx in range(num_gt):
            k_safe = min(int(dynamic_ks[gt_idx].item()), num_in_boxes_anchor)
            _, pos_idx = torch.topk(cost[gt_idx], k=k_safe, largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

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

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds