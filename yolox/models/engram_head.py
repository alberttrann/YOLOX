#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.3: Unified Single-Dispatch Engram Head & Level-Aware ATSS-SimOTA

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
    Hyperspherical Associative Memory Bank with Inverted Atmospheric Entropy Modulation.
    """
    def __init__(self, num_classes=9, num_modes=64, latent_dim=128, temperature=16.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.temperature = float(temperature)
        
        # 576 Prototypes living on unit hypersphere S^127
        self.prototypes = nn.Parameter(torch.randn(num_classes * num_modes, latent_dim))
        nn.init.orthogonal_(self.prototypes)
        self.prototypes.is_hypersphere = True
        
        self.dw_hf = nn.Conv2d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim)
        self.w_hf = nn.Parameter(torch.ones(1) * 0.1)
        
        self.v_q = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.v_k = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.omega = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, anchor_feat, h_local, objectness_mask, epoch=0, h_s=0.0):
        B, C, H, W = h_local.shape
        D = self.latent_dim
        
        p_interp = F.interpolate(anchor_feat, size=(H, W), mode='bilinear', align_corners=True)
        dw_local = self.dw_hf(h_local)
        
        # FP32 RMS Shield to prevent half-precision overflow
        p_fp32 = p_interp.float()
        dw_fp32 = dw_local.float()
        
        p_scale = p_fp32 * torch.rsqrt(p_fp32.pow(2).mean(dim=1, keepdim=True) + 1e-5)
        dw_scale = dw_fp32 * torch.rsqrt(dw_fp32.pow(2).mean(dim=1, keepdim=True) + 1e-5)
        
        q = (p_scale + torch.tanh(self.w_hf.float()) * dw_scale).to(h_local.dtype)
        
        # Orthonormal Hyperspherical Query and Prototype Alignment
        q_norm = F.normalize(q.float().permute(0, 2, 3, 1).reshape(B, H * W, D), p=2, dim=-1, eps=1e-5)
        p_norm = F.normalize(self.prototypes.float(), p=2, dim=-1, eps=1e-5)
        
        # Scale-Adaptive Temperature Lookup
        scores_fp32 = torch.matmul(q_norm, p_norm.t()) * self.temperature
        attn_weights = F.softmax(scores_fp32, dim=-1)
        retrieved_memory = torch.matmul(attn_weights, p_norm).to(q.dtype)
        
        # Direct Hyperspherical Logit Extraction (Eliminates Dispatch 1!)
        scores_by_class = scores_fp32.view(B, H * W, self.num_classes, self.num_modes)
        log_k = math.log(float(self.num_modes))
        raw_class_scores = torch.logsumexp(scores_by_class, dim=-1) - log_k
        
        # Orthonormal Margin Uncertainty Calculation (Direct on Prototype Scores)
        p_mem = torch.sigmoid(raw_class_scores.detach())
        top2_p, _ = torch.topk(p_mem, k=2, dim=-1)
        margin = torch.clamp(top2_p[..., 0:1] - top2_p[..., 1:2], min=0.0, max=1.0)
        
        obj_flat = objectness_mask.permute(0, 2, 3, 1).reshape(B, H * W, 1)
        uncertainty = (1.0 - margin) * obj_flat.detach()
        
        # Geodesic Distance Penalty & Atmospheric Modulation
        h_flat = h_local.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h_dir = F.normalize(h_flat, p=2, dim=-1, eps=1e-5)
        mem_dir = F.normalize(retrieved_memory, p=2, dim=-1, eps=1e-5)
        
        slope_q = F.softplus(torch.matmul(h_dir, self.v_q))
        slope_k = F.softplus(torch.matmul(mem_dir, self.v_k))
        dist_penalty = torch.clamp(self.omega, min=0.0, max=2.0) * (slope_q.unsqueeze(-1) + slope_k.unsqueeze(-1))
        
        # Balanced Gate Logit: Eliminates the oppressive -2.5 baseline clamp
        gate_logit = 4.0 * uncertainty - dist_penalty - 0.5
        g_raw = torch.sigmoid(gate_logit) * obj_flat
        
        curriculum_floor = 0.20 * max(0.0, 1.0 - float(epoch) / 15.0)
        g_effective = torch.max(g_raw, torch.tensor(curriculum_floor, device=q.device, dtype=q.dtype))
        
        # Inverted Atmospheric Entropy Modulator:
        # Opens memory support under visual degradation rather than shutting off!
        h_max = math.log(400.0)
        h_ratio = torch.clamp(h_s.float() / h_max, min=0.0, max=1.0)
        atmo_modulator = 0.75 + 0.50 * torch.sigmoid(4.0 * (h_ratio - 0.50)).to(q.dtype)
        g_final = torch.clamp(g_effective * atmo_modulator, min=0.0, max=0.70)
        
        mem_spatial = retrieved_memory.reshape(B, H, W, D).permute(0, 3, 1, 2).to(h_local.dtype)
        gate_spatial = g_final.reshape(B, H, W, 1).permute(0, 3, 1, 2).to(h_local.dtype)
        
        return mem_spatial, gate_spatial, raw_class_scores


class TDE_Head(YOLOXHead):
    """
    TDE-YOLOX v3.3 Master Head:
      - Memory-Free Regression Tower (GIoU + Adaptive NWD Loss)
      - Single-Dispatch Classification Tower with Adaptive RMS Scale Matching
      - Level-Aware Dynamic ATSS-SimOTA Assigner
    """
    def __init__(self, num_classes=9, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        self.latent_dim = 128
        self.current_epoch = 0
        
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

        # Scale-Dependent Temperature Hierarchy
        # P3 (Stride 8): T=32.0 (Sharp discrimination for Bikes, Signs, Lights)
        # P4 (Stride 16): T=24.0 (Balanced discrimination for Cars, Riders)
        # P5 (Stride 32): T=16.0 (Continuous multi-mode density for Buses, Trucks)
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
            
            # Kaiming Uniform Decoder Initialization: Eliminates early-epoch gradient starvation
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
            
            # Standard unconstrained linear classification heads with independent biases
            self.cls_preds.append(
                nn.Conv2d(feat_c, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.reg_preds.append(nn.Conv2d(feat_c, 4, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(feat_c, 1, kernel_size=1, stride=1, padding=0))

    def initialize_biases(self, prior_prob):
        # 1. Initialize objectness prior bias
        for conv in self.obj_preds:
            if conv.bias is not None:
                b = conv.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # 2. Initialize independent classification head prior biases (-4.595)
        prior_b = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            if conv.bias is not None:
                b = conv.bias.view(1, -1)
                b.data.fill_(prior_b)
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        # 3. Zero-initialize regression offsets
        for conv in self.reg_preds:
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, inputs, labels=None, imgs=None):
        xin, p_anchors, h_s = inputs
        h_s = torch.nan_to_num(h_s, nan=math.log(400.0), posinf=math.log(400.0), neginf=0.0)

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        aux_memory_logits = []

        scale_anchors = [p_anchors[1], p_anchors[2], p_anchors[2]]

        for k, (cls_conv, reg_conv, stride_l, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            
            # 1. REGRESSION TOWER (Memory-Free)
            reg_feat = reg_conv(x)
            reg_out = self.reg_preds[k](reg_feat)
            obj_out = self.obj_preds[k](reg_feat)
            obj_mask = torch.sigmoid(obj_out)

            # 2. CLASSIFICATION TOWER: UNIFIED SINGLE DISPATCH
            cls_feat = cls_conv(x)
            
            anchor_lat = self.anchor_projectors[k](scale_anchors[k])
            local_lat = self.local_projectors[k](cls_feat)
            
            # Memory retrieval directly yields raw_scores without redundant cls_preds calls!
            mem_restored, gate, raw_scores = self.memory_banks[k](
                anchor_lat, local_lat, obj_mask, epoch=self.current_epoch, h_s=h_s
            )
            aux_memory_logits.append(raw_scores)
            
            mem_conv = self.manifold_decoders[k](mem_restored)
            
            # === FP32 RMS SCALE SHIELD with Strict DType Preservation ===
            cls_fp32 = cls_feat.float()
            mem_fp32 = mem_conv.float()
            rms_cls = torch.sqrt(torch.mean(cls_fp32.pow(2), dim=1, keepdim=True) + 1e-5)
            rms_mem = torch.sqrt(torch.mean(mem_fp32.pow(2), dim=1, keepdim=True) + 1e-5)
            scale_ratio = torch.clamp(rms_cls / rms_mem, min=0.10, max=10.0)
            mem_conv_scaled = (mem_fp32 * scale_ratio).to(cls_feat.dtype)
            # ============================================================
            
            # Enforce gate matches cls_feat.dtype to prevent float32 type promotion
            gate = gate.to(cls_feat.dtype)
            restored_feat = ((1.0 - gate) * cls_feat + gate * mem_conv_scaled).to(cls_feat.dtype)
            
            # Scale-Proportional Post-Smoothing: Stride 8 gets 0.05, Stride 32 gets 0.25
            smooth_weight = torch.tensor(0.05 + 0.10 * float(k), device=restored_feat.device, dtype=cls_feat.dtype)
            dw_out = self.post_engram_dw[k](restored_feat)
            restored_feat = (restored_feat + smooth_weight * torch.tanh(dw_out)).to(cls_feat.dtype)
            
            # Single Classification Dispatch: Evaluated strictly once per scale!
            # Direct Hyperspherical Prototype Logit Extraction
            B, _, H, W = cls_feat.shape

            w_conv = self.cls_preds[k].weight
            w_norm = F.normalize(w_conv.float(), p=2, dim=[1, 2, 3], eps=1e-5).to(cls_feat.dtype)
            
            # 1. Normalized dot-product + Shared Scalar Prior Bias
            # Unconstrained Convolutional Highway (Carries full clean-domain capacity)
            z_conv = self.cls_preds[k](restored_feat)
            
            # Zero-Mean Centered Prototype Logit Injection (Residual memory assist)
            raw_scores_spatial = raw_scores.permute(0, 2, 1).reshape(B, self.num_classes, H, W).to(cls_feat.dtype)
            proto_mean = raw_scores_spatial.mean(dim=1, keepdim=True)
            s_proto_centered = raw_scores_spatial - proto_mean
            
            # Additive Decision Boundary
            cls_out = z_conv + gate * s_proto_centered

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
        
        # =========================================================================
        # SYNCHRONIZED REGRESSION HANDOFF:
        # At Epoch 65 (80 - 15 = 65), when Mosaic turns off and L1 activates:
        #   - reg_weight (GIoU) scales to 5.0
        #   - nwd_weight fades from 2.0 -> 0.25 (stops curvature conflict with L1!)
        #   - loss_l1_weight activates at 1.0
        # =========================================================================
        max_epochs = getattr(self, "max_epochs", 80)
        if self.current_epoch >= (max_epochs - 15):
            reg_weight = 5.0
            nwd_weight = 0.25  # Faded down to prevent curvature conflict with L1
            loss_l1_weight = 1.0
        else:
            reg_weight = 3.0
            nwd_weight = 2.0   # Active during representation phase for micro road users
            loss_l1_weight = 0.0
        # =========================================================================

        loss_giou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg
        loss_nwd = (self.nwd_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg

        obj_preds_fp32 = obj_preds.view(-1, 1).float()
        loss_obj = F.binary_cross_entropy_with_logits(obj_preds_fp32, obj_targets_concat.float(), reduction="sum") / num_fg

        # =========================================================================
        # TDE-YOLOX v3.4: Certified Positive-Balanced Focal Classification Loss
        # =========================================================================
        fg_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks_concat].float()
        p_cls = torch.sigmoid(fg_cls_preds)
        p_t = p_cls * cls_targets_concat.float() + (1.0 - p_cls) * (1.0 - cls_targets_concat.float())
        p_t_safe = torch.clamp(p_t, min=1e-4, max=1.0 - 1e-4)

        gamma_adapt = torch.clamp(2.0 * (1.0 - torch.exp(-3.0 * p_t_safe)), min=0.5, max=2.0)
        focal_factor = (1.0 - p_t_safe).pow(gamma_adapt).detach()

        raw_bce = F.binary_cross_entropy_with_logits(fg_cls_preds, cls_targets_concat.float(), reduction="none")
        weighted_bce = focal_factor * raw_bce

        pos_mask = (cls_targets_concat > 0.0).float()
        neg_mask = 1.0 - pos_mask

        # 1. Negative loss normalized globally across all num_fg
        loss_cls_neg = (weighted_bce * neg_mask).sum() / num_fg

        # 2. Positive loss balanced dynamically per active class
        pos_bce_per_class = (weighted_bce * pos_mask).sum(dim=0)
        class_fg_counts = cls_targets_concat.sum(dim=0)
        active_class_mask = class_fg_counts > 0.0

        if active_class_mask.any():
            loss_cls_pos = (pos_bce_per_class[active_class_mask] / torch.clamp(class_fg_counts[active_class_mask], min=1.0)).mean()
        else:
            loss_cls_pos = 0.0

        loss_cls = loss_cls_pos + loss_cls_neg
        # =========================================================================

        # Replace the old 'l1_ramp' logic with the clean loss_l1_weight multiplier:
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
        mask_p3 = (strides_1d == 8)
        mask_p4 = (strides_1d == 16)
        mask_p5 = (strides_1d == 32)
        level_masks = [mask_p3, mask_p4, mask_p5]
        level_strides = [8.0, 16.0, 32.0]

        gt_cx = gt_bboxes_per_image[:, 0:1]
        gt_cy = gt_bboxes_per_image[:, 1:2]
        gt_w = gt_bboxes_per_image[:, 2:3]
        gt_h = gt_bboxes_per_image[:, 3:4]
        gt_scale = torch.sqrt(gt_w * gt_h + 1e-5)

        anchor_cx = (x_shifts[0] + 0.5) * strides_1d
        anchor_cy = (y_shifts[0] + 0.5) * strides_1d

        # 1. PHYSICAL BODY ANCHOR SAMPLING WINDOW
        delta_x = torch.max(strides_1d.unsqueeze(0) * 2.5, gt_w * 0.50)
        delta_y = torch.max(strides_1d.unsqueeze(0) * 2.5, gt_h * 0.50)

        x_centers = anchor_cx.unsqueeze(0)
        y_centers = anchor_cy.unsqueeze(0)

        c_l = x_centers - (gt_cx - delta_x)
        c_r = (gt_cx + delta_x) - x_centers
        c_t = y_centers - (gt_cy - delta_y)
        c_b = (gt_cy + delta_y) - y_centers
        is_in_centers = torch.stack([c_l, c_r, c_t, c_b], 2).min(dim=-1).values > 0.0

        # 2. ANISOTROPIC MAHALANOBIS CANDIDATE SELECTION
        k_cand_per_level = 16
        candidate_indices_list = []

        for l_mask, s_l in zip(level_masks, level_strides):
            l_indices = torch.nonzero(l_mask).squeeze(-1)
            l_cx = anchor_cx[l_mask]
            l_cy = anchor_cy[l_mask]

            norm_dx = (gt_cx - l_cx.unsqueeze(0)) / torch.clamp(gt_w * 0.50, min=s_l)
            norm_dy = (gt_cy - l_cy.unsqueeze(0)) / torch.clamp(gt_h * 0.50, min=s_l)
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

        # 3. Asymmetric Scale-Prior Barrier
        canonical_scale = cand_strides * 4.0
        scale_ratio = gt_scale / (canonical_scale + 1e-5)
        c_scale_prior = torch.where(
            scale_ratio < 1.0,
            torch.log(scale_ratio + 1e-5).pow(2) * 0.15,
            torch.clamp(torch.log(scale_ratio + 1e-5).pow(2) * 0.02, max=0.05)
        )

        p_c = bboxes_preds[:, :2]
        g_c = gt_bboxes_per_image[:, :2]
        c_dist_sq = (p_c.unsqueeze(0) - g_c.unsqueeze(1)).pow(2).sum(-1)
        p_s = bboxes_preds[:, 2:4]
        g_s = gt_bboxes_per_image[:, 2:4]
        s_dist_sq = (p_s.unsqueeze(0) - g_s.unsqueeze(1)).pow(2).sum(-1) / 4.0
        w2_sq = c_dist_sq + s_dist_sq + 1e-7
        diag_gt_safe = torch.clamp(torch.sqrt(g_s.pow(2).sum(-1) + 1e-7).unsqueeze(1) + 1e-5, min=8.0)
        pair_wise_nwd = torch.exp(-2.0 * (torch.sqrt(w2_sq) / diag_gt_safe))

        # 4. Scale-Adaptive Dynamic-K Budget in SimOTA (100% Vectorized GPU Implementation)
        # FIX: Define hybrid_overlap before using it
        hybrid_overlap = 0.5 * (pair_wise_ious + pair_wise_nwd)

        is_macro = (gt_scale.view(-1, 1) >= 96.0)
        n_cand = min(16, hybrid_overlap.size(1))
        
        topk_overlap, _ = torch.topk(hybrid_overlap, n_cand, dim=1)  # [num_gt, n_cand]

        cand_mask = torch.ones_like(topk_overlap)
        if n_cand > 10:
            cand_mask[:, 10:] = is_macro.float()

        dynamic_ks = (topk_overlap * cand_mask).sum(dim=1).round().int()

        min_k = torch.where(is_macro.squeeze(-1), 3, 1)
        max_k = torch.where(is_macro.squeeze(-1), 16, 10)
        
        dynamic_ks = torch.clamp(dynamic_ks, min=min_k, max=max_k)
        # Note: The slow Python 'for g in range(num_gt):' loop is deleted.

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

        # 5. Continuous Loss-Aligned SimOTA Cost Weights
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