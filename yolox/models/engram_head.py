import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_head import YOLOXHead 
from .ttt_modules import EngramMemoryBank, UncertaintyEstimator

class TDE_Head(YOLOXHead):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        
        self.latent_dim = 128
        self.memory_banks = nn.ModuleList()
        self.uncertainty_gates = nn.ModuleList()
        self.latent_projectors = nn.ModuleList()

        for i in range(len(in_channels)):
            feat_channels = int(256 * width)
            self.latent_projectors.append(nn.Linear(feat_channels, self.latent_dim))
            self.memory_banks.append(EngramMemoryBank(num_classes, self.latent_dim))
            self.uncertainty_gates.append(UncertaintyEstimator(self.latent_dim))

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
            
            # --- BRANCH 1: REGRESSION (High Fidelity, Memory-Free) ---
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # --- BRANCH 2: CLASSIFICATION (Engram-Augmented) ---
            cls_feat = cls_conv(x)
            B, C, H, W = cls_feat.shape
            
            # --- PATCH: SCHEDULED ADVERSARIAL WAKEUP ---
            # Avoid noise in the very first epochs to allow base feature formation
            if self.training:
                # Extract epoch progress from the model wrapper if available
                epoch = getattr(self, "current_epoch", 0)
                adv_scale = min(1.0, epoch / 20.0) if epoch > 3 else 0.0
                
                if adv_scale > 0:
                    # 1. Gradual Noise Injection
                    cls_feat = cls_feat + torch.randn_like(cls_feat) * (0.05 * adv_scale)
                    # 2. Gradual Spatial Feature Dropout
                    f_mask = (torch.rand(B, 1, H, W, device=x.device) > (0.15 * adv_scale)).float()
                    cls_feat = cls_feat * f_mask
            
            cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
            # Step A: Latent Projection
            latent_vec = self.latent_projectors[k](cls_feat_flat) # [B, HW, 128]
            
            # Step B: Calculate Uncertainty 
            # Passing the full [B, HW, 128] vector to the estimator
            uncertainty = self.uncertainty_gates[k](latent_vec) # [B, HW, 1]
            
            # Step C: Retrieve Memory
            memory_feat = self.memory_banks[k](latent_vec, uncertainty)
            
            # 3. Project back to feature space
            memory_inflated = F.linear(memory_feat, self.latent_projectors[k].weight.t())
            
            # 4. Convex Switch Fusion
            # At Epoch 1, gate_spatial ≈ 0.0, so restored_cls_feat ≈ cls_feat
            # The model behaves exactly like Baseline YOLOX, preventing the 0.000 AP crash.
            gate_spatial = uncertainty.view(B, H, W, 1).permute(0, 3, 1, 2)
            restored_cls_feat = (1.0 - gate_spatial) * cls_feat + gate_spatial * memory_inflated.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            cls_output = self.cls_preds[k](restored_cls_feat)

            # --- YOLOX Standard Logic & Target Collection ---
            if self.training:
                # Calculate exact retrieval scores for Contrastive Anchor Loss
                x_norm = F.normalize(latent_vec, p=2, dim=-1)
                p_norm = F.normalize(self.memory_banks[k].prototype_layer.weight, p=2, dim=-1)
                exact_retrieval_scores = torch.matmul(x_norm, p_norm.t()) * self.memory_banks[k].temperature
                aux_memory_logits_list.append(exact_retrieval_scores)

                # Standard YOLOX Target preparation
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output_tmp = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output_tmp = reg_output_tmp.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output_tmp.clone())
            else:
                # Efficient Inference Logic
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            
            outputs.append(output)

        if self.training:
            # Return targets so Memory Bank can be supervised in yolox.py
            total_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, cls_targets, fg_masks = self.get_losses_with_targets(
                imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype
            )
            return total_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, aux_memory_logits_list, cls_targets, fg_masks
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            return self.decode_outputs(outputs, dtype=xin[0].type()) if self.decode_in_inference else outputs

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
        """
        Supervised Assignment Extraction.
        This function identifies which pixels belong to which object class
        and returns those targets for Engram Memory Bank supervision.
        """
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]   # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]    # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects per image

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
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

                # Run SimOTA Dynamic Label Assignment
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

                # Construct Classification Targets
                # [num_fg_img, num_classes]
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

        # Concatenate all targets across the batch
        cls_targets_concat = torch.cat(cls_targets, 0)
        reg_targets_concat = torch.cat(reg_targets, 0)
        obj_targets_concat = torch.cat(obj_targets, 0)
        fg_masks_concat = torch.cat(fg_masks, 0)
        
        # Loss Calculation Logic
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks_concat], reg_targets_concat)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets_concat)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks_concat], cls_targets_concat)).sum() / num_fg
        
        loss_l1 = 0.0
        if self.use_l1:
            l1_targets_concat = torch.cat(l1_targets, 0)
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks_concat], l1_targets_concat)).sum() / num_fg

        reg_weight = 5.0
        total_det_loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        # Return: Standard Metrics + The Targets needed for Memory Bank supervision
        return (
            total_det_loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
            cls_targets_concat, # [Total_FG_Objects, Num_Classes]
            fg_masks_concat     # [Batch, Total_Pixels] (Boolean)
        )