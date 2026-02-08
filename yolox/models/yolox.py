#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
import math

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

class YOLOX(nn.Module):
    """
    YOLOX model module with TDE-YOLOX Meta-Learning Extensions.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        
        # --- META-LEARNING STATE ---
        self.current_epoch = 0
        self.max_epochs = 80 # Default, will be updated by Exp
        
        # Annealing Schedule Config
        self.warmup_end = 10
        self.ramp_end = 40
        self.prob_start = 0.1
        self.prob_end = 0.5

    def set_meta_training_state(self, epoch, max_epochs):
        """
        Called by the Trainer at the start of each epoch.
        Updates the internal state to calculate TTT probability.
        """
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def _get_ttt_probability(self):
        """
        Calculates Meta-Saliency Annealing Probability.
        Schedule:
        - Epoch 0-10: Fixed 0.1 (Warmup stability)
        - Epoch 11-40: Linear Ramp 0.1 -> 0.5 (Curriculum Learning)
        - Epoch 40+: Fixed 0.5 (High-fidelity Adaptation)
        """

        # --- GENTLE LANDING ---
        if self.current_epoch >= 79:
            return 0.2  # Low probability to stabilize features

        # --- 73++ STRESS TEST ---
        #if self.current_epoch > 72:
            #return 0.7 # Push to 70% TTT in final phase for stress testing
        # --- 73++ STRESS TEST ---

        if not self.training:
            return 1.0 # Always TTT during inference/validation
            
        if self.current_epoch < self.warmup_end:
            return self.prob_start
        
        if self.current_epoch >= self.ramp_end:
            return self.prob_end
        
            
        # Linear Ramp
        progress = (self.current_epoch - self.warmup_end) / (self.ramp_end - self.warmup_end)
        return self.prob_start + (self.prob_end - self.prob_start) * progress

    def forward(self, x, targets=None):
        # Calculate current TTT probability for this batch
        current_ttt_prob = self._get_ttt_probability()

        # Pass prob down to backbone (through the neck/pafpn)
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "ttt_prob": current_ttt_prob 
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        # Visualize always uses inference mode, so TTT is active
        fpn_outs = self.backbone(x, ttt_prob=1.0)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)