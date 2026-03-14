import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

class YOLOX(nn.Module):
    """
    TDE-YOLOX Unified Architecture.
    Integrates Functional TTT-Stem, Tribrid Neck, and Engram Head.
    Fixed: Check 3 (Memory Collapse) & Check 4 (Gate Training).
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            # Default fallback, should be TDE_Head in actual Exp
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        # main wrapper needs to know the class count for the Memory Anchor Loss
        self.num_classes = head.num_classes 
        
        # --- META-LEARNING ENGINE STATE ---
        self.current_epoch = 0
        self.max_epochs = 80 
        
        # Annealing Schedule 
        self.warmup_end = 10
        self.ramp_end = 80 # Extended ramp (was 50) - Slower increase
        self.prob_start = 0.1
        self.prob_end = 0.7 # Lower cap (was 0.6) - Less noise injection

    def set_meta_training_state(self, epoch, max_epochs):
        """Called by Trainer at epoch start to drive the adaptation schedule."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def _get_ttt_probability(self):
        """Calculates Stochastic Meta-Learning Probability."""
        if not self.training:
            return 1.0 # Always adapt during inference/validation
            
        if self.current_epoch < self.warmup_end:
            return self.prob_start
            
        # Unified Linear Ramp - NO GENTLE LANDING
        progress = (self.current_epoch - self.warmup_end) / (self.ramp_end - self.warmup_end)
        
        # Clamp progress to 1.0 just in case current_epoch > ramp_end
        progress = max(0.0, min(1.0, progress))
        
        return self.prob_start + (self.prob_end - self.prob_start) * progress

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob)

        if self.training:
            (det_loss, iou_l, conf_l, cls_l, l1_l, num_fg, 
             aux_mem_logits, cls_targets, fg_masks) = self.head(fpn_outs, targets, x)
            
            # Supervised Memory Anchor Loss (sufficient to prevent collapse)
            memory_anchor_loss = self._calculate_supervised_memory_loss(
                aux_mem_logits, cls_targets, fg_masks
            )
            
            # Dropped the dangerous Orthogonal Loss. 
            # 0.05 weight allows detection to lead, memory to follow safely.
            total_loss = det_loss + (0.05 * memory_anchor_loss) 

            return {
                "total_loss": total_loss,
                "iou_loss": iou_l,
                "l1_loss": l1_l,
                "conf_loss": conf_l,
                "cls_loss": cls_l,
                "mem_loss": memory_anchor_loss,
                "ttt_prob": current_ttt_prob 
            }
        else:
            return self.head(fpn_outs)
        
    def _calculate_supervised_memory_loss(self, aux_mem_logits, cls_targets, fg_masks):
        """
        FIXED: Correct ordering to match fg_masks anchor grid layout.
        
        aux_mem_logits: list of 3 tensors, each [B, HW_k, num_classes]
        fg_masks: [B * total_anchors] boolean, ordered as:
                [img0: P3+P4+P5 | img1: P3+P4+P5 | ...]
        """

        # BEFORE (incorrect - scale-first ordering):
        # all_logits = torch.cat(
        #     [l.view(-1, self.num_classes) for l in aux_mem_logits], dim=0
        # )
        
        # AFTER (correct - batch-first ordering matching fg_masks):
        all_logits = torch.cat(aux_mem_logits, dim=1).view(-1, self.num_classes)
        # aux_mem_logits: list of [B, HW_k, 9]
        # cat(dim=1): [B, HW_P3+HW_P4+HW_P5, 9]  <- scales within each image
        # view(-1, 9): [B*total_anchors, 9]  <- matches fg_masks ordering exactly
        
        # 2. Extract retrieval scores for Foreground (Object) pixels only
        # matches the dimension of cls_targets [num_fg, num_classes]
        fg_logits = all_logits[fg_masks.view(-1)]
        
        # 3. Supervise via Binary Cross Entropy (matching YOLOX classification style)
        if fg_logits.shape[0] > 0:
            # use the same targets the main detector uses!
            # ensures the Memory Bank and the Conv Head are perfectly synced.
            mem_loss = F.binary_cross_entropy_with_logits(fg_logits, cls_targets)
        else:
            mem_loss = all_logits.sum() * 0.0 # Zero loss if no objects
            
        return mem_loss

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        # Inference mode: Always run TTT (1.0 probability)
        fpn_outs = self.backbone(x, ttt_prob=1.0)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)