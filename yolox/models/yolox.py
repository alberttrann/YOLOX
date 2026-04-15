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
        self.ramp_end = 60 # Extended ramp (was 50) - Slower increase
        self.prob_start = 0.1
        self.prob_end = 0.6 # noise injection

    def set_meta_training_state(self, epoch, max_epochs):
        """Called by Trainer at epoch start to drive the adaptation schedule."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def _get_ttt_probability(self):
        """
        Unified Meta-Learning Scheduler.
        Respects __init__ variables to allow external control.
        """
        # Rule 1: Inference always uses 100% adaptation (TTT)
        if not self.training:
            return 1.0
            
        # Rule 2: Warmup Phase (Fixed low probability for stability)
        if self.current_epoch < self.warmup_end:
            return self.prob_start
            
        # Rule 3: Post-Ramp Phase (Stay at the defined limit)
        if self.current_epoch >= self.ramp_end:
            return self.prob_end
            
        # Rule 4: The Linear Ramp (Dynamic Complexity Scaling)
        progress = (self.current_epoch - self.warmup_end) / (self.ramp_end - self.warmup_end)
        return self.prob_start + (self.prob_end - self.prob_start) * progress

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()
        # Calculate Curriculum Progress [0.0 to 1.0]
        progress = 0.0
        if self.training and self.current_epoch >= self.warmup_end:
            progress = min(1.0, (self.current_epoch - self.warmup_end) / (self.max_epochs - self.warmup_end))
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob)

        if self.training:
            # Head now returns detection metrics + memory scores + ground truth targets
            (det_loss, iou_l, conf_l, cls_l, l1_l, num_fg, 
             aux_mem_logits, cls_targets, fg_masks, proto_loss) = self.head(fpn_outs, targets, x, curriculum_progress=progress)
            
            # --- SUPERVISED IDENTITY ANCHORING ---
            # align the Memory Retrieval scores directly with Ground Truth.
            # forces 'Car' features to align with the 'Car Prototype'.
            memory_anchor_loss = self._calculate_supervised_memory_loss(
                aux_mem_logits, cls_targets, fg_masks
            )
            
            # We use 0.2 weight for memory loss to ensure strong prototype separation
            total_loss = det_loss + (0.2 * memory_anchor_loss)

            return {
                "total_loss": total_loss,
                "iou_loss": iou_l,
                "l1_loss": l1_l,
                "conf_loss": conf_l,
                "cls_loss": cls_l,
                "mem_loss": memory_anchor_loss,
                "proto_loss": proto_loss,
                "num_fg": num_fg,
                "ttt_prob": current_ttt_prob 
            }
        else:
            return self.head(fpn_outs)

    def _calculate_supervised_memory_loss(self, aux_mem_logits, cls_targets, fg_masks):
        """
        EXPERT FIX: Quality Focal Memory Anchoring.
        Handles Soft Labels (SimOTA targets) and heavily penalizes rare class misalignment.
        """
        all_logits = torch.cat([l.view(-1, self.num_classes) for l in aux_mem_logits], dim=0)
        fg_logits = all_logits[fg_masks.view(-1)]
        
        if fg_logits.shape[0] == 0:
            return all_logits.sum() * 0.0
            
        # 1. Base BCE Loss (No reduction)
        bce_loss = F.binary_cross_entropy_with_logits(fg_logits, cls_targets, reduction='none')
        
        # 2. Quality Focal Weight (Handles Soft Labels perfectly)
        # Weight = |Target - Prediction|^gamma
        pt = torch.sigmoid(fg_logits)
        focal_weight = torch.abs(cls_targets - pt) ** 2.0
        
        # 3. Class-Balanced Alpha (Empirical BDD100K Weights)
        # 0:Car, 1:Bus, 2:Truck, 3:Person, 4:Rider, 5:Bike, 6:Motor, 7:Light, 8:Sign
        alpha_weights = torch.tensor([
            0.1, 0.5, 0.4, 0.2, 0.9, 0.9, 0.8, 0.6, 0.5
        ], device=fg_logits.device)
        alpha = alpha_weights.unsqueeze(0).expand(fg_logits.shape[0], -1)
        
        # 4. Final Loss
        focal_loss = alpha * focal_weight * bce_loss
        return focal_loss.mean()

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        # Inference mode: Always run TTT (1.0 probability)
        fpn_outs = self.backbone(x, ttt_prob=1.0)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
    def get_engram_prototypes(self):
        """
        Returns the latent prototypes for each class across the 3 FPN scales.
        Useful for t-SNE visualization to prove class separation.
        """
        prototypes = []
        for memory_bank in self.head.memory_banks:
            # Shape: [num_classes, latent_dim]
            prototypes.append(memory_bank.prototypes.detach().cpu().numpy())
        return prototypes