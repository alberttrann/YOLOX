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
        
        # --- TTT ANNEALING SCHEDULE ---
        self.warmup_end = 30        # TTT stays at 0.1 until E30
        self.ramp_end = 100         # Slow, stable ramp to 0.6
        self.prob_start = 0.1
        self.prob_end = 0.75

    def set_meta_training_state(self, epoch, max_epochs):
        """Called by Trainer at epoch start to drive the adaptation schedule."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs
        
        # --- PATCH: PASS STATE TO HEAD AND BACKBONE ---
        if hasattr(self.head, "current_epoch"):
            self.head.current_epoch = epoch
            
        if hasattr(self.backbone, "current_epoch"):
            self.backbone.current_epoch = epoch

    def _get_ttt_probability(self):
        if not self.training:
            return 1.0 # Always adapt at test time
            
        if self.current_epoch < 30: # Use hardcoded or self.ttt_warmup_end
            return 0.10
            
        if self.current_epoch >= 100: # Use self.ttt_ramp_end
            return 0.75
            
        # Phase III: Linear Ramp (E30 to E100)
        progress = (self.current_epoch - 30) / (100 - 30)
        return 0.10 + (0.75 - 0.10) * progress

    def forward(self, x, targets=None):
        current_ttt_prob = self._get_ttt_probability()
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob)

        if self.training:
            # Head now returns detection metrics + memory scores + ground truth targets
            (det_loss, iou_l, conf_l, cls_l, l1_l, num_fg, 
             aux_mem_logits, cls_targets, fg_masks) = self.head(fpn_outs, targets, x)
            
            # --- SUPERVISED IDENTITY ANCHORING ---
            # align the Memory Retrieval scores directly with Ground Truth.
            # forces 'Car' features to align with the 'Car Prototype'.
            memory_anchor_loss = self._calculate_supervised_memory_loss(
                aux_mem_logits, cls_targets, fg_masks
            )
            
            # OLD: total_loss = det_loss + (0.2 * memory_anchor_loss) 
            # NEW: Reduced weight to 0.05 (Let detection drive 95% of gradients)
            # Increased weight slightly for faster prototype clustering
            total_loss = det_loss + (0.10 * memory_anchor_loss)

            return {
                "total_loss": total_loss,
                "iou_loss": iou_l,
                "l1_loss": l1_l,
                "conf_loss": conf_l,
                "cls_loss": cls_l,
                "mem_loss": memory_anchor_loss,
                "num_fg": num_fg,
                "ttt_prob": current_ttt_prob 
            }
        else:
            return self.head(fpn_outs)

    def _calculate_supervised_memory_loss(self, aux_mem_logits, cls_targets, fg_masks):
        """
        RE-EVALUATED FULL-FIDELITY IMPLEMENTATION.
        
        Objective: Supervised Contrastive Prototype Alignment (InfoNCE logic).
        This forces the latent space to map noisy features to 'canonical' class prototypes.
        """
        # 1. FLATTEN MULTI-SCALE LOGITS
        # aux_mem_logits is a list of [B, HW_k, num_classes] from P3, P4, P5.
        # We reshape to [B*HW_k, num_classes] and concatenate to [B*Total_HW, num_classes]
        # This matches the YOLOX flattened anchor order.
        all_logits = torch.cat([l.reshape(-1, self.num_classes) for l in aux_mem_logits], dim=0)

        # 2. FLATTEN FOREGROUND MASK
        # fg_masks shape: [B, Total_Anchors]. Flatten to [B * Total_Anchors].
        flat_mask = fg_masks.view(-1)

        # 3. EXTRACT FOREGROUND SAMPLES
        # Extract the retrieval scores (similarity map) for pixels that actually contain objects.
        # Shape: [num_foreground_pixels, num_classes]
        fg_logits = all_logits[flat_mask]

        # 4. NUMERICAL & LOGICAL SAFETY GUARD
        # In batches with only background (common in BDD100K 'Undefined' scenes), 
        # fg_logits will be empty. We return a zero loss with a gradient tether.
        if fg_logits.shape[0] == 0:
            return all_logits.sum() * 0.0

        # 5. CONTRASTIVE SOFT-TARGET CROSS ENTROPY
        # cls_targets is [num_foreground_pixels, num_classes] from SimOTA.
        # It contains IoU-weighted labels (e.g., [0.92, 0, 0...] for a Car).
        
        # We use these as 'Soft Targets'. 
        # This is more robust than 'argmax' because it weights the prototype 
        # update by the detection confidence (IoU).
        # High IoU Car -> Strong pull to Car Prototype.
        # Low IoU Car -> Gentle pull to Car Prototype.
        
        # InfoNCE requires a valid probability distribution as target.
        # We ensure the soft-labels sum to 1.0 across classes.
        target_probs = F.normalize(cls_targets, p=1, dim=-1)

        # CrossEntropy on normalized similarities is mathematically 
        # identical to the InfoNCE Contrastive Loss used in SoTA models like CLIP/DINO.
        contrastive_loss = F.cross_entropy(fg_logits, target_probs)

        return contrastive_loss

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        # Inference mode: Always run TTT (1.0 probability)
        fpn_outs = self.backbone(x, ttt_prob=1.0)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)