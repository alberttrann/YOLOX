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
        self.no_aug_epochs = 15 # Sync this with your Exp file
        
        # --- ANNEALING SCHEDULE CONFIG ---
        self.warmup_end = 10
        self.ramp_end = 65 # Ramps up until the exact moment Mosaic turns off
        
        # Probability Schedule
        self.prob_start = 0.1
        self.prob_end = 0.7 
        
        # Noise Schedule (Task Hardening)
        self.noise_start_scale = 0.5 
        self.noise_end_scale = 2.0   

    def set_meta_training_state(self, epoch, max_epochs):
        """Called by Trainer at epoch start to drive the adaptation schedule."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def _get_ttt_schedule(self):
        """
        Calculates Stochastic Meta-Learning Probability AND Dynamic Noise Scale.
        Implements the Marathon Strategy.
        """
        if not self.training:
            return 1.0, 1.0 # Inference: Always TTT, Standard Noise
            
        # 1. Warmup Phase (Easy)
        if self.current_epoch < self.warmup_end:
            return self.prob_start, self.noise_start_scale
            
        # 2. The 'No-Aug' Fine-Tuning Phase (The Bridge Phase)
        # CRITICAL: Keep TTT active (0.5) to maintain feature distribution.
        # Drop noise (0.5) to allow high-precision regression learning.
        if self.current_epoch >= self.max_epochs - self.no_aug_epochs:
            return 0.5, 0.5 
            
        # 3. The Curriculum Ramp (Increasing Difficulty)
        # Scale progress from WarmupEnd -> RampEnd (Epoch 65)
        # Clamp between 0 and 1 to be safe
        denom = self.ramp_end - self.warmup_end
        if denom <= 0: denom = 1 # Safety
        
        progress = (self.current_epoch - self.warmup_end) / denom
        progress = max(0.0, min(1.0, progress))
        
        curr_prob = self.prob_start + (self.prob_end - self.prob_start) * progress
        curr_noise = self.noise_start_scale + (self.noise_end_scale - self.noise_start_scale) * progress
        
        return curr_prob, curr_noise

    def forward(self, x, targets=None):
        # Get current curriculum values
        current_ttt_prob, current_noise_scale = self._get_ttt_schedule()
        
        # Pass both prob and noise scale to backbone
        # NOTE: You must update CSPDarknet.forward to accept ttt_noise_scale!
        fpn_outs = self.backbone(x, ttt_prob=current_ttt_prob, ttt_noise_scale=current_noise_scale)

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
        The 'Holy Grail' Supervised Identity Loss.
        Ensures the Memory retrieval queries are supervised by GT labels.
        """
        # aux_mem_logits: list of [B, HW, num_classes]
        # cls_targets: [Total_FG_Pixels, num_classes] - The targets assigned by SimOTA
        # fg_masks: [B, Total_Pixels] - Boolean mask of which pixels are objects
        
        # 1. Flatten all scales into one long sequence
        all_logits = torch.cat([l.view(-1, self.num_classes) for l in aux_mem_logits], dim=0)
        
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