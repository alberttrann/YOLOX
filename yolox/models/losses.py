import torch
import torch.nn as nn
import math

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        # THE CRITICAL FIX: Force FP32 Promotion
        # Bounding box geometry (Area = W * H) easily exceeds 65,504 (FP16 Max).
        # Casting to float32 prevents Inf overflows and 0.0 underflows in SWAWIoU.
        pred = pred.view(-1, 4).float()
        target = target.view(-1, 4).float()
        
        pred_cx, pred_cy, pred_w, pred_h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        gt_cx, gt_cy, gt_w, gt_h = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Calculate standard IoU geometry
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        area_p = pred_w * pred_h
        area_g = gt_w * gt_h

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
            
        elif self.loss_type == "swawiou":
            # ----------------------------------------------------------------
            # TDE-YOLOX SWAWIoU: Wasserstein-Adaptive WIoU Loss
            # Based on DFA-YOLO SWAWIoU (Xie & Cheng, 2026), restructured for
            # TTT stability and adverse-weather BDD100K training.
            # ----------------------------------------------------------------

            # --- STEP 1: Base IoU Loss (primary gradient signal) ---
            # WIoU-multiplicative structure: R amplifies ON TOP of (1-IoU),
            # so even IoU=0 boxes receive a correction gradient.
            # NOT similarity-based to avoid gradient starvation.
            loss_iou = 1.0 - iou

            # --- STEP 2: Enclosing Box Diagonal (DETACHED) ---
            # MUST be detached: without detach, the optimizer can minimize loss
            # by expanding the predicted box to trivially enlarge c_diag_sq.
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2),
                (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2),
                (target[:, :2] + target[:, 2:] / 2)
            )
            cw_ch = (c_br - c_tl).clamp(min=1e-6)
            c_diag_sq = (cw_ch[:, 0]**2 + cw_ch[:, 1]**2).detach()

            # --- STEP 3: Gaussian Wasserstein Distance W² (DFA-YOLO Eq 13, k=2) ---
            # Formula: W² = Δcx² + Δcy² + (Δw/(2k))² + (Δh/(2k))²
            # k=2 → divides by (2*2)² = 16.
            # CRITICAL: ((Δw)/4)² = Δw²/16. Writing Δw²/4 is a 4× error.
            w_dist_sq = (
                (pred_cx - gt_cx)**2 +
                (pred_cy - gt_cy)**2 +
                (pred_w - gt_w)**2 / 16.0 +
                (pred_h - gt_h)**2 / 16.0
            )

            # --- STEP 4: Normalize to Dimensionless Ratio ---
            # Resolves the pixel² vs. ratio unit mismatch from the original version.
            # w_dist_norm ∈ [0, ~1.0] for all realistic box configurations on 640×640.
            # Proof: at absolute worst (boxes at image corners), w_dist_norm ≈ 0.997.
            w_dist_norm = w_dist_sq / c_diag_sq

            # --- STEP 5: Scale-Sensitive Factor γ (DFA-YOLO Eq 14) ---
            # γ → 1 for objects < 32×32px (area < ~1000), γ → 0 for large objects.
            # Effect: area penalty and c_prime adjustment are ONLY active for small objects.
            # DETACHED: γ is a hyper-weighting coefficient, not an optimization target.
            # area_g comes from GT (no grad graph anyway), but explicit detach signals intent.
            gamma = torch.clamp(torch.exp(-area_g / 1000.0), min=1e-6).detach()

            # --- STEP 6: Bounded Area Penalty (DFA-YOLO Eq 15, restructured) ---
            # Original: |area_p - area_g| / area_g * γ — unbounded ratio, gradient dies at clamp.
            # Restructured: (1 - min/max) * γ — naturally in [0, γ] ⊆ [0, 1], no clamp needed.
            # Gradient through p_area w.r.t. pred_w:
            #   Under-predict (area_p < area_g): ∂p_area/∂pred_w = -γ/area_g < 0 → grow correctly
            #   Over-predict  (area_p > area_g): ∂p_area/∂pred_w = γ*area_g/area_p² > 0 → shrink correctly
            area_min = torch.min(area_p, area_g)
            area_max = torch.max(area_p, area_g).clamp(min=1e-6)
            p_area = (1.0 - (area_min / area_max)) * gamma

            # --- STEP 7: Augmented Normalized Wasserstein Distance (DFA-YOLO Eq 16, λ=0.5) ---
            # Both w_dist_norm and p_area are dimensionless and bounded.
            # w_aug ∈ [0, ~1.5] — well-bounded, no clamp needed.
            w_aug = w_dist_norm + 0.5 * p_area

            # --- STEP 8: Adaptive Scaling Constant (DFA-YOLO Eq 17, C₀ adapted for normalized space) ---
            # Paper uses C₀=12.8 in pixel² space. After normalization, c_prime = 1.0 + γ/2 ∈ [1.0, 1.5].
            # For large objects (γ→0): c_prime=1.0, full Wasserstein sensitivity.
            # For small objects (γ→1): c_prime=1.5, smoothed penalty (avoids over-penalizing noisy small boxes).
            c_prime = 1.0 + (gamma / 2.0)

            # --- STEP 9: Wasserstein Multiplier R_swa (WIoU v3 exponential structure) ---
            # Uses sqrt for sub-linear growth: exp(√x) vs exp(x) at x=1.5 gives 3.40 vs 4.48.
            # The sqrt gradient profile (strong near 0, graceful decay) matches the Wasserstein
            # distribution's geometric properties.
            # NOT detached: p_area and w_dist_norm carry valid gradient information about
            # area mismatch and center/size displacement. Detaching loses the Wasserstein geometry signal.
            # Max R_swa = exp(√(1.5/1.0)) ≈ 3.40 — bounded amplification, safe for TTT norm gradients.
            R_swa = torch.exp(torch.sqrt(w_aug / c_prime + 1e-8))

            # --- STEP 10: Wasserstein-Weighted IoU Loss ---
            # R_swa amplifies the gradient for high-Wasserstein-distance predictions.
            # (1-IoU) guarantees non-zero gradient for all boxes including IoU=0.
            loss_base = R_swa * loss_iou

            # --- STEP 11: WIoU v3 Non-Monotonic Focusing (TDE-adapted) ---
            # η = β_out * exp(1 - β_out), where β_out = per-box loss / batch mean loss.
            # This function peaks at β_out=1 (average difficulty) and DECREASES for both extremes:
            #   β_out=0.1 (easy):    η = 0.1 * exp(0.9) ≈ 0.25  ← suppress: near-perfect, low signal value
            #   β_out=1.0 (average): η = 1.0 * exp(0.0) = 1.00  ← full gradient
            #   β_out=2.0 (hard):    η = 2.0 * exp(-1.0) ≈ 0.74 ← mild suppression
            #   β_out=5.0 (extreme): η = 5.0 * exp(-4.0) ≈ 0.09 ← strong suppression
            # TDE CRITICAL: fog-obscured objects with IoU≈0 (β_out>>1) are the worst-quality
            # label assignments. Monotonic focusing gives them massive gradient weight, destabilizing
            # TTT norm parameters adapted to that image. Non-monotonic suppresses them.
            # DETACHED: prevents cross-box coupling through the batch mean in functional_call context.
            iou_detach = torch.clamp(iou.detach(), min=0.0, max=1.0)
            loss_iou_detach = 1.0 - iou_detach
            beta_out = loss_iou_detach / (torch.mean(loss_iou_detach) + 1e-6)
            eta = (beta_out * torch.exp(1.0 - beta_out)).detach()

            # --- STEP 12: Final Loss ---
            # Expected eta ≈ 1.0 (batch-normalized), so total loss magnitude
            # is comparable to plain IoU loss — no need to retune loss weights.
            loss = eta * loss_base
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss