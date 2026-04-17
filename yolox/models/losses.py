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
            # --- DFA-YOLO SWAWIoU Integration (Now Numerically Safe) ---
            
            # 1. 2D Gaussian Wasserstein Distance (k=2)
            w_dist_sq = (pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2 + \
                        ((pred_w - gt_w)/4)**2 + ((pred_h - gt_h)/4)**2
            
            # 2. Adaptive Area Penalty
            # Gamma is a scale-sensitive factor. Clamp min prevents exact 0.0.
            gamma = torch.clamp(torch.exp(-(area_g) / 1000.0), min=1e-6)
            
            # Area difference penalty
            p_area = (torch.abs(area_p - area_g) / torch.clamp(area_g, min=1e-6)) * gamma
            
            # 3. Augmented Wasserstein Distance 
            # Max clamp prevents exploding gradients if pred box is completely wrong
            w_dist_aug = w_dist_sq + 0.5 * torch.clamp(p_area, max=10.0)
            
            # 4. Adaptive Scaling Metric
            c_prime = 12.8 * (1 + gamma / 2)
            similarity = torch.exp(-w_dist_aug / c_prime)
            
            # 5. Dynamic Non-Monotonic Focusing (WIoU Logic)
            # Clamp detached IoU strictly to [0, 1] so (1 - IoU)**1.5 never produces Complex NaNs
            iou_detach = torch.clamp(iou.detach(), min=0.0, max=1.0)
            beta = 1.5
            outlierness = (1.0 - iou_detach) ** beta
            eta = outlierness / (torch.mean(outlierness) + 1e-6)
            
            loss = eta * (1.0 - similarity)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss