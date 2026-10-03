#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.3: Large-Object Cost Matrix & Confidence Cliff Inspector

import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

# === WINDOWS UTF-8 ENCODING SHIELD ===
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.exp import get_exp
from yolox.utils import load_ckpt
from yolox.data import COCODataset, TrainTransform

CATEGORY_MAP = {
    0: "car", 1: "bus", 2: "truck", 3: "person", 
    4: "rider", 5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign"
}

class Style:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"


class LargeObjectCostAuditor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.large_instance_costs = []
        self.confidence_records = []

    def record_cost_decomposition(
        self, gt_classes, gt_bboxes, strides_in_boxes, pair_wise_ious, cost_matrix, dynamic_ks, matched_gt_inds, fg_mask
    ):
        gt_areas = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).cpu().numpy()
        gt_classes_np = gt_classes.cpu().numpy().astype(int)

        # Large objects: Area >= 96^2 = 9216 px^2
        large_gt_indices = np.where(gt_areas >= 9216.0)[0]
        if len(large_gt_indices) == 0:
            return

        for gt_idx in large_gt_indices:
            cid = gt_classes_np[gt_idx]
            cname = CATEGORY_MAP.get(cid, str(cid))
            area = gt_areas[gt_idx]

            cost_gt = cost_matrix[gt_idx].cpu().numpy()
            iou_gt = pair_wise_ious[gt_idx].cpu().numpy()
            strides = strides_in_boxes.cpu().numpy()

            for s in [8, 16, 32]:
                mask_s = (strides == s)
                if mask_s.any():
                    mean_cost = float(cost_gt[mask_s].mean())
                    min_cost = float(cost_gt[mask_s].min())
                    mean_iou = float(iou_gt[mask_s].mean())
                    count_anchors = int(mask_s.sum())
                else:
                    mean_cost = min_cost = mean_iou = count_anchors = 0.0

                self.large_instance_costs.append({
                    "class": cname,
                    "area": area,
                    "stride": s,
                    "level": "P3" if s == 8 else ("P4" if s == 16 else "P5"),
                    "mean_cost": mean_cost,
                    "min_cost": min_cost,
                    "mean_iou": mean_iou,
                    "available_anchors": count_anchors
                })

    def record_confidence_logits(self, outputs, num_classes):
        """Extracts pre-NMS raw scores safely to inspect the 0.001 cutoff boundary."""
        obj_preds = outputs[:, :, 4:5].sigmoid()
        cls_preds = outputs[:, :, 5:].sigmoid()
        
        # Use .contiguous().reshape to eliminate PyTorch non-contiguous view error
        scores = (obj_preds * cls_preds).contiguous().reshape(-1, num_classes)

        # Extract Bus (1) and Truck (2) scores
        bus_scores = scores[:, 1].detach().cpu().numpy()
        truck_scores = scores[:, 2].detach().cpu().numpy()

        b_window = bus_scores[(bus_scores >= 0.0001) & (bus_scores <= 0.0100)]
        t_window = truck_scores[(truck_scores >= 0.0001) & (truck_scores <= 0.0100)]

        for s in b_window[:100]:
            self.confidence_records.append({"class": "bus", "score": float(s)})
        for s in t_window[:100]:
            self.confidence_records.append({"class": "truck", "score": float(s)})


def instrument_head_for_cost_audit(head, auditor):
    """Hooks directly into SimOTA and loss pipeline to capture cost and predictions."""
    orig_losses_fn = head.get_losses_with_targets
    orig_matching_fn = head.simota_matching_level_aware
    orig_assign_fn = head.get_assignments_with_nwd

    def tapped_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        # Capture raw predictions with memory fusion directly from native head
        auditor.record_confidence_logits(outputs, head.num_classes)
        return orig_losses_fn(imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype)

    def tapped_assignments(batch_idx, num_gt, gt_bboxes_per_image, gt_classes, *args, **kwargs):
        head._curr_gt_bboxes = gt_bboxes_per_image
        return orig_assign_fn(batch_idx, num_gt, gt_bboxes_per_image, gt_classes, *args, **kwargs)

    def tapped_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask, dynamic_ks, num_in_boxes_anchor, strides_in_boxes):
        res = orig_matching_fn(cost, pair_wise_ious, gt_classes, num_gt, fg_mask, dynamic_ks, num_in_boxes_anchor, strides_in_boxes)
        # UNPACK EXACTLY 4 VALUES FROM simota_matching_level_aware:
        num_fg, gt_matched_classes, pred_ious, matched_gt_inds = res

        gt_bboxes = getattr(head, "_curr_gt_bboxes", None)
        if gt_bboxes is not None and num_gt > 0:
            auditor.record_cost_decomposition(
                gt_classes=gt_classes,
                gt_bboxes=gt_bboxes,
                strides_in_boxes=strides_in_boxes,
                pair_wise_ious=pair_wise_ious,
                cost_matrix=cost,
                dynamic_ks=dynamic_ks,
                matched_gt_inds=matched_gt_inds,
                fg_mask=fg_mask
            )
        return res

    head.get_losses_with_targets = tapped_losses
    head.get_assignments_with_nwd = tapped_assignments
    head.simota_matching_level_aware = tapped_matching

    return orig_losses_fn, orig_assign_fn, orig_matching_fn


def run_large_object_audit(exp, ckpt_path, ckpt_tag, num_batches=40, device="cuda"):
    print(f"\n{Style.BOLD}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}   AUDITING LARGE OBJECT SIMOTA COSTS: [{ckpt_tag}] -> {ckpt_path}{Style.RESET}")
    print(f"{Style.BOLD}================================================================================{Style.RESET}")

    model = exp.get_model()
    model.to(device).half().train()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = ckpt.get("ema", ckpt.get("model", ckpt))
    model = load_ckpt(model, ckpt_state)

    dataset = COCODataset(
        data_dir=exp.data_dir,
        json_file=exp.val_ann,
        name="",
        img_size=exp.input_size,
        preproc=TrainTransform(max_labels=120, flip_prob=0.0, hsv_prob=0.0),
        cache=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    m = model.module if hasattr(model, "module") else model
    head = m.head.module if hasattr(m.head, "module") else m.head

    auditor = LargeObjectCostAuditor()
    orig_losses_fn, orig_assign_fn, orig_matching_fn = instrument_head_for_cost_audit(head, auditor)

    data_iter = iter(dataloader)
    total_iters = min(num_batches, len(dataloader))

    with torch.no_grad():
        for _ in tqdm(range(total_iters), desc=f"Evaluating Large Instances [{ckpt_tag}]"):
            batch = next(data_iter)
            inps = batch[0].to(device).half()
            targets = batch[1].to(device).half()
            inps, targets = exp.preprocess(inps, targets, exp.input_size)
            
            # Forward pass natively triggers SimOTA and prediction extraction
            _ = model(inps, targets)

    # Clean un-hooking
    head.get_losses_with_targets = orig_losses_fn
    head.get_assignments_with_nwd = orig_assign_fn
    head.simota_matching_level_aware = orig_matching_fn

    return auditor


def main():
    parser = argparse.ArgumentParser("Large Object SimOTA Cost & Threshold Inspector")
    parser.add_argument("-f", "--exp_file", type=str, default="exps/default/tde_yolox_s.py")
    parser.add_argument("-c36", "--ckpt_36", type=str, default="YOLOX_outputs/tde_yolox_s/epoch_35_ckpt.pth")
    parser.add_argument("-c49", "--ckpt_49", type=str, default="YOLOX_outputs/tde_yolox_s/epoch_49_ckpt.pth")
    parser.add_argument("--save_dir", type=str, default="./probe_results/large_objects")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    exp = get_exp(args.exp_file, None)

    audit_36 = run_large_object_audit(exp, args.ckpt_36, "Epoch_35")
    audit_49 = run_large_object_audit(exp, args.ckpt_49, "Epoch_49")

    df_costs_36 = pd.DataFrame(audit_36.large_instance_costs)
    df_costs_49 = pd.DataFrame(audit_49.large_instance_costs)
    df_conf_36 = pd.DataFrame(audit_36.confidence_records)
    df_conf_49 = pd.DataFrame(audit_49.confidence_records)

    print(f"\n{Style.BOLD}===================================================================================================={Style.RESET}")
    print(f"{Style.BOLD}               LARGE OBJECT SIMOTA COST BREAKDOWN (STRIDE 16 vs. STRIDE 32)                         {Style.RESET}")
    print(f"{Style.BOLD}===================================================================================================={Style.RESET}")

    for tag, df in [("Epoch 35", df_costs_36), ("Epoch 49", df_costs_49)]:
        if df.empty:
            continue
        print(f"\n{Style.CYAN}--- {tag} Mean Cost & Overlap on Large Objects (Area >= 96² px) ---{Style.RESET}")
        summary = df.groupby(["class", "level"])[["min_cost", "mean_iou", "available_anchors"]].mean().reset_index()
        for cname in ["bus", "truck", "car"]:
            c_df = summary[summary["class"] == cname]
            if not c_df.empty:
                print(f"  • Class: {Style.BOLD}{cname.upper()}{Style.RESET}")
                for _, row in c_df.iterrows():
                    print(f"      - {row['level']:<4} -> Min Cost: {row['min_cost']:.4f} | Mean IoU: {row['mean_iou']:.4f} | Cand Anchors: {row['available_anchors']:.1f}")

    # Plotting Confidence Cliff
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#121212')

    for idx, cname in enumerate(["bus", "truck"]):
        ax = axes[idx]
        ax.set_facecolor('#1e1e1e')
        s36 = df_conf_36[df_conf_36["class"] == cname]["score"] if not df_conf_36.empty else []
        s49 = df_conf_49[df_conf_49["class"] == cname]["score"] if not df_conf_49.empty else []

        if len(s36) > 0 and len(s49) > 0:
            sns.kdeplot(s36, ax=ax, label="Epoch 35", color="cyan", fill=True, alpha=0.3)
            sns.kdeplot(s49, ax=ax, label="Epoch 49", color="red", fill=True, alpha=0.3)
            ax.axvline(x=0.001, color="yellow", linestyle="--", label="--conf 0.001 Cutoff")
            ax.set_title(f"{cname.upper()} Score Density near 0.001 Threshold", color="white", fontsize=13, fontweight="bold")
            ax.set_xlabel("Prediction Confidence", color="white")
            ax.set_ylabel("Density", color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#222", edgecolor="none", labelcolor="white")

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "LARGE_OBJECT_CONFIDENCE_CLIFF.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=200)
    plt.close()
    print(f"\n{Style.GREEN}✔ Large-Object Diagnostic Plot saved to: {Style.BOLD}{plot_path}{Style.RESET}")


if __name__ == "__main__":
    main()