#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.2: SimOTA Dynamic Scale-Allocation Forensic Probe
# Patch v3.2.4: 4-Tuple Safe Unpack, Padded Dataloader & 0-Indexed Class Calibration

import os
import sys
import math
import argparse
from collections import defaultdict

# === WINDOWS UTF-8 ENCODING SHIELD ===
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
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

# Exact 0-indexed YOLOX category mapping:
CATEGORY_MAP = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "person",
    4: "rider",
    5: "bike",
    6: "motor",
    7: "traffic light",
    8: "traffic sign"
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


class SimOTATelemetryCollector:
    """
    Hooks directly into the SimOTA matching engine to record ground-truth
    to anchor-stride assignments without altering backpropagation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_gt = 0
        self.total_fg_anchors = 0
        self.stride_counts = {8: 0, 16: 0, 32: 0}
        self.class_stride_counts = defaultdict(lambda: {8: 0, 16: 0, 32: 0, "gt_count": 0, "total_anchors": 0})
        self.scale_stride_counts = {
            "small":  {8: 0, 16: 0, 32: 0, "gt_count": 0},
            "medium": {8: 0, 16: 0, 32: 0, "gt_count": 0},
            "large":  {8: 0, 16: 0, 32: 0, "gt_count": 0}
        }
        self.raw_instance_records = []

    def record_match(self, gt_classes, gt_bboxes, matched_gt_inds, fg_mask, expanded_strides):
        strides_matched = expanded_strides[0][fg_mask].cpu().numpy().astype(int)
        matched_classes = gt_classes[matched_gt_inds].cpu().numpy().astype(int)
        matched_gt_indices = matched_gt_inds.cpu().numpy()
        gt_bboxes_np = gt_bboxes.cpu().numpy()

        num_fg = len(strides_matched)
        self.total_fg_anchors += num_fg
        self.total_gt += len(gt_classes)

        # Track ground truth counts per class
        for c in gt_classes.cpu().numpy().astype(int):
            self.class_stride_counts[c]["gt_count"] += 1

        # Track area categorization for ground truth boxes
        gt_areas = gt_bboxes_np[:, 2] * gt_bboxes_np[:, 3]
        for area in gt_areas:
            if area < 32 * 32:
                self.scale_stride_counts["small"]["gt_count"] += 1
            elif area < 96 * 96:
                self.scale_stride_counts["medium"]["gt_count"] += 1
            else:
                self.scale_stride_counts["large"]["gt_count"] += 1

        # Track positive anchor assignments
        for s, c, gt_idx in zip(strides_matched, matched_classes, matched_gt_indices):
            self.stride_counts[s] += 1
            self.class_stride_counts[c][s] += 1
            self.class_stride_counts[c]["total_anchors"] += 1

            area = gt_areas[gt_idx]
            if area < 32 * 32:
                self.scale_stride_counts["small"][s] += 1
            elif area < 96 * 96:
                self.scale_stride_counts["medium"][s] += 1
            else:
                self.scale_stride_counts["large"][s] += 1

            self.raw_instance_records.append({
                "class_id": c,
                "class_name": CATEGORY_MAP.get(c, str(c)),
                "stride": s,
                "scale_level": "P3" if s == 8 else ("P4" if s == 16 else "P5"),
                "area": area,
                "w": gt_bboxes_np[gt_idx, 2],
                "h": gt_bboxes_np[gt_idx, 3],
            })


def instrument_head_for_telemetry(head, collector):
    """Wraps get_assignments_with_nwd to log allocation statistics."""
    orig_fn = head.get_assignments_with_nwd

    def tapped_assignments(
        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
        cls_preds, obj_preds
    ):
        result = orig_fn(
            batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
            bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
            cls_preds, obj_preds
        )
        gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = result

        if num_gt > 0 and num_fg > 0:
            collector.record_match(
                gt_classes=gt_classes,
                gt_bboxes=gt_bboxes_per_image,
                matched_gt_inds=matched_gt_inds,
                fg_mask=fg_mask,
                expanded_strides=expanded_strides
            )

        return result

    head.get_assignments_with_nwd = tapped_assignments
    return orig_fn


def generate_visual_reports(collector, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df_raw = pd.DataFrame(collector.raw_instance_records)
    if df_raw.empty:
        print(f"{Style.RED}No foreground assignments were captured to plot.{Style.RESET}")
        return

    csv_path = os.path.join(save_dir, "simota_raw_allocations.csv")
    df_raw.to_csv(csv_path, index=False)
    print(f"  • Raw instance data exported: {Style.BOLD}{csv_path}{Style.RESET}")

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#121212')

    palette = {'P3': '#3498db', 'P4': '#2ecc71', 'P5': '#e74c3c'}

    # Panel 1: Global Anchor Stride Distribution
    ax1 = axes[0, 0]
    ax1.set_facecolor('#1e1e1e')
    stride_labels = ['P3 (Stride 8)', 'P4 (Stride 16)', 'P5 (Stride 32)']
    stride_vals = [collector.stride_counts[8], collector.stride_counts[16], collector.stride_counts[32]]
    total_assigned = max(1, sum(stride_vals))
    pcts = [v / total_assigned * 100.0 for v in stride_vals]

    bars1 = ax1.bar(stride_labels, stride_vals, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.85, width=0.5)
    ax1.set_ylabel("Total Positive Anchors", color='white', fontsize=12)
    ax1.set_title(f"Global Scale Assignment Ratio (Total: {total_assigned:,})", color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    for bar, pct, cnt in zip(bars1, pcts, stride_vals):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, y + 10, f"{cnt:,}\n({pct:.1f}%)", ha='center', color='white', fontweight='bold')

    # Panel 2: Per-Class Assignment Proportions (Stacked Percentage)
    ax2 = axes[0, 1]
    ax2.set_facecolor('#1e1e1e')
    class_names = [CATEGORY_MAP[i] for i in range(9)]
    p3_pct, p4_pct, p5_pct = [], [], []

    for cid in range(9):
        tot = max(1, collector.class_stride_counts[cid]["total_anchors"])
        p3_pct.append(collector.class_stride_counts[cid][8] / tot * 100.0)
        p4_pct.append(collector.class_stride_counts[cid][16] / tot * 100.0)
        p5_pct.append(collector.class_stride_counts[cid][32] / tot * 100.0)

    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, p3_pct, label='P3 (Stride 8)', color='#3498db', alpha=0.85)
    ax2.barh(y_pos, p4_pct, left=p3_pct, label='P4 (Stride 16)', color='#2ecc71', alpha=0.85)
    ax2.barh(y_pos, p5_pct, left=np.array(p3_pct) + np.array(p4_pct), label='P5 (Stride 32)', color='#e74c3c', alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names, color='white', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Anchor Allocation Percentage (%)", color='white', fontsize=12)
    ax2.set_title("Per-Class Scale Distribution Matrix", color='white', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#2c3e50', edgecolor='none', labelcolor='white', loc='lower right')

    # Panel 3: Object Physical Area vs. Assigned Stride
    ax3 = axes[1, 0]
    ax3.set_facecolor('#1e1e1e')
    sample_df = df_raw.sample(min(2500, len(df_raw))) if len(df_raw) > 0 else df_raw
    if not sample_df.empty:
        sns.stripplot(
            data=sample_df, x='class_name', y='area', hue='scale_level',
            palette=palette, jitter=0.25, alpha=0.6, ax=ax3, dodge=True
        )
    ax3.set_yscale('log')
    ax3.set_ylabel("Bounding Box Area (px²)", color='white', fontsize=12)
    ax3.set_xlabel("Object Class", color='white', fontsize=12)
    ax3.set_title("Instance Area vs. Assigned Scale Level", color='white', fontsize=14, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
    ax3.legend(facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # Panel 4: Physical Area Slices
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1e1e1e')
    scale_cats = ['small', 'medium', 'large']
    sp3 = [collector.scale_stride_counts[k][8] for k in scale_cats]
    sp4 = [collector.scale_stride_counts[k][16] for k in scale_cats]
    sp5 = [collector.scale_stride_counts[k][32] for k in scale_cats]

    x_idx = np.arange(len(scale_cats))
    w = 0.25
    ax4.bar(x_idx - w, sp3, w, label='P3 (Stride 8)', color='#3498db', alpha=0.85)
    ax4.bar(x_idx,     sp4, w, label='P4 (Stride 16)', color='#2ecc71', alpha=0.85)
    ax4.bar(x_idx + w, sp5, w, label='P5 (Stride 32)', color='#e74c3c', alpha=0.85)

    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(['Small (<32²)', 'Medium (32²-96²)', 'Large (>96²)'], color='white', fontsize=11, fontweight='bold')
    ax4.set_ylabel("Matched Anchor Count", color='white', fontsize=12)
    ax4.set_title("Physical Scale Slices vs. Anchor Allocation", color='white', fontsize=14, fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.legend(facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "SIMOTA_SCALE_ALLOCATION_AUDIT.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    plt.close()
    print(f"  • Visual audit dashboard generated: {Style.BOLD}{plot_path}{Style.RESET}")


def print_forensic_table(collector):
    print(f"\n{Style.BOLD}{Style.WHITE}===================================================================================================={Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}             SimOTA DYNAMIC ANCHOR ALLOCATION FORENSIC MATRIX (TDE-YOLOX v3.2)                     {Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}===================================================================================================={Style.RESET}")

    total_anchors = max(1, collector.total_fg_anchors)
    p3_total = collector.stride_counts[8]
    p4_total = collector.stride_counts[16]
    p5_total = collector.stride_counts[32]

    print(f"  • Total Ground-Truth Instances Evaluated: {Style.BOLD}{collector.total_gt:,}{Style.RESET}")
    print(f"  • Total Positive Anchors Allocated:       {Style.BOLD}{total_anchors:,}{Style.RESET}")
    print(f"  • Global Scale Allocation:")
    print(f"      - P3 (Stride 8,  80x80): {Style.CYAN}{p3_total:>8,}{Style.RESET} ({p3_total / total_anchors * 100.0:>5.1f}%) [Pool: 6,400 anchors / 76.2%]")
    print(f"      - P4 (Stride 16, 40x40): {Style.GREEN}{p4_total:>8,}{Style.RESET} ({p4_total / total_anchors * 100.0:>5.1f}%) [Pool: 1,600 anchors / 19.0%]")
    print(f"      - P5 (Stride 32, 20x20): {Style.RED}{p5_total:>8,}{Style.RESET} ({p5_total / total_anchors * 100.0:>5.1f}%) [Pool:   400 anchors /  4.8%]")
    print("-" * 100)

    # Class Table
    print(f"{'Class Name':<15} | {'GT Count':<8} | {'Total FG':<9} | {'Avg k':<6} | {'P3 (80x80)':<12} | {'P4 (40x40)':<12} | {'P5 (20x20)':<12} | {'P5 Status':<12}")
    print("-" * 100)

    for cid in range(9):
        cname = CATEGORY_MAP[cid]
        data = collector.class_stride_counts[cid]
        gt_cnt = data["gt_count"]
        tot_fg = data["total_anchors"]
        avg_k = (tot_fg / gt_cnt) if gt_cnt > 0 else 0.0

        p3_cnt = data[8]
        p4_cnt = data[16]
        p5_cnt = data[32]

        p3_p = (p3_cnt / tot_fg * 100.0) if tot_fg > 0 else 0.0
        p4_p = (p4_cnt / tot_fg * 100.0) if tot_fg > 0 else 0.0
        p5_p = (p5_cnt / tot_fg * 100.0) if tot_fg > 0 else 0.0

        p5_status = f"{Style.RED}STARVED (0.0%){Style.RESET}" if p5_p < 0.5 else (
            f"{Style.YELLOW}LOW ({p5_p:.1f}%){Style.RESET}" if p5_p < 5.0 else f"{Style.GREEN}ACTIVE ({p5_p:.1f}%){Style.RESET}"
        )

        p3_str = f"{p3_cnt:>5} ({p3_p:>4.1f}%)"
        p4_str = f"{p4_cnt:>5} ({p4_p:>4.1f}%)"
        p5_str = f"{p5_cnt:>5} ({p5_p:>4.1f}%)"

        print(f"{cname:<15} | {gt_cnt:<8} | {tot_fg:<9} | {avg_k:<6.2f} | {p3_str:<12} | {p4_str:<12} | {p5_str:<12} | {p5_status}")

    print("-" * 100)

    # Physical Box Area Breakdown
    print(f"{'Physical Area':<15} | {'GT Count':<8} | {'Total FG':<9} | {'Avg k':<6} | {'P3 %':<12} | {'P4 %':<12} | {'P5 %':<12}")
    print("-" * 100)
    for cat, label in [("small", "Small (<32²)"), ("medium", "Medium (32²-96²)"), ("large", "Large (>96²)")]:
        sd = collector.scale_stride_counts[cat]
        gt_cnt = sd["gt_count"]
        tot_fg = sd[8] + sd[16] + sd[32]
        avg_k = (tot_fg / gt_cnt) if gt_cnt > 0 else 0.0
        p3_p = (sd[8] / tot_fg * 100.0) if tot_fg > 0 else 0.0
        p4_p = (sd[16] / tot_fg * 100.0) if tot_fg > 0 else 0.0
        p5_p = (sd[32] / tot_fg * 100.0) if tot_fg > 0 else 0.0
        print(f"{label:<15} | {gt_cnt:<8} | {tot_fg:<9} | {avg_k:<6.2f} | {p3_p:>10.1f}% | {p4_p:>10.1f}% | {p5_p:>10.1f}%")

    print(f"{Style.BOLD}{Style.WHITE}===================================================================================================={Style.RESET}\n")


def main():
    parser = argparse.ArgumentParser("SimOTA Scale Allocation Forensic Probe")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Path to experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Dataset split to evaluate")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to evaluate (0 for full dataset)")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="./probe_results/simota_audit", help="Directory for output")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    exp = get_exp(args.exp_file, None)
    
    print(f"\n{Style.BOLD}Initializing SimOTA Allocation Probe on [{args.split.upper()}] dataset...{Style.RESET}")
    print(f"Loading checkpoint: {args.ckpt}")

    model = exp.get_model()
    model.to(device)
    
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt_state = ckpt["model"]
    elif isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"] is not None:
        ckpt_state = ckpt["ema"]
    else:
        ckpt_state = ckpt
        
    model = load_ckpt(model, ckpt_state)
    
    # Put model into training mode so SimOTA assignment executes
    model.train()

    # Build an unaugmented, padded dataloader to prevent tuple unpacking crashes
    target_ann = exp.val_ann if args.split == "val" else exp.train_ann
    target_data_dir = getattr(exp, "eval_data_dir", exp.data_dir)

    dataset = COCODataset(
        data_dir=target_data_dir,
        json_file=target_ann,
        name="",
        img_size=exp.input_size,
        preproc=TrainTransform(max_labels=120, flip_prob=0.0, hsv_prob=0.0),
        cache=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    collector = SimOTATelemetryCollector()

    m = model.module if hasattr(model, "module") else model
    head = m.head.module if hasattr(m.head, "module") else m.head
    orig_fn = instrument_head_for_telemetry(head, collector)

    total_iters = len(dataloader) if args.num_batches <= 0 else min(args.num_batches, len(dataloader))
    print(f"Running assignment audit across {total_iters} batches ({total_iters * args.batch_size:,} images)...")

    data_iter = iter(dataloader)
    for _ in tqdm(range(total_iters), desc="SimOTA Extraction"):
        batch = next(data_iter)
        # Safe extraction: ignore trailing ID/metadata elements
        inps = batch[0].to(device).float()
        targets = batch[1].to(device).float()

        inps, targets = exp.preprocess(inps, targets, exp.input_size)

        # Forward pass runs SimOTA assignment inside get_losses_with_targets
        with torch.no_grad():
            _ = model(inps, targets)

    # Restore original function pointer
    head.get_assignments_with_nwd = orig_fn

    # Output Telemetry Table & Diagnostic Charts
    print_forensic_table(collector)
    generate_visual_reports(collector, args.save_dir)


if __name__ == "__main__":
    main()