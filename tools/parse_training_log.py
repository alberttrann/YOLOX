#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
TDE-YOLOX Comprehensive Training & Evaluation Log Parser
Extracts:
  - Epoch-by-epoch validation trajectory: AP (all, 50, 75, s, m, l), AR (all, s, m, l)
  - Full per-class AP breakdown across all 9 categories
  - Iteration-averaged training loss curves (total, iou, nwd, conf, cls, mem, sem, phase, etc.)
  - Automated Saturation & Plateau Diagnostics (identifies the exact wall and decay rates)
  - Exports structured CSVs and multi-panel diagnostic PNG plots
"""

import argparse
import csv
import os
import re
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser("TDE-YOLOX Log Parser & Plateau Analyzer")
    parser.add_argument(
        "-f",
        "--file",
        default="YOLOX_outputs/tde_yolox_s/train_log.txt",
        type=str,
        help="path to train_log.txt",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default="YOLOX_outputs/tde_yolox_s/parsed_analytics",
        type=str,
        help="output directory for CSVs and plots",
    )
    parser.add_argument(
        "--window",
        default=3,
        type=int,
        help="plateau detection rolling window in epochs",
    )
    return parser.parse_args()


def parse_log(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at: {log_path}")

    print(f"\nParsing log file: {log_path} ...\n")

    # Data structures
    train_iters = defaultdict(lambda: defaultdict(list))
    eval_epochs = {}

    current_epoch = 1
    in_eval_block = False
    in_class_ap = False
    in_class_ar = False
    temp_eval = {}
    temp_class_ap = {}
    temp_class_ar = {}

    # Regex patterns
    iter_pattern = re.compile(
        r"epoch:\s*\[(\d+)/\d+\]\[(\d+)/\d+\].*?total_loss:\s*([\d\.]+)"
    )
    stat_pattern = re.compile(r"(\w+):\s*([e\d\.\-\+]+)(?:s|Mb)?")
    start_epoch_pattern = re.compile(r"---> start train epoch(\d+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 1. Detect start of new epoch
            m_start = start_epoch_pattern.search(line)
            if m_start:
                # If we just finished an evaluation block, save it for the completed epoch
                if temp_eval and current_epoch not in eval_epochs:
                    temp_eval["class_ap"] = temp_class_ap
                    temp_eval["class_ar"] = temp_class_ar
                    eval_epochs[current_epoch] = temp_eval
                    temp_eval = {}
                    temp_class_ap = {}
                    temp_class_ar = {}
                current_epoch = int(m_start.group(1))
                in_eval_block = False
                in_class_ap = False
                in_class_ar = False
                continue

            # 2. Extract Training Iteration Stats
            m_iter = iter_pattern.search(line)
            if m_iter:
                ep = int(m_iter.group(1))
                current_epoch = ep
                # Parse all key-value metrics on the line
                pairs = stat_pattern.findall(line)
                for k, v in pairs:
                    try:
                        val = float(v)
                        train_iters[ep][k].append(val)
                    except ValueError:
                        continue
                continue

            # 3. Detect Evaluation Phase
            if (
                "Average forward time:" in line
                or "Average Precision  (AP) @[ IoU=0.50:0.95" in line
            ):
                in_eval_block = True
                in_class_ap = False
                in_class_ar = False

            if in_eval_block:
                # Extract COCO AP/AR metrics
                if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all" in line:
                    temp_eval["ap"] = float(line.split("=")[-1].strip())
                elif "Average Precision  (AP) @[ IoU=0.50      | area=   all" in line:
                    temp_eval["ap50"] = float(line.split("=")[-1].strip())
                elif "Average Precision  (AP) @[ IoU=0.75      | area=   all" in line:
                    temp_eval["ap75"] = float(line.split("=")[-1].strip())
                elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small" in line:
                    temp_eval["ap_s"] = float(line.split("=")[-1].strip())
                elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium" in line:
                    temp_eval["ap_m"] = float(line.split("=")[-1].strip())
                elif "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large" in line:
                    temp_eval["ap_l"] = float(line.split("=")[-1].strip())
                elif "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all" in line:
                    if "maxDets=100" in line:
                        temp_eval["ar"] = float(line.split("=")[-1].strip())
                elif "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small" in line:
                    temp_eval["ar_s"] = float(line.split("=")[-1].strip())
                elif "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium" in line:
                    temp_eval["ar_m"] = float(line.split("=")[-1].strip())
                elif "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large" in line:
                    temp_eval["ar_l"] = float(line.split("=")[-1].strip())

                # Per-class AP table
                if "per class AP:" in line:
                    in_class_ap = True
                    in_class_ar = False
                    continue
                if "per class AR:" in line:
                    in_class_ar = True
                    in_class_ap = False
                    continue

                if in_class_ap and line.strip().startswith("|") and not line.strip().startswith("|:"):
                    parts = [p.strip() for p in line.strip().split("|") if p.strip()]
                    # Format: | class | AP | class | AP | class | AP |
                    for idx in range(0, len(parts), 2):
                        if idx + 1 < len(parts):
                            c_name, c_val = parts[idx], parts[idx + 1]
                            if c_name not in ["class", "AP"]:
                                try:
                                    temp_class_ap[c_name] = float(c_val)
                                except ValueError:
                                    temp_class_ap[c_name] = float("nan")

                if in_class_ar and line.strip().startswith("|") and not line.strip().startswith("|:"):
                    parts = [p.strip() for p in line.strip().split("|") if p.strip()]
                    for idx in range(0, len(parts), 2):
                        if idx + 1 < len(parts):
                            c_name, c_val = parts[idx], parts[idx + 1]
                            if c_name not in ["class", "AR"]:
                                try:
                                    temp_class_ar[c_name] = float(c_val)
                                except ValueError:
                                    temp_class_ar[c_name] = float("nan")

    # Flush final evaluation block if present at EOF
    if temp_eval and current_epoch not in eval_epochs:
        temp_eval["class_ap"] = temp_class_ap
        temp_eval["class_ar"] = temp_class_ar
        eval_epochs[current_epoch] = temp_eval

    # Compute epoch-averaged training losses
    epoch_losses = {}
    for ep, metrics in sorted(train_iters.items()):
        epoch_losses[ep] = {k: sum(v) / max(1, len(v)) for k, v in metrics.items()}

    return epoch_losses, eval_epochs


def print_evaluation_summary(eval_epochs, window=3):
    print("=" * 105)
    print(" " * 32 + "EPOCH-BY-EPOCH EVALUATION TRAJECTORY")
    print("=" * 105)
    header = f"{'Epoch':^6} │ {'AP (all)':^8} │ {'AP50':^6} │ {'AP75':^6} │ {'AP_s':^6} │ {'AP_m':^6} │ {'AP_l':^6} │ {'AR':^6} │ {'Δ AP':^8} │ Status"
    print(header)
    print("─" * 105)

    sorted_eps = sorted(eval_epochs.keys())
    prev_ap = None
    peak_ap = -1.0
    peak_ep = None

    for ep in sorted_eps:
        d = eval_epochs[ep]
        ap = d.get("ap", 0.0)
        ap50 = d.get("ap50", 0.0)
        ap75 = d.get("ap75", 0.0)
        ap_s = d.get("ap_s", 0.0)
        ap_m = d.get("ap_m", 0.0)
        ap_l = d.get("ap_l", 0.0)
        ar = d.get("ar", 0.0)

        delta_str = "  --   "
        if prev_ap is not None:
            delta = ap - prev_ap
            delta_str = f"{delta:+0.4f}"

        status = ""
        if ap > peak_ap:
            peak_ap = ap
            peak_ep = ep
            status = "★ NEW PEAK"
        elif prev_ap is not None and abs(ap - prev_ap) < 0.002:
            status = "≈ FLATLINE"
        elif prev_ap is not None and ap < prev_ap:
            status = "▼ DECAY"

        print(
            f"{ep:^6} │ {ap:^8.4f} │ {ap50:^6.3f} │ {ap75:^6.3f} │ {ap_s:^6.3f} │ {ap_m:^6.3f} │ {ap_l:^6.3f} │ {ar:^6.3f} │ {delta_str:^8} │ {status}"
        )
        prev_ap = ap

    print("=" * 105)
    print(f" • Historical Peak AP : {peak_ap:.4f} at Epoch {peak_ep}")
    if sorted_eps:
        recent_ap = eval_epochs[sorted_eps[-1]].get("ap", 0.0)
        diff_from_peak = recent_ap - peak_ap
        print(f" • Latest AP (Epoch {sorted_eps[-1]}): {recent_ap:.4f} (Change from Peak: {diff_from_peak:+0.4f})")
    print("=" * 105 + "\n")


def print_class_ap_matrix(eval_epochs):
    all_classes = []
    for ep in eval_epochs.values():
        for c in ep.get("class_ap", {}).keys():
            if c not in all_classes:
                all_classes.append(c)

    if not all_classes:
        return

    print("=" * 115)
    print(" " * 38 + "PER-CLASS AP TRAJECTORY ACROSS EPOCHS")
    print("=" * 115)
    cls_header = " │ ".join([f"{c[:7]:^7}" for c in all_classes])
    print(f"{'Epoch':^6} │ {cls_header}")
    print("─" * 115)

    for ep in sorted(eval_epochs.keys()):
        cap = eval_epochs[ep].get("class_ap", {})
        row = " │ ".join([f"{cap.get(c, float('nan')) * 100:^7.2f}" for c in all_classes])
        print(f"{ep:^6} │ {row}")
    print("=" * 115 + "\n")


def print_training_loss_summary(epoch_losses):
    print("=" * 110)
    print(" " * 34 + "EPOCH-AVERAGED TRAINING LOSS DECAY")
    print("=" * 110)
    header = f"{'Epoch':^6} │ {'Total':^7} │ {'IoU':^6} │ {'NWD':^6} │ {'Conf':^6} │ {'Cls':^6} │ {'Mem':^6} │ {'Proj':^6} │ {'Sem':^6} │ {'LR':^9}"
    print(header)
    print("─" * 110)

    for ep in sorted(epoch_losses.keys()):
        m = epoch_losses[ep]
        print(
            f"{ep:^6} │ {m.get('total_loss', 0):^7.3f} │ {m.get('iou_loss', 0):^6.3f} │ {m.get('nwd_loss', 0):^6.3f} │ "
            f"{m.get('conf_loss', 0):^6.3f} │ {m.get('cls_loss', 0):^6.3f} │ {m.get('mem_loss', 0):^6.3f} │ "
            f"{m.get('proj_loss', 0):^6.3f} │ {m.get('sem_loss', 0):^6.3f} │ {m.get('lr', 0):^9.3e}"
        )
    print("=" * 110 + "\n")


def export_csvs(epoch_losses, eval_epochs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    eval_csv_path = os.path.join(out_dir, "eval_trajectory.csv")
    loss_csv_path = os.path.join(out_dir, "loss_trajectory.csv")

    # Export Evaluation CSV
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        all_classes = sorted({c for ep in eval_epochs.values() for c in ep.get("class_ap", {}).keys()})
        writer.writerow(["epoch", "ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar", *[f"ap_{c}" for c in all_classes]])

        for ep in sorted(eval_epochs.keys()):
            d = eval_epochs[ep]
            cap = d.get("class_ap", {})
            writer.writerow([
                ep,
                d.get("ap", ""),
                d.get("ap50", ""),
                d.get("ap75", ""),
                d.get("ap_s", ""),
                d.get("ap_m", ""),
                d.get("ap_l", ""),
                d.get("ar", ""),
                *[cap.get(c, "") for c in all_classes],
            ])

    # Export Training Loss CSV
    with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        all_loss_keys = sorted({k for ep in epoch_losses.values() for k in ep.keys()})
        writer.writerow(["epoch", *all_loss_keys])

        for ep in sorted(epoch_losses.keys()):
            m = epoch_losses[ep]
            writer.writerow([ep, *[m.get(k, "") for k in all_loss_keys]])

    print(f"CSVs saved:\n  • {eval_csv_path}\n  • {loss_csv_path}\n")


def export_plots(epoch_losses, eval_epochs, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Notice] Matplotlib is not installed. Skipping PNG plot generation.")
        return

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "training_and_eval_trajectory.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # Plot 1: Overall AP and Scale Dissection
    ax1 = axes[0, 0]
    eps = sorted(eval_epochs.keys())
    if eps:
        ap = [eval_epochs[e].get("ap", 0) for e in eps]
        ap_s = [eval_epochs[e].get("ap_s", 0) for e in eps]
        ap_m = [eval_epochs[e].get("ap_m", 0) for e in eps]
        ap_l = [eval_epochs[e].get("ap_l", 0) for e in eps]

        ax1.plot(eps, ap, "k-o", lw=2.5, label="AP (all)")
        ax1.plot(eps, ap_s, "r--s", lw=1.8, label="AP small (micro)")
        ax1.plot(eps, ap_m, "b-.^", lw=1.8, label="AP medium")
        ax1.plot(eps, ap_l, "g-d", lw=1.8, label="AP large (macro)")
        ax1.axhline(0.270, color="grey", linestyle=":", label="0.270 Plateau Wall")
        ax1.set_title("Validation AP Scaling Across Object Sizes", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("AP (IoU=0.50:0.95)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right", fontsize=9)

    # Plot 2: Per-Class AP Evolution
    ax2 = axes[0, 1]
    all_classes = sorted({c for ep in eval_epochs.values() for c in ep.get("class_ap", {}).keys()})
    for c in all_classes:
        y = [eval_epochs[e].get("class_ap", {}).get(c, float("nan")) * 100 for e in eps]
        ax2.plot(eps, y, "-o", ms=4, label=c)
    ax2.set_title("Per-Class AP Progression (0 to 100)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Class AP (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    # Plot 3: Primary Training Losses vs LR
    ax3 = axes[1, 0]
    t_eps = sorted(epoch_losses.keys())
    if t_eps:
        tot = [epoch_losses[e].get("total_loss", 0) for e in t_eps]
        iou = [epoch_losses[e].get("iou_loss", 0) for e in t_eps]
        cls_l = [epoch_losses[e].get("cls_loss", 0) for e in t_eps]
        conf = [epoch_losses[e].get("conf_loss", 0) for e in t_eps]

        ax3.plot(t_eps, tot, "k-", lw=2, label="Total Loss")
        ax3.plot(t_eps, iou, "b--", label="GIoU Loss")
        ax3.plot(t_eps, cls_l, "g-.", label="Cls Loss")
        ax3.plot(t_eps, conf, "r:", label="Conf Loss")
        ax3.set_title("Primary Detection Loss Decay", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss Magnitude")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=9)

    # Plot 4: Auxiliary Invariance Losses
    ax4 = axes[1, 1]
    if t_eps:
        mem = [epoch_losses[e].get("mem_loss", 0) for e in t_eps]
        sem = [epoch_losses[e].get("sem_loss", 0) for e in t_eps]
        nwd = [epoch_losses[e].get("nwd_loss", 0) for e in t_eps]
        phase = [epoch_losses[e].get("phase_loss", 0) for e in t_eps]

        ax4.plot(t_eps, sem, "m-", lw=1.8, label="Semantic Affordance Loss")
        ax4.plot(t_eps, nwd, "c--", lw=1.8, label="Scale NWD Loss")
        ax4.plot(t_eps, mem, "y-.", lw=1.8, label="Memory Anchor Loss")
        ax4.plot(t_eps, phase, "g:", lw=1.8, label="Fourier Phase Loss")
        ax4.set_title("Auxiliary Invariance Loss Constraints", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Loss Magnitude")
        ax4.grid(True, linestyle="--", alpha=0.6)
        ax4.legend(loc="upper right", fontsize=9)

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Analytics Plot saved to:\n  • {plot_path}\n")


def perform_saturation_analysis(eval_epochs, epoch_losses, window=3):
    sorted_eps = sorted(eval_epochs.keys())
    if len(sorted_eps) < window:
        print("[Notice] Not enough evaluated epochs yet to perform formal saturation analysis.\n")
        return

    recent_eps = sorted_eps[-window:]
    recent_aps = [eval_epochs[e].get("ap", 0.0) for e in recent_eps]
    peak_ap = max(eval_epochs[e].get("ap", 0.0) for e in sorted_eps)
    peak_ep = [e for e in sorted_eps if eval_epochs[e].get("ap", 0.0) == peak_ap][0]

    ap_spread = max(recent_aps) - min(recent_aps)
    latest_ep = sorted_eps[-1]
    latest_ap = recent_aps[-1]

    print("=" * 80)
    print(" " * 22 + "MATHEMATICAL SATURATION AUDIT")
    print("=" * 80)
    print(f" • Analyzed Epoch Range       : Epochs {sorted_eps[0]} to {latest_ep}")
    print(f" • Historical Absolute Peak   : {peak_ap:.4f} (Achieved at Epoch {peak_ep})")
    print(f" • Latest Metric (Epoch {latest_ep:02d})  : {latest_ap:.4f}")
    print(f" • Recent {window}-Epoch AP Spread    : {ap_spread:.5f}")

    # Check for plateau
    if ap_spread < 0.0030 and latest_ep >= 15:
        print("\n [!] SATURATION DETECTED:")
        print(f"     Validation AP has entered a flatline regime around {latest_ap:.3f} AP.")
        print(f"     Delta over the last {window} epochs is only {ap_spread:.4f} AP points.")

        # Identify which scale is dragging
        ap_s_change = eval_epochs[latest_ep].get("ap_s", 0) - eval_epochs[peak_ep].get("ap_s", 0)
        ap_m_change = eval_epochs[latest_ep].get("ap_m", 0) - eval_epochs[peak_ep].get("ap_m", 0)
        ap_l_change = eval_epochs[latest_ep].get("ap_l", 0) - eval_epochs[peak_ep].get("ap_l", 0)
        print("\n [!] Scaling Velocity Since Peak:")
        print(f"     • Micro  (AP_small)  Change: {ap_s_change:+0.4f}")
        print(f"     • Medium (AP_medium) Change: {ap_m_change:+0.4f}")
        print(f"     • Macro  (AP_large)  Change: {ap_l_change:+0.4f}")
    else:
        print("\n [OK] Active Scaling: The model is currently still acquiring positive gradient updates.")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    epoch_losses, eval_epochs = parse_log(args.file)

    if not eval_epochs and not epoch_losses:
        print(f"No valid iteration or evaluation data could be extracted from: {args.file}")
        return

    if eval_epochs:
        print_evaluation_summary(eval_epochs, window=args.window)
        print_class_ap_matrix(eval_epochs)
        perform_saturation_analysis(eval_epochs, epoch_losses, window=args.window)

    if epoch_losses:
        print_training_loss_summary(epoch_losses)

    export_csvs(epoch_losses, eval_epochs, args.out_dir)
    export_plots(epoch_losses, eval_epochs, args.out_dir)


if __name__ == "__main__":
    main()