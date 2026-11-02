#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

TELEMETRY_FILE = os.path.join("YOLOX_outputs", "tde_yolox_s", "telemetry", "simota_telemetry.jsonl")

def main():
    if not os.path.exists(TELEMETRY_FILE):
        print(f"[!] File not found: {TELEMETRY_FILE}")
        return

    records = []
    with open(TELEMETRY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("[!] No records found in simota_telemetry.jsonl")
        return

    df = pd.DataFrame(records)
    print(f"[*] Loaded {len(df)} records from {TELEMETRY_FILE}")
    print(f"[*] Available Columns: {list(df.columns)}")

    # Check available metric fields
    has_num_fg = "num_fg_total" in df.columns
    has_iou = "matched_iou_mean" in df.columns
    has_k = "dynamic_k_mean" in df.columns

    # Dynamic subplot layout based on logged columns
    num_plots = sum([has_num_fg, has_iou, has_k])
    if num_plots == 0:
        print("[!] No plottable metrics found in dataframe.")
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    ax_idx = 0

    # 1. Matched Foregrounds (num_fg_total)
    if has_num_fg:
        ax = axes[ax_idx]
        ax.plot(df["iter"], df["num_fg_total"], color="royalblue", label="Matched Foregrounds (num_fg_total)")
        ax.set_ylabel("Foreground Count / Ratio")
        ax.set_title("SimOTA Target Assignment: Matched Foreground Anchors")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax_idx += 1

    # 2. Matched IoU Mean
    if has_iou:
        ax = axes[ax_idx]
        ax.plot(df["iter"], df["matched_iou_mean"], color="teal", label="Mean Matched IoU")
        if "matched_iou_min" in df.columns and "matched_iou_max" in df.columns:
            ax.fill_between(df["iter"], df["matched_iou_min"], df["matched_iou_max"], color="teal", alpha=0.2, label="IoU Range")
        ax.set_ylabel("Matched Anchor IoU")
        ax.set_title("SimOTA Assignment Quality: Matched IoU with Ground Truth")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax_idx += 1

    # 3. Dynamic-K Mean
    if has_k:
        ax = axes[ax_idx]
        ax.plot(df["iter"], df["dynamic_k_mean"], color="crimson", label="Mean Dynamic-k")
        ax.set_ylabel("Dynamic-k Value")
        ax.set_title("SimOTA Dynamic-k Allocation per Ground Truth Target")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax_idx += 1

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()

    out_png = "simota_assignment_health.png"
    plt.savefig(out_png, dpi=300)
    print(f"[+] Saved diagnostic plot to: {out_png}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print(" SIMOTA TELEMETRY SUMMARY STATS")
    print("=" * 60)
    if has_num_fg:
        print(f"num_fg_total: Mean = {df['num_fg_total'].mean():.2f}, Min = {df['num_fg_total'].min()}, Max = {df['num_fg_total'].max()}")
    if has_iou:
        print(f"matched_iou : Mean = {df['matched_iou_mean'].mean():.4f}, Min = {df['matched_iou_min'].min():.4f}, Max = {df['matched_iou_max'].max():.4f}")
    if has_k:
        print(f"dynamic_k   : Mean = {df['dynamic_k_mean'].mean():.2f}, Min = {df['dynamic_k_min'].min()}, Max = {df['dynamic_k_max'].max()}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()