#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.3: Professional Multi-Engine Forensic Dashboard & Diagnostics

import os
import re
import math
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Headless execution guard
import matplotlib.pyplot as plt
import seaborn as sns


class TDEForensicParser:
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_records = []
        self.eval_summaries = [] 
        self.eval_per_class = [] 
        self.eval_timing = []    
        
        self.header_re = re.compile(
            r"epoch:\s*\[(\d+)/\d+\]\[(\d+)/\d+\].*?mem:\s*(\d+)Mb.*?ETA:\s*[^,]+,\s*(.*)"
        )
        self.timing_re = re.compile(
            r"Average forward time:\s*([\d.]+)\s*ms,\s*Average NMS time:\s*([\d.]+)\s*ms,\s*Average inference time:\s*([\d.]+)\s*ms"
        )
        self.coco_re = re.compile(r"\]\s*=\s*([0-9.]+)")

    def parse(self):
        if not os.path.exists(self.log_path):
            print(f"Error: Log file not found at {self.log_path}")
            return False

        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        curr_epoch = 0
        i = 0
        total_lines = len(lines)

        while i < total_lines:
            line = lines[i].strip()

            if "---> start train epoch" in line:
                match = re.search(r"epoch(\d+)", line)
                if match:
                    curr_epoch = int(match.group(1))

            header_match = self.header_re.search(line)
            if header_match:
                epoch_num = int(header_match.group(1))
                iter_num = int(header_match.group(2))
                mem_mb = int(header_match.group(3))
                metrics_str = header_match.group(4)

                record = {
                    "epoch": epoch_num,
                    "iter": iter_num,
                    "mem_usage": mem_mb,
                }

                kv_pairs = re.findall(r"(\w+):\s*([0-9eE.+-]+)", metrics_str)
                for k, v in kv_pairs:
                    try:
                        record[k] = float(v)
                    except ValueError:
                        continue

                self.train_records.append(record)

            tm = self.timing_re.search(line)
            if tm:
                fwd_t = float(tm.group(1))
                nms_t = float(tm.group(2))
                tot_t = float(tm.group(3))
                self.eval_timing.append({
                    "epoch": curr_epoch, 
                    "fwd": fwd_t, 
                    "nms": nms_t, 
                    "total": tot_t
                })
                
                metrics_map = [
                    "AP_all", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
                    "AR_1_all", "AR_10_all", "AR_100_all", "AR_100_small", "AR_100_medium", "AR_100_large"
                ]
                for k in range(12):
                    i += 1
                    if i >= total_lines:
                        break
                    val = self.coco_re.search(lines[i])
                    if val:
                        self.eval_summaries.append({
                            "epoch": curr_epoch, 
                            "metric": metrics_map[k], 
                            "value": float(val.group(1))
                        })

            if "per class AP:" in line or "per class AR:" in line:
                m_type = "AP" if "AP" in line else "AR"
                i += 3
                while i < total_lines and "|" in lines[i]:
                    matches = re.findall(r"\|\s*([\w\s]+?)\s*\|\s*([\d.nan]+)\s*", lines[i])
                    for c_name, val in matches:
                        c_clean = c_name.strip()
                        if c_clean and c_clean != "class":
                            self.eval_per_class.append({
                                "epoch": curr_epoch, 
                                "class": c_clean, 
                                "metric": m_type, 
                                "value": float(val) if val != 'nan' else 0.0
                            })
                    i += 1
                continue

            i += 1

        print(f"Parsed {len(self.train_records)} training steps and {len(self.eval_timing)} evaluation checkpoints.")
        return len(self.train_records) > 0


def generate_report(parser, base_name):
    df_t_raw = pd.DataFrame(parser.train_records)
    if df_t_raw.empty:
        print("No training records found to plot.")
        return

    df_t = df_t_raw.groupby('epoch').mean().reset_index()
    df_e = pd.DataFrame(parser.eval_summaries)
    df_c = pd.DataFrame(parser.eval_per_class)
    df_tm = pd.DataFrame(parser.eval_timing)

    # Global Aesthetic Theme (Cyber-Dark Forensic Styling)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#181818", "figure.facecolor": "#101010"})

    # =========================================================================
    # PAGE 1: MULTI-ENGINE METRICS REPORT
    # =========================================================================
    fig1, axes = plt.subplots(4, 2, figsize=(24, 28))
    fig1.patch.set_facecolor('#101010')
    latest_epoch = df_t['epoch'].max()
    fig1.suptitle(f"TDE-YOLOX v3.3 MULTI-ENGINE METRICS AUDIT (Epoch {latest_epoch})", 
                  fontsize=24, fontweight='bold', color='white', y=0.985)

    # -------------------------------------------------------------------------
    # 1.1 Macro Detection Losses
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    sns.lineplot(data=df_t, x='epoch', y='total_loss', ax=ax, label='Total Loss', linewidth=3.0, color='#ffffff')
    loss_pal = [('iou_loss', '#3498db', 'IoU (GIoU)'), ('nwd_loss', '#1abc9c', 'NWD (Gaussian)'),
                ('conf_loss', '#2ecc71', 'Confidence (Obj)'), ('cls_loss', '#e74c3c', 'Class (Focal BCE)'),
                ('l1_loss', '#95a5a6', 'L1 (Box Offset)')]
    for l, c, lab in loss_pal:
        if l in df_t.columns:
            sns.lineplot(data=df_t, x='epoch', y=l, ax=ax, label=lab, color=c, linewidth=2.0)
    ax.set_title("1.1 Macro Detection Losses (Primary Objectives)", fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("Loss Magnitude", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(loc="upper right", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # -------------------------------------------------------------------------
    # 1.2 Auxiliary Representation Anchors (Decoupled Twin-Axis)
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    fine_aux = [
        ('proj_loss', '#f39c12', 'proj (TTT Autoenc)'),
        ('phase_loss', '#d35400', 'phase (Fourier Consist)'),
        ('idx_loss', '#e84393', 'idx (BQSA Focal)'),
        ('rep_loss', '#00b894', 'rep (S^127 Margin)')
    ]
    for l, c, lab in fine_aux:
        if l in df_t.columns:
            sns.lineplot(data=df_t, x='epoch', y=l, ax=ax, label=lab, color=c, linewidth=2.0)
    
    ax.set_title("1.2 Auxiliary Representation Anchors (Decoupled Scales)", fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("Fine Structural Losses (proj, phase, idx, rep)", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    if 'mem_loss' in df_t.columns:
        ax_twin = ax.twinx()
        sns.lineplot(data=df_t, x='epoch', y='mem_loss', ax=ax_twin, label='mem_loss (Prototypes)', 
                     color='#9b59b6', linewidth=2.5, linestyle='--')
        ax_twin.set_ylabel("Macro Prototype Loss (mem_loss)", color='#9b59b6', fontsize=11, fontweight='bold')
        ax_twin.tick_params(colors='#9b59b6')
        ax_twin.grid(False)
        ax_twin.legend(loc="upper right", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # -------------------------------------------------------------------------
    # 1.3 Multi-Engine Optimizer Learning Rates (MuSGD Dynamic Spectrum)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    if 'lr' in df_t.columns:
        base_lr = df_t['lr']
        # Group 0: Muon (Prototypes) -> lr * 4.0
        sns.lineplot(x=df_t['epoch'], y=base_lr * 4.0, ax=ax, color='#9b59b6', linewidth=2.5, label='Muon (Prototypes, 4.0x)')
        # Group 1 & 2: Convs / Norms -> lr * 1.0
        sns.lineplot(x=df_t['epoch'], y=base_lr * 1.0, ax=ax, color='#f39c12', linewidth=2.5, label='SGD (Convs/Norms, 1.0x)')
        # Group 3: Meta-SGD (TTT Step Sizes) -> lr * 0.50
        sns.lineplot(x=df_t['epoch'], y=base_lr * 0.50, ax=ax, color='#00cec9', linewidth=2.5, label='Meta-SGD (ttt_lrs, 0.5x)', linestyle='-.')
        
        ax.set_yscale('log')
        ax.set_title("1.3 Multi-Engine Optimizer Learning Rates (MuSGD Spectrum)", fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel("Learning Rate (Log Scale)", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(loc="lower left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # -------------------------------------------------------------------------
    # 1.4 Stochastic TTT Adaptation Ramp vs. Memory Bank Convergence
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    if 'ttt_prob' in df_t.columns:
        sns.lineplot(data=df_t, x='epoch', y='ttt_prob', ax=ax, color='#e74c3c', linewidth=2.5, label='TTT Adaptation Probability')
        ax.set_ylabel("TTT Probability [0.10 -> 0.30]", color='#e74c3c', fontsize=11, fontweight='bold')
        ax.set_ylim(0.0, 0.40)
        ax.tick_params(colors='white')
        ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

        if 'mem_loss' in df_t.columns:
            ax_twin = ax.twinx()
            sns.lineplot(data=df_t, x='epoch', y='mem_loss', ax=ax_twin, color='#9b59b6', linewidth=2.5, linestyle='--', label='mem_loss (Log Scale)')
            ax_twin.set_yscale('log')
            ax_twin.set_ylabel("Memory Loss (Log Scale)", color='#9b59b6', fontsize=11, fontweight='bold')
            ax_twin.tick_params(colors='#9b59b6')
            ax_twin.grid(False)
            ax_twin.legend(loc="upper right", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    ax.set_title("1.4 Stochastic TTT Ramp vs. Memory Anchor Convergence", fontsize=13, fontweight='bold', color='white')
    ax.set_xlabel("Epoch", color='white', fontsize=11)

    # -------------------------------------------------------------------------
    # 1.5 VRAM Allocation Profile & Hardware Stability
    # -------------------------------------------------------------------------
    ax = axes[2, 0]
    vram_gb = df_t_raw['mem_usage'] / 1024.0
    vram_epoch_gb = df_t['mem_usage'] / 1024.0
    
    sns.lineplot(x=df_t['epoch'], y=vram_epoch_gb, ax=ax, color='#2ecc71', linewidth=2.5, label='Mean VRAM (GB)')
    ax.set_title("1.5 VRAM Allocation Profile & Memory Footprint", fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("VRAM Consumption (GB)", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", color='white', fontsize=11)
    ax.set_ylim(bottom=max(0.0, vram_epoch_gb.min() - 2.0), top=vram_epoch_gb.max() + 2.0)
    ax.tick_params(colors='white')
    ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # -------------------------------------------------------------------------
    # 1.6 Inter-Subsystem Alignment Ratios (Decoupled Log-Linear Twin-Axis)
    # -------------------------------------------------------------------------
    ax = axes[2, 1]
    if 'nwd_loss' in df_t.columns and 'iou_loss' in df_t.columns:
        df_t['geo_ratio'] = df_t['nwd_loss'] / (df_t['iou_loss'] + 1e-6)
        sns.lineplot(data=df_t, x='epoch', y='geo_ratio', ax=ax, color='#1abc9c', label='Geometry Ratio (NWD / IoU)', linewidth=2.5)
        ax.set_ylabel("Geometry Ratio (NWD / IoU)", color='#1abc9c', fontsize=11, fontweight='bold')
        ax.set_ylim(0.0, max(0.8, df_t['geo_ratio'].max() * 1.25))
        ax.tick_params(colors='white')
        ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    if 'mem_loss' in df_t.columns and 'cls_loss' in df_t.columns:
        df_t['sync_ratio'] = df_t['mem_loss'] / (df_t['cls_loss'] + 1e-6)
        ax_twin = ax.twinx()
        sns.lineplot(data=df_t, x='epoch', y='sync_ratio', ax=ax_twin, color='#e84393', label='Sync Ratio (Mem / Cls - Log)', linewidth=2.5, linestyle='--')
        ax_twin.set_yscale('log')
        ax_twin.set_ylabel("Semantic Sync Ratio (Mem / Cls - Log)", color='#e84393', fontsize=11, fontweight='bold')
        ax_twin.tick_params(colors='#e84393')
        ax_twin.grid(False)
        ax_twin.legend(loc="upper right", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    ax.set_title("1.6 Inter-Subsystem Alignment Ratios (Geometry & Memory Hand-off)", fontsize=13, fontweight='bold', color='white')
    ax.set_xlabel("Epoch", color='white', fontsize=11)

    # -------------------------------------------------------------------------
    # 1.7 Latency & Real-Time Throughput Benchmark (FPS Boundaries)
    # -------------------------------------------------------------------------
    ax = axes[3, 0]
    if not df_tm.empty:
        sns.lineplot(data=df_tm, x='epoch', y='total', ax=ax, label='Total Inference Latency (ms)', color='#0984e3', linewidth=2.5)
        sns.lineplot(data=df_tm, x='epoch', y='fwd', ax=ax, label='TTT + Fwd Backbone (ms)', color='#00b894', linestyle='--')
        sns.lineplot(data=df_tm, x='epoch', y='nms', ax=ax, label='NMS Head (ms)', color='#b2bec3', linestyle=':')
        
        # Real-Time Frame Rate Threshold Lines
        ax.axhline(y=16.67, color='#e74c3c', linestyle='--', linewidth=1.5, label='60 FPS Bound (16.67 ms)')
        ax.axhline(y=8.33,  color='#f1c40f', linestyle=':',  linewidth=1.5, label='120 FPS Bound (8.33 ms)')
        ax.axhline(y=6.94,  color='#2ecc71', linestyle='-.', linewidth=1.5, label='144 FPS Bound (6.94 ms)')
        
        ax.set_ylim(0.0, max(20.0, df_tm['total'].max() * 1.2))
    
    ax.set_title("1.7 Real-Time Latency Profile & Throughput Boundaries", fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("Latency per Frame (ms)", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(loc="upper right", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # -------------------------------------------------------------------------
    # 1.8 Representation Restoration Efficiency Index
    # -------------------------------------------------------------------------
    ax = axes[3, 1]
    if 'mem_loss' in df_t.columns and 'total_loss' in df_t.columns:
        df_t['rest_eff'] = (10.0 / (df_t['total_loss'] + 1e-6)) * (1.0 / (df_t['mem_loss'] + 1.0))
        sns.lineplot(data=df_t, x='epoch', y='rest_eff', ax=ax, color='#00cec9', linewidth=3.0, label='Restoration Efficiency Index')
        ax.fill_between(df_t['epoch'], df_t['rest_eff'], color='#00cec9', alpha=0.15)
        
    ax.set_title("1.8 Representation Restoration Efficiency (Monotonic Scaling Index)", fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("Efficiency Index (Higher is Better)", color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch", color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    engine_path = f"{base_name}_ENGINE_REPORT.png"
    fig1.savefig(engine_path, dpi=200, facecolor=fig1.get_facecolor(), edgecolor='none')
    plt.close(fig1)
    print(f"✔ Saved Enhanced Engine Report: {engine_path}")

    # =========================================================================
    # PAGE 2: DETECTION ACCURACY & SCALE ANALYSIS
    # =========================================================================
    if not df_e.empty:
        fig2, axes2 = plt.subplots(2, 2, figsize=(24, 18))
        fig2.patch.set_facecolor('#101010')
        fig2.suptitle(f"TDE-YOLOX v3.3 ACCURACY & MULTI-SCALE RESOLUTION ANALYSIS (Epoch {latest_epoch})", 
                      fontsize=24, fontweight='bold', color='white', y=0.985)

        # 2.1 Strict vs. Relaxed IoU Accuracy
        ax = axes2[0, 0]
        for m, c, lab in [('AP_all', '#3498db', 'mAP (0.50:0.95)'), ('AP_50', '#f39c12', 'AP 50 (Relaxed)'), ('AP_75', '#2ecc71', 'AP 75 (Strict)')]:
            d = df_e[df_e['metric'] == m]
            if not d.empty:
                sns.lineplot(data=d, x='epoch', y='value', ax=ax, label=lab, color=c, marker='o', linewidth=2.5)
        ax.set_title("2.1 Global Precision Thresholds (Strict vs. Relaxed IoU)", fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel("AP Score", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

        # 2.2 Scale-Specific Precision
        ax = axes2[0, 1]
        for area, color, lab in [('small', '#e74c3c', 'Small Objects (<32² px)'), 
                                 ('medium', '#f39c12', 'Medium Objects (32²-96² px)'), 
                                 ('large', '#2ecc71', 'Large Objects (>96² px)')]:
            d = df_e[df_e['metric'] == f'AP_{area}']
            if not d.empty:
                sns.lineplot(data=d, x='epoch', y='value', ax=ax, label=lab, color=color, marker='s', linewidth=2.5)
        ax.set_title("2.2 Scale Precision (Small vs. Medium vs. Large)", fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel("AP Score", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

        # 2.3 Global Recall Capacity
        ax = axes2[1, 0]
        for m, c, lab in [('AR_10_all', '#00cec9', 'AR @ 10 Detections'), ('AR_100_all', '#e84393', 'AR @ 100 Detections (Max)')]:
            d = df_e[df_e['metric'] == m]
            if not d.empty:
                sns.lineplot(data=d, x='epoch', y='value', ax=ax, label=lab, color=c, marker='^', linewidth=2.5)
        ax.set_title("2.3 Global Detection Recall Capacity", fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel("AR Score", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

        # 2.4 Scale-Specific Recall
        ax = axes2[1, 1]
        for area, color, lab in [('small', '#e74c3c', 'Small Recall (<32² px)'), 
                                 ('medium', '#f39c12', 'Medium Recall (32²-96² px)'), 
                                 ('large', '#2ecc71', 'Large Recall (>96² px)')]:
            d = df_e[df_e['metric'] == f'AR_100_{area}']
            if not d.empty:
                sns.lineplot(data=d, x='epoch', y='value', ax=ax, label=lab, color=color, marker='^', linewidth=2.5)
        ax.set_title("2.4 Scale Recall (Small Object Recovery Highlighted)", fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel("AR Score", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.legend(loc="upper left", facecolor='#2c3e50', edgecolor='none', labelcolor='white')

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        acc_path = f"{base_name}_ACCURACY_REPORT.png"
        fig2.savefig(acc_path, dpi=200, facecolor=fig2.get_facecolor(), edgecolor='none')
        plt.close(fig2)
        print(f"✔ Saved Enhanced Accuracy Report: {acc_path}")

    # =========================================================================
    # PAGE 3: PER-CLASS MATRIX HEATMAP
    # =========================================================================
    if not df_c.empty:
        fig3, ax3 = plt.subplots(figsize=(18, 12))
        fig3.patch.set_facecolor('#101010')
        latest_eval_epoch = df_c['epoch'].max()
        lcd = df_c[df_c['epoch'] == latest_eval_epoch].pivot(index="class", columns="metric", values="value")
        
        # Categorical Hierarchy Re-Ordering: (1) Heavy Vehicles -> (2) VRUs -> (3) Signs
        cat_order = ['car', 'bus', 'truck', 'person', 'rider', 'bike', 'motor', 'traffic light', 'traffic sign']
        existing_order = [c for c in cat_order if c in lcd.index]
        if len(existing_order) == len(lcd.index):
            lcd = lcd.reindex(existing_order)

        sns.heatmap(lcd, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax3, 
                    annot_kws={"size": 13, "weight": "bold"},
                    cbar_kws={'label': 'Metric Percentage (%)'})
        
        ax3.set_title(f"3.1 Per-Class Performance Matrix (Epoch {latest_eval_epoch})", fontsize=18, fontweight='bold', color='white')
        ax3.tick_params(colors='white', labelsize=12)
        ax3.set_xlabel("Metric Type", color='white', fontsize=12, fontweight='bold')
        ax3.set_ylabel("Object Category", color='white', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        class_path = f"{base_name}_CLASS_MATRIX.png"
        fig3.savefig(class_path, dpi=200, facecolor=fig3.get_facecolor(), edgecolor='none')
        plt.close(fig3)
        print(f"✔ Saved Enhanced Class Matrix: {class_path}")


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX Multi-Engine Forensic Dashboard")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to train_log.txt")
    args = parser.parse_args()

    forensics = TDEForensicParser(args.file)
    if forensics.parse():
        base_name = args.file.replace(".txt", "")
        generate_report(forensics, base_name)
        print("TDE-YOLOX FORENSIC ANALYSIS COMPLETE.")
    else:
        print("Parsing failed. Check the file path.")


if __name__ == "__main__":
    main()