import re
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np

class TDEForensicParser:
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_records = []
        self.eval_summaries = [] 
        self.eval_per_class = [] 
        self.eval_timing = []
        self.probe_records = [] 
        self.engram_records = []

        # Captures training metrics including the optional sep_loss
        self.train_re = re.compile(
            r"epoch: \[(\d+)/\d+\]\[(\d+)/\d+\], mem: (\d+)Mb,.*?total_loss: ([\d.]+), iou_loss: ([\d.]+), l1_loss: ([\d.]+), conf_loss: ([\d.]+), cls_loss: ([\d.]+), mem_loss: ([\d.]+),(?: sep_loss: ([\d.]+),)? ttt_prob: ([\d.]+), lr: ([\d.e-]+)"
        )
        self.timing_re = re.compile(
            r"Average forward time: ([\d.]+) ms, Average NMS time: ([\d.]+) ms, Average inference time: ([\d.]+) ms"
        )
        self.coco_re = re.compile(r"= ([\d.]+)")
        
        self.probe_re = re.compile(
            r"\[PROBE E(\d+)I(\d+)\] grad=([\d.]+) param_norm=([\d.]+)"
        )

        # Captures Engram metrics logged at epoch boundaries
        self.engram_re = re.compile(
            r"\[ENGRAM E(\d+)\] param_norm=([\d.]+) \| mean_off_diag_sim=([\d.]+) \| max_off_diag_sim=([\d.]+) \| engram_lr=([\d.e-]+)"
        )

    def parse(self):
        if not os.path.exists(self.log_path): return False
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        curr_epoch = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Update current epoch context from start markers
            if "---> start train epoch" in line:
                match = re.search(r"epoch(\d+)", line)
                if match: curr_epoch = int(match.group(1)) - 1

            train_match = self.train_re.search(line)
            if train_match:
                record = {
                    "epoch": int(train_match.group(1)) - 1,
                    "iter": int(train_match.group(2)),
                    "mem_usage": int(train_match.group(3)),
                    "total_loss": float(train_match.group(4)),
                    "iou_loss": float(train_match.group(5)),
                    "l1_loss": float(train_match.group(6)),
                    "conf_loss": float(train_match.group(7)),
                    "cls_loss": float(train_match.group(8)),
                    "mem_loss": float(train_match.group(9)),
                }
                record["sep_loss"] = float(train_match.group(10)) if train_match.group(10) else 0.0
                record["ttt_prob"] = float(train_match.group(11))
                record["lr"] = float(train_match.group(12))
                self.train_records.append(record)

            probe_match = self.probe_re.search(line)
            if probe_match:
                self.probe_records.append({
                    "epoch": int(probe_match.group(1)),
                    "iter": int(probe_match.group(2)),
                    "grad": float(probe_match.group(3)),
                    "param_norm": float(probe_match.group(4))
                })

            engram_match = self.engram_re.search(line)
            if engram_match:
                self.engram_records.append({
                    "epoch": int(engram_match.group(1)),
                    "mean_sim": float(engram_match.group(3)),
                    "max_sim": float(engram_match.group(4)),
                    "engram_lr": float(engram_match.group(5))
                })

            tm = self.timing_re.search(line)
            if tm:
                self.eval_timing.append({"epoch": curr_epoch, "fwd": float(tm.group(1)), "total": float(tm.group(3))})
                metrics_map = ["AP_all", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large", "AR_1_all", "AR_10_all", "AR_100_all", "AR_100_small", "AR_100_medium", "AR_100_large"]
                for k in range(12):
                    i += 1
                    if i >= len(lines): break
                    val = self.coco_re.search(lines[i])
                    if val: self.eval_summaries.append({"epoch": curr_epoch, "metric": metrics_map[k], "value": float(val.group(1))})

            if "per class AP:" in line or "per class AR:" in line:
                m_type = "AP" if "AP" in line else "AR"
                i += 3
                while i < len(lines) and "|" in lines[i]:
                    matches = re.findall(r"\|\s*([\w\s]+?)\s*\|\s*([\d.nan]+)\s*", lines[i])
                    for c_name, val in matches:
                        self.eval_per_class.append({"epoch": curr_epoch, "class": c_name.strip(), "metric": m_type, "value": float(val) if val != 'nan' else 0.0})
                    i += 1
                continue
            i += 1
        return True

def generate_report(parser, base_name):
    if not parser.train_records:
        print("No training records found.")
        return

    df_t_raw = pd.DataFrame(parser.train_records)
    df_t = df_t_raw.groupby('epoch').mean().reset_index()
    df_p = pd.DataFrame(parser.probe_records).groupby('epoch').mean().reset_index() if parser.probe_records else pd.DataFrame()
    df_eng = pd.DataFrame(parser.engram_records)
    df_e = pd.DataFrame(parser.eval_summaries)
    df_c = pd.DataFrame(parser.eval_per_class)
    df_tm = pd.DataFrame(parser.eval_timing)

    sns.set_style("darkgrid")
    
    # --- PAGE 1: ENGINE REPORT ---
    fig1, axes = plt.subplots(6, 2, figsize=(22, 38))
    fig1.suptitle(f"TDE-YOLOX ENGINE STATS (Epoch {df_t['epoch'].max()})", fontsize=24, fontweight='bold')
    
    # Row 0: Losses and LRs
    sns.lineplot(data=df_t, x='epoch', y='total_loss', ax=axes[0,0], label='total', linewidth=3)
    for l in ['iou_loss', 'conf_loss', 'cls_loss', 'mem_loss', 'sep_loss']:
        if l in df_t.columns: sns.lineplot(data=df_t, x='epoch', y=l, ax=axes[0,0], label=l)
    axes[0,0].set_title("Loss Components")
            
    sns.lineplot(data=df_t, x='epoch', y='lr', ax=axes[0,1], color='orange', label='Main LR').set_yscale('log')
    if not df_eng.empty:
        sns.lineplot(data=df_eng, x='epoch', y='engram_lr', ax=axes[0,1], color='cyan', label='Engram LR')
    axes[0,1].set_title('Learning Rates')

    # Row 1: TTT Dynamics and Memory
    sns.lineplot(data=df_t, x='epoch', y='ttt_prob', ax=axes[1,0], color='red', label='ttt_prob')
    ax_t1 = axes[1,0].twinx()
    sns.lineplot(data=df_t, x='epoch', y='mem_loss', ax=ax_t1, color='purple', label='mem_loss', alpha=0.5)
    axes[1,0].set_title('TTT Prob vs Mem Loss')

    sns.lineplot(data=df_t_raw, x='epoch', y='mem_usage', ax=axes[1,1], color='green')
    axes[1,1].set_title('Memory Usage (MB)')

    # Row 2: Sync Ratio and Timing
    df_t['sync_ratio'] = df_t['mem_loss'] / (df_t['cls_loss'] + 1e-6)
    sns.lineplot(data=df_t, x='epoch', y='sync_ratio', ax=axes[2,0], color='magenta')
    axes[2,0].set_title('Sync Ratio (Mem Loss / Cls Loss)')

    if not df_tm.empty:
        sns.lineplot(data=df_tm, x='epoch', y='total', ax=axes[2,1], label='Total Inference')
        sns.lineplot(data=df_tm, x='epoch', y='fwd', ax=axes[2,1], label='TTT + Fwd')
        axes[2,1].set_title('Inference Timing (ms)')
    else: axes[2,1].axis('off')

    # Row 3: Norms
    if not df_p.empty:
        sns.lineplot(data=df_p, x='epoch', y='grad', ax=axes[3,0], color='darkred', linewidth=2)
        axes[3,0].set_title('Avg Gradient Norm (Protos)')
        sns.lineplot(data=df_p, x='epoch', y='param_norm', ax=axes[3,1], color='navy', linewidth=2)
        axes[3,1].set_title('Avg Parameter Norm (Protos)')
    else:
        axes[3,0].axis('off')
        axes[3,1].axis('off')

    # Row 4: Engram Similarity and Efficiency
    if not df_eng.empty:
        sns.lineplot(data=df_eng, x='epoch', y='mean_sim', ax=axes[4,0], color='forestgreen', label='Mean Off-Diag', marker='o')
        sns.lineplot(data=df_eng, x='epoch', y='max_sim', ax=axes[4,0], color='limegreen', label='Max Off-Diag', alpha=0.6)
        axes[4,0].set_title('Engram Orthogonality (Similarity)')
    else: axes[4,0].axis('off')

    df_t['rest_eff'] = (1.0 - df_t['mem_loss']) / (df_t['total_loss'] + 1e-6)
    sns.lineplot(data=df_t, x='epoch', y='rest_eff', ax=axes[4,1], color='teal', linewidth=3)
    axes[4,1].set_title("Restoration Efficiency Index")

    # Row 5: Blank
    axes[5,0].axis('off')
    axes[5,1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1.savefig(f"{base_name}_ENGINE_REPORT.png")
    print(f"Saved: {base_name}_ENGINE_REPORT.png")

    # --- PAGE 2: ACCURACY REPORT ---
    if not df_e.empty:
        fig2, axes2 = plt.subplots(2, 2, figsize=(22, 16))
        fig2.suptitle("TDE-YOLOX DETECTION ACCURACY: TOTAL SCALE & THRESHOLD ANALYSIS", fontsize=24, fontweight='bold')

        for m in ['AP_all', 'AP_50', 'AP_75']:
            d = df_e[df_e['metric'] == m]
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[0,0], label=m, marker='o')
        axes2[0,0].set_title('Global AP')
        
        for area in ['small', 'medium', 'large']:
            d = df_e[df_e['metric'] == f'AP_{area}']
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[0,1], label=f'AP_{area}')
        axes2[0,1].set_title('AP by Object Scale')

        for m in ['AR_10_all', 'AR_100_all']:
            d = df_e[df_e['metric'] == m]
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[1,0], label=m.replace("_all",""), marker='^')
        axes2[1,0].set_title('Global AR')

        for area in ['small', 'medium', 'large']:
            d = df_e[df_e['metric'] == f'AR_100_{area}']
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[1,1], label=f'AR_{area}')
        axes2[1,1].set_title('AR by Object Scale')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig2.savefig(f"{base_name}_ACCURACY_REPORT.png")
        print(f"Saved: {base_name}_ACCURACY_REPORT.png")

    # --- PAGE 3: CLASS HEATMAP ---
    if not df_c.empty:
        fig3, ax = plt.subplots(figsize=(16, 12))
        latest = df_c['epoch'].max()
        lcd = df_c[df_c['epoch'] == latest].pivot_table(index="class", columns="metric", values="value", aggfunc='first')
        sns.heatmap(lcd, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"Per-Class Metrics (Epoch {latest})")
        fig3.savefig(f"{base_name}_CLASS_MATRIX.png")
        print(f"Saved: {base_name}_CLASS_MATRIX.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()
    forensics = TDEForensicParser(args.file)
    if forensics.parse():
        generate_report(forensics, args.file.replace(".txt", ""))
        print("EXPERT FORENSIC ANALYSIS COMPLETE.")
    else: print("Log Error.")

if __name__ == "__main__": main()