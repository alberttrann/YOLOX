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
        
        self.train_re = re.compile(
            r"epoch: \[(\d+)/\d+\]\[(\d+)/\d+\], mem: (\d+)Mb,.*?total_loss: ([\d.]+), iou_loss: ([\d.]+), l1_loss: ([\d.]+), conf_loss: ([\d.]+), cls_loss: ([\d.]+), mem_loss: ([\d.]+), ttt_prob: ([\d.]+), lr: ([\d.e-]+)"
        )
        self.timing_re = re.compile(
            r"Average forward time: ([\d.]+) ms, Average NMS time: ([\d.]+) ms, Average inference time: ([\d.]+) ms"
        )
        self.coco_re = re.compile(r"= ([\d.]+)")

    def parse(self):
        if not os.path.exists(self.log_path): return False
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        curr_epoch = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "---> start train epoch" in line:
                match = re.search(r"epoch(\d+)", line)
                if match: curr_epoch = int(match.group(1)) - 1

            train_match = self.train_re.search(line)
            if train_match:
                self.train_records.append({
                    "epoch": int(train_match.group(1)),
                    "mem_usage": int(train_match.group(3)),
                    "total_loss": float(train_match.group(4)),
                    "iou_loss": float(train_match.group(5)),
                    "l1_loss": float(train_match.group(6)),
                    "conf_loss": float(train_match.group(7)),
                    "cls_loss": float(train_match.group(8)),
                    "mem_loss": float(train_match.group(9)),
                    "ttt_prob": float(train_match.group(10)),
                    "lr": float(train_match.group(11))
                })

            tm = self.timing_re.search(line)
            if tm:
                self.eval_timing.append({"epoch": curr_epoch, "fwd": float(tm.group(1)), "total": float(tm.group(3))})
                
                # FIXED: Exact mapping to the 12 sequential lines in COCO eval output
                metrics_map = [
                    "AP_all", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
                    "AR_1_all", "AR_10_all", "AR_100_all", "AR_100_small", "AR_100_medium", "AR_100_large"
                ]
                for k in range(12):
                    i += 1
                    if i >= len(lines): break
                    val = self.coco_re.search(lines[i])
                    if val:
                        self.eval_summaries.append({"epoch": curr_epoch, "metric": metrics_map[k], "value": float(val.group(1))})

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
    df_t_raw = pd.DataFrame(parser.train_records)
    df_t = df_t_raw.groupby('epoch').mean().reset_index()
    df_e = pd.DataFrame(parser.eval_summaries)
    df_c = pd.DataFrame(parser.eval_per_class)
    df_tm = pd.DataFrame(parser.eval_timing)

    sns.set_style("darkgrid")
    
    # --- PAGE 1: ENGINE REPORT ---
    fig1, axes = plt.subplots(4, 2, figsize=(22, 26))
    fig1.suptitle(f"TDE-YOLOX ENGINE STATS (Epoch {df_t['epoch'].max()})", fontsize=24, fontweight='bold')
    sns.lineplot(data=df_t, x='epoch', y='total_loss', ax=axes[0,0], label='total', linewidth=3)
    for l in ['iou_loss', 'conf_loss', 'cls_loss', 'mem_loss']:
        sns.lineplot(data=df_t, x='epoch', y=l, ax=axes[0,0], label=l)
    sns.lineplot(data=df_t, x='epoch', y='lr', ax=axes[0,1], color='orange').set_yscale('log')
    sns.lineplot(data=df_t, x='epoch', y='ttt_prob', ax=axes[1,0], color='red', label='ttt_prob')
    ax_t = axes[1,0].twinx()
    sns.lineplot(data=df_t, x='epoch', y='mem_loss', ax=ax_t, color='purple', label='mem_loss', alpha=0.5)
    sns.lineplot(data=df_t_raw, x='epoch', y='mem_usage', ax=axes[1,1], color='green')
    df_t['sync_ratio'] = df_t['mem_loss'] / (df_t['cls_loss'] + 1e-6)
    sns.lineplot(data=df_t, x='epoch', y='sync_ratio', ax=axes[2,0], color='magenta')
    if not df_tm.empty:
        sns.lineplot(data=df_tm, x='epoch', y='total', ax=axes[2,1], label='Total Inference')
        sns.lineplot(data=df_tm, x='epoch', y='fwd', ax=axes[2,1], label='TTT + Fwd')
    # Restoration Efficiency
    df_t['rest_eff'] = (1.0 - df_t['mem_loss']) / (df_t['total_loss'] + 1e-6)
    sns.lineplot(data=df_t, x='epoch', y='rest_eff', ax=axes[3,0], color='teal', linewidth=3)
    axes[3,0].set_title("Restoration Efficiency Index")
    axes[3,1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1.savefig(f"{base_name}_ENGINE_REPORT.png")

    # --- PAGE 2: ACCURACY REPORT (FIXED) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(22, 16))
    fig2.suptitle("TDE-YOLOX DETECTION ACCURACY: TOTAL SCALE & THRESHOLD ANALYSIS", fontsize=24, fontweight='bold')

    # 2.1 Global Precision
    for m in ['AP_all', 'AP_50', 'AP_75']:
        d = df_e[df_e['metric'] == m]
        if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[0,0], label=m, marker='o')
    
    # 2.2 Precision Scale
    for area in ['small', 'medium', 'large']:
        d = df_e[df_e['metric'] == f'AP_{area}']
        if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[0,1], label=f'AP_{area}')

    # 2.3 Global Recall (Matches Plot Labels in your image)
    for m in ['AR_10_all', 'AR_100_all']:
        d = df_e[df_e['metric'] == m]
        if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[1,0], label=m.replace("_all",""), marker='^')

    # 2.4 Recall Scale
    for area in ['small', 'medium', 'large']:
        d = df_e[df_e['metric'] == f'AR_100_{area}']
        if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[1,1], label=f'AR_{area}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig2.savefig(f"{base_name}_ACCURACY_REPORT.png")

    # PAGE 3: HEATMAP
    fig3, ax = plt.subplots(figsize=(16, 12))
    latest = df_c['epoch'].max()
    lcd = df_c[df_c['epoch'] == latest].pivot(index="class", columns="metric", values="value")
    sns.heatmap(lcd, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    fig3.savefig(f"{base_name}_CLASS_MATRIX.png")

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