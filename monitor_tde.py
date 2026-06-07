#monitor_tde.py: A forensic log parser and report generator for TDE-YOLOX training runs. Extracts training dynamics, evaluation metrics, and timing information from the training log to produce comprehensive visual reports on model performance, adaptation behavior, and resource usage. Critical for debugging and optimizing TDE-YOLOX under distribution shift.
import re
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TDEForensicParser:
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_records = []
        self.eval_summaries = [] 
        self.eval_per_class = [] 
        self.eval_timing = []    
        
        # --- FIXED REGEX ---
        # 1. Changed 'proto_sim' to 'proto_loss'
        # 2. Changed [\d.]+ to [\d.e-]+ to support negative signs and scientific notation
        self.train_re = re.compile(
            r"epoch: \[(\d+)/\d+\]\[(\d+)/\d+\], mem: (\d+)Mb,.*?total_loss: ([\d.e-]+), iou_loss: ([\d.e-]+), l1_loss: ([\d.e-]+), conf_loss: ([\d.e-]+), cls_loss: ([\d.e-]+), mem_loss: ([\d.e-]+), proto_loss: ([\d.e-]+), ttt_prob: ([\d.e-]+), lr: ([\d.e-]+)"
        )
        self.timing_re = re.compile(
            r"Average forward time: ([\d.]+) ms, Average NMS time: ([\d.]+) ms, Average inference time: ([\d.]+) ms"
        )
        self.coco_re = re.compile(r"= ([\d.]+)")

    def parse(self):
        if not os.path.exists(self.log_path): 
            print(f"Error: File not found at {self.log_path}")
            return False
            
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        curr_epoch = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Identify current epoch for eval mapping
            if "---> start train epoch" in line:
                match = re.search(r"epoch(\d+)", line)
                if match: curr_epoch = int(match.group(1)) - 1

            # Match Training Iteration
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
                    "proto_loss": float(train_match.group(10)), 
                    "ttt_prob": float(train_match.group(11)),
                    "lr": float(train_match.group(12))
                })

            # Match Eval Timing
            tm = self.timing_re.search(line)
            if tm:
                self.eval_timing.append({"epoch": curr_epoch, "fwd": float(tm.group(1)), "total": float(tm.group(3))})
                
                # COCO Metrics (12 lines following timing)
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

            # Match Per-Class Tables
            if "per class AP:" in line or "per class AR:" in line:
                m_type = "AP" if "AP" in line else "AR"
                i += 3 # Skip header lines
                while i < len(lines) and "|" in lines[i]:
                    matches = re.findall(r"\|\s*([\w\s]+?)\s*\|\s*([\d.nan]+)\s*", lines[i])
                    for c_name, val in matches:
                        if c_name.strip().lower() != "class":
                            self.eval_per_class.append({
                                "epoch": curr_epoch, 
                                "class": c_name.strip(), 
                                "metric": m_type, 
                                "value": float(val) if val != 'nan' else 0.0
                            })
                    i += 1
                continue
            i += 1
        return True

def generate_report(parser, base_name):
    if not parser.train_records:
        print("CRITICAL ERROR: No training records found. The regex failed to match your log lines.")
        print("Double check the format in monitor_tde.py vs your train_log.txt")
        return
        
    df_t_raw = pd.DataFrame(parser.train_records)
    df_t = df_t_raw.groupby('epoch').mean().reset_index()
    df_e = pd.DataFrame(parser.eval_summaries)
    df_c = pd.DataFrame(parser.eval_per_class)
    df_tm = pd.DataFrame(parser.eval_timing)

    sns.set_style("darkgrid")
    
    # --- PAGE 1: ENGINE REPORT ---
    fig1, axes = plt.subplots(4, 2, figsize=(22, 26))
    fig1.suptitle(f"TDE-YOLOX ENGINE STATS (Epoch {df_t['epoch'].max()})", fontsize=24, fontweight='bold')
    
    # [0,0] Core Losses
    sns.lineplot(data=df_t, x='epoch', y='total_loss', ax=axes[0,0], label='total', linewidth=3)
    for l in ['iou_loss', 'conf_loss', 'cls_loss', 'mem_loss']:
        sns.lineplot(data=df_t, x='epoch', y=l, ax=axes[0,0], label=l)
        
    # [0,1] Learning Rate
    sns.lineplot(data=df_t, x='epoch', y='lr', ax=axes[0,1], color='orange').set_yscale('log')
    
    # [1,0] TTT Prob vs Mem Loss
    sns.lineplot(data=df_t, x='epoch', y='ttt_prob', ax=axes[1,0], color='red', label='ttt_prob')
    ax_t = axes[1,0].twinx()
    sns.lineplot(data=df_t, x='epoch', y='mem_loss', ax=ax_t, color='purple', label='mem_loss', alpha=0.5)
    
    # [1,1] VRAM Usage
    sns.lineplot(data=df_t_raw, x='epoch', y='mem_usage', ax=axes[1,1], color='green')
    
    # [2,0] *** PROTO_LOSS (Identity Separation) ***
    if 'proto_loss' in df_t.columns:
        sns.lineplot(data=df_t, x='epoch', y='proto_loss', ax=axes[2,0], color='dodgerblue', linewidth=3)
        axes[2,0].set_title("Memory Bank Orthogonality (proto_loss ~ 0.0 is ideal)", fontsize=14)
    
    # [2,1] Inference Timing
    if not df_tm.empty:
        sns.lineplot(data=df_tm, x='epoch', y='total', ax=axes[2,1], label='Total Inference')
        sns.lineplot(data=df_tm, x='epoch', y='fwd', ax=axes[2,1], label='TTT + Fwd')
        
    # [3,0] Restoration Efficiency
    df_t['rest_eff'] = (1.0 - df_t['mem_loss']) / (df_t['total_loss'] + 1e-6)
    sns.lineplot(data=df_t, x='epoch', y='rest_eff', ax=axes[3,0], color='teal', linewidth=3)
    axes[3,0].set_title("Restoration Efficiency Index")
    
    axes[3,1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1.savefig(f"{base_name}_ENGINE_REPORT.png")
    print(f"Saved: {base_name}_ENGINE_REPORT.png")

    # --- PAGE 2: ACCURACY REPORT ---
    if not df_e.empty:
        fig2, axes2 = plt.subplots(2, 2, figsize=(22, 16))
        # AP Global
        for m in ['AP_all', 'AP_50', 'AP_75']:
            d = df_e[df_e['metric'] == m]
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[0,0], label=m, marker='o')
        # AP Scales
        for area in ['small', 'medium', 'large']:
            d = df_e[df_e['metric'] == f'AP_{area}']
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[0,1], label=f'AP_{area}')
        # AR Global
        for m in ['AR_10_all', 'AR_100_all']:
            d = df_e[df_e['metric'] == m]
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[1,0], label=m, marker='^')
        # AR Scales
        for area in ['small', 'medium', 'large']:
            d = df_e[df_e['metric'] == f'AR_100_{area}']
            if not d.empty: sns.lineplot(data=d, x='epoch', y='value', ax=axes2[1,1], label=f'AR_{area}')
        
        plt.tight_layout()
        fig2.savefig(f"{base_name}_ACCURACY_REPORT.png")
        print(f"Saved: {base_name}_ACCURACY_REPORT.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()
    forensics = TDEForensicParser(args.file)
    if forensics.parse():
        generate_report(forensics, args.file.replace(".txt", ""))
        print("EXPERT FORENSIC ANALYSIS COMPLETE.")
    else: print("Log Parsing Error.")

if __name__ == "__main__": main()