import re
import argparse
import pandas as pd
import numpy as np

def parse_training_log(log_path):
    print(f"Parsing log file: {log_path}...")
    train_records = []

    # Regex tailored to the specific 12-metric TDE-YOLOX v3.3 output
    log_pattern = re.compile(
        r"epoch: \[(\d+)/\d+\]\[\d+/\d+\], mem: (\d+)Mb,.*?total_loss: ([\d.nan]+), "
        r"iou_loss: ([\d.nan]+), nwd_loss: ([\d.nan]+), l1_loss: ([\d.nan]+), "
        r"conf_loss: ([\d.nan]+), cls_loss: ([\d.nan]+), mem_loss: ([\d.nan]+), "
        r"proj_loss: ([\d.nan]+), phase_loss: ([\d.nan]+), idx_loss: ([\d.nan]+), "
        r"rep_loss: ([\d.nan]+), ttt_prob: ([\d.nan]+), lr: ([\d.e-]+)",
        re.IGNORECASE
    )

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # Convert 'nan' strings to np.nan for safe averaging
                def safe_float(val):
                    return np.nan if val.lower() == 'nan' else float(val)

                train_records.append({
                    "epoch": int(match.group(1)),
                    "total_loss": safe_float(match.group(3)),
                    "iou_loss": safe_float(match.group(4)),
                    "nwd_loss": safe_float(match.group(5)),
                    "l1_loss": safe_float(match.group(6)),
                    "conf_loss": safe_float(match.group(7)),
                    "cls_loss": safe_float(match.group(8)),
                    "mem_loss": safe_float(match.group(9)),
                    "proj_loss": safe_float(match.group(10)),
                    "phase_loss": safe_float(match.group(11)),
                    "idx_loss": safe_float(match.group(12)),
                    "rep_loss": safe_float(match.group(13)),
                    "ttt_prob": safe_float(match.group(14)),
                    "lr": safe_float(match.group(15))
                })

    if not train_records:
        print("Error: No matching training metrics found in the log.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(train_records)
    
    # Calculate Epoch-level averages
    epoch_stats = df.groupby('epoch').mean().round(4)
    
    # Drop learning rate from the mean table (it's better viewed raw, but we'll keep it for tracking scale)
    
    print("\n" + "="*120)
    print("TDE-YOLOX MULTI-ENGINE TRAINING TRAJECTORY (EPOCH AVERAGES)")
    print("="*120)
    
    # Output as a clean Markdown table
    markdown_table = epoch_stats.to_markdown()
    print(markdown_table)
    print("="*120)

    # Save to CSV for easy copy-pasting if needed
    out_csv = log_path.replace(".txt", "_trajectory.csv")
    epoch_stats.to_csv(out_csv)
    print(f"Data saved to {out_csv}")
    print("\n[NEXT STEP] Copy the markdown table above and paste it back to the AI for Forensic Analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDE-YOLOX Log Parser")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to train_log.txt")
    args = parser.parse_args()
    
    parse_training_log(args.file)