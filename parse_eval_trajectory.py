import re
import argparse
import pandas as pd
import numpy as np

def parse_eval_log(log_path):
    print(f"Parsing evaluation metrics from: {log_path}...\n")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    eval_records = []
    curr_epoch = 0
    current_eval = {}
    
    for i, line in enumerate(lines):
        # 1. Track the current epoch from the trainer logs
        epoch_match = re.search(r"epoch:\s*\[(\d+)/\d+\]", line)
        if epoch_match:
            curr_epoch = int(epoch_match.group(1))
            
        # 2. Extract Global COCO Metrics
        if "Average Precision" in line or "Average Recall" in line:
            # Detect the start of a new evaluation block
            if "AP" in line and "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line:
                if current_eval: 
                    eval_records.append(current_eval)
                current_eval = {"epoch": curr_epoch}
                current_eval["AP"] = float(line.split("=")[-1].strip())
            
            elif "AP" in line and "IoU=0.50 " in line and "area=   all" in line:
                current_eval["AP50"] = float(line.split("=")[-1].strip())
            elif "AP" in line and "IoU=0.75" in line:
                current_eval["AP75"] = float(line.split("=")[-1].strip())
            elif "AP" in line and "area= small" in line:
                current_eval["AP_s"] = float(line.split("=")[-1].strip())
            elif "AP" in line and "area=medium" in line:
                current_eval["AP_m"] = float(line.split("=")[-1].strip())
            elif "AP" in line and "area= large" in line:
                current_eval["AP_l"] = float(line.split("=")[-1].strip())
                
            elif "AR" in line and "area=   all" in line and "maxDets=100" in line:
                current_eval["AR"] = float(line.split("=")[-1].strip())
            elif "AR" in line and "area= small" in line and "maxDets=100" in line:
                current_eval["AR_s"] = float(line.split("=")[-1].strip())
            elif "AR" in line and "area= large" in line and "maxDets=100" in line:
                current_eval["AR_l"] = float(line.split("=")[-1].strip())
                
        # 3. Extract Per-Class AP
        if "per class AP:" in line:
            j = i + 2 # Skip the header lines
            while j < len(lines) and "|" in lines[j]:
                matches = re.findall(r"\|\s*([a-zA-Z\s]+?)\s*\|\s*([\d.nan]+)\s*", lines[j])
                for cls_name, val in matches:
                    cls_name = cls_name.strip().replace(" ", "_")
                    val = np.nan if 'nan' in val.lower() else float(val)
                    current_eval[f"AP_{cls_name}"] = val
                j += 1
                
        # 4. Extract Per-Class AR
        if "per class AR:" in line:
            j = i + 2 # Skip the header lines
            while j < len(lines) and "|" in lines[j]:
                matches = re.findall(r"\|\s*([a-zA-Z\s]+?)\s*\|\s*([\d.nan]+)\s*", lines[j])
                for cls_name, val in matches:
                    cls_name = cls_name.strip().replace(" ", "_")
                    val = np.nan if 'nan' in val.lower() else float(val)
                    current_eval[f"AR_{cls_name}"] = val
                j += 1

    # Append the final block
    if current_eval:
        eval_records.append(current_eval)

    df = pd.DataFrame(eval_records)
    if df.empty:
        print("Error: No evaluation records found in the log.")
        return

    # Filter to display nicely
    global_cols = ["epoch", "AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR", "AR_s", "AR_l"]
    
    # Get all class AP columns dynamically
    ap_class_cols = ["epoch"] + [c for c in df.columns if c.startswith("AP_") and c not in ["AP_s", "AP_m", "AP_l"]]

    print("=========================================================================================================")
    print("GLOBAL EVALUATION METRICS TRAJECTORY")
    print("=========================================================================================================")
    print(df[global_cols].to_markdown(index=False))
    
    print("\n=========================================================================================================")
    print("PER-CLASS AP TRAJECTORY (THE IMBALANCE AUDIT)")
    print("=========================================================================================================")
    print(df[ap_class_cols].to_markdown(index=False))
    print("=========================================================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDE-YOLOX Eval Parser")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to train_log.txt")
    args = parser.parse_args()
    
    parse_eval_log(args.file)