#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import torch
import torch.nn as nn
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import get_local_rank, setup_logger

class TTTAblationWrapper(nn.Module):
    """Wraps model to forcefully disable or enable the TTT loop."""
    def __init__(self, model, disable_ttt=False):
        super().__init__()
        self.model = model
        self.disable_ttt = disable_ttt
        # Monkey-patch the TTT probability function
        self.original_get_prob = self.model._get_ttt_probability
        
        if self.disable_ttt:
            self.model._get_ttt_probability = lambda: 0.0 # Force OFF
        else:
            self.model._get_ttt_probability = lambda: 1.0 # Force ON

    def restore(self):
        self.model._get_ttt_probability = self.original_get_prob

    def forward(self, x, targets=None):
        return self.model(x, targets)

@logger.catch
def run_ttt_ablation(exp, ckpt_path, batch_size=8, fp16=False):
    rank = get_local_rank()
    torch.cuda.set_device(rank)
    
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{rank}")
    model.load_state_dict(ckpt["model"])
    model.cuda(rank).eval()

    logger.info("\n" + "="*50)
    logger.info("PHASE 1: EVALUATING FULL TDE-YOLOX (TTT ACTIVE)")
    logger.info("="*50)
    
    eval_on = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    eval_on.per_class_AP = True
    
    active_model = TTTAblationWrapper(model, disable_ttt=False)
    *_, active_summary = eval_on.evaluate(active_model, False, fp16, None, None, exp.test_size)
    active_model.restore()
    
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: EVALUATING FROZEN YOLOX (TTT DISABLED)")
    logger.info("="*50)
    
    eval_off = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    eval_off.per_class_AP = True
    
    severed_model = TTTAblationWrapper(model, disable_ttt=True)
    *_, severed_summary = eval_off.evaluate(severed_model, False, fp16, None, None, exp.test_size)
    severed_model.restore()

    # --- REPORT GENERATION ---
    logger.info("\n" + "="*50)
    logger.info("TTT-STEM CAUSAL IMPACT REPORT")
    logger.info("="*50)
    
    def extract_ap(summary_str, class_name):
        import re
        match = re.search(rf"\|\s*{class_name}\s*\|\s*([\d.]+)\s*\|", summary_str)
        return float(match.group(1)) if match else 0.0

    classes = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
    print(f"{'Class':<15} | {'TTT OFF AP':<15} | {'TTT ON AP':<15} | {'Delta (TTT Lift)':<15}")
    print("-" * 65)
    
    for cls in classes:
        ap_off = extract_ap(severed_summary, cls)
        ap_on = extract_ap(active_summary, cls)
        delta = ap_on - ap_off
        delta_str = f"\033[92m+{delta:.2f}\033[0m" if delta > 0.5 else f"\033[91m{delta:.2f}\033[0m" if delta < -0.5 else f"{delta:.2f}"
        print(f"{cls:<15} | {ap_off:<15.2f} | {ap_on:<15.2f} | {delta_str:<15}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TTT Ablation")
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    setup_logger("YOLOX_outputs/ablation", filename="ttt_ablation.txt", mode="a")
    run_ttt_ablation(get_exp(args.exp_file), args.ckpt, args.batch_size, args.fp16)