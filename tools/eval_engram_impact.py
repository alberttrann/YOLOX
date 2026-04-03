#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from loguru import logger

from yolox.exp import get_exp
from yolox.evaluators import COCOEvaluator
from yolox.utils import get_local_rank, setup_logger

class EngramAblationWrapper(nn.Module):
    """
    Wraps the TDE-YOLOX model to forcibly sever the Memory Bank connection.
    This allows us to test the exact same convolutional weights with and without memory.
    """
    def __init__(self, model, disable_memory=False):
        super().__init__()
        self.model = model
        self.disable_memory = disable_memory
        self._register_ablation_hooks()

    def _register_ablation_hooks(self):
        self.hooks = []
        if self.disable_memory:
            def force_zero_gate(module, input, output):
                # Force the uncertainty gate to exactly 0.0 (Memory Off)
                return torch.zeros_like(output)
            
            for gate in self.model.head.uncertainty_gates:
                self.hooks.append(gate.register_forward_hook(force_zero_gate))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, x, targets=None):
        if hasattr(self.model, "set_meta_training_state"):
            self.model.set_meta_training_state(999, 999) 
        return self.model(x, targets)

@logger.catch  # This ensures any future crashes print the FULL traceback
def run_ablation_eval(exp, ckpt_path, batch_size=8, fp16=False):
    rank = get_local_rank()
    torch.cuda.set_device(rank)
    
    # 1. Load Base Model
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{rank}")
    model.load_state_dict(ckpt["model"])
    model.cuda(rank)
    model.eval()

    logger.info("\n" + "="*50)
    logger.info("PHASE 1: EVALUATING FULL TDE-YOLOX (MEMORY ACTIVE)")
    logger.info("="*50)
    
    # Create fresh evaluator for Phase 1
    evaluator1 = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    evaluator1.per_class_AP = True
    evaluator1.per_class_AR = True
    
    active_model = EngramAblationWrapper(model, disable_memory=False)
    # FIX: Use *_ to absorb the first two return values safely. Pass fp16.
    *_, active_summary = evaluator1.evaluate(active_model, False, fp16, None, None, exp.test_size)
    active_model.remove_hooks()
    
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: EVALUATING SEVERED YOLOX (MEMORY DISABLED)")
    logger.info("="*50)
    
    # Create fresh evaluator for Phase 2 to prevent state leaking
    evaluator2 = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    evaluator2.per_class_AP = True
    evaluator2.per_class_AR = True
    
    severed_model = EngramAblationWrapper(model, disable_memory=True)
    # FIX: Use *_ to safely unpack
    *_, severed_summary = evaluator2.evaluate(severed_model, False, fp16, None, None, exp.test_size)
    severed_model.remove_hooks()

    # --- FORENSIC DELTA ANALYSIS ---
    logger.info("\n" + "="*50)
    logger.info("ENGRAM CAUSAL IMPACT REPORT")
    logger.info("="*50)
    
    def extract_ap(summary_str, class_name):
        import re
        match = re.search(rf"\|\s*{class_name}\s*\|\s*([\d.]+)\s*\|", summary_str)
        if match: return float(match.group(1))
        return 0.0

    classes = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
    
    print(f"{'Class':<15} | {'No Memory AP':<15} | {'With Memory AP':<15} | {'Delta (Engram Lift)':<15}")
    print("-" * 65)
    
    total_lift = 0
    for cls in classes:
        ap_off = extract_ap(severed_summary, cls)
        ap_on = extract_ap(active_summary, cls)
        delta = ap_on - ap_off
        total_lift += delta
        
        if delta > 0.5:
            delta_str = f"\033[92m+{delta:.2f}\033[0m" # Green
        elif delta < -0.5:
            delta_str = f"\033[91m{delta:.2f}\033[0m" # Red
        else:
            delta_str = f"{delta:.2f}"
            
        print(f"{cls:<15} | {ap_off:<15.2f} | {ap_on:<15.2f} | {delta_str:<15}")
        
    print("-" * 65)
    print(f"Total Cumulative AP Lift across all classes: {total_lift:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TDE-YOLOX Engram Ablation")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="Experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="Checkpoint for eval")
    parser.add_argument("-b", "--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for faster eval")
    args = parser.parse_args()

    setup_logger("YOLOX_outputs/ablation", filename="ablation_log.txt", mode="a")
    exp = get_exp(args.exp_file)
    
    # Pass args.fp16 into the function
    run_ablation_eval(exp, args.ckpt, args.batch_size, args.fp16)