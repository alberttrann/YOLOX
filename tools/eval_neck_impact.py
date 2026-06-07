#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# eval_neck_impact.py: Evaluates the causal impact of the Tribrid Neck's global branch by severing it and comparing AP metrics.
import argparse
import torch
import torch.nn as nn
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import get_local_rank, setup_logger
from yolox.models.tribrid_neck import C2f_Tribrid

class NeckAblationWrapper(nn.Module):
    """Wraps model to sever the Sparse Global branch in the Tribrid Neck."""
    def __init__(self, model, disable_global=False):
        super().__init__()
        self.model = model
        self.disable_global = disable_global
        self.hooks = []
        
        if self.disable_global:
            def force_zero_global(module, input, output):
                # Returns absolute zeros, completely neutralizing the global context addition
                return torch.zeros_like(output)
            
            for m in self.model.modules():
                if isinstance(m, C2f_Tribrid):
                    self.hooks.append(m.global_mixer.register_forward_hook(force_zero_global))

    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def forward(self, x, targets=None):
        if hasattr(self.model, "set_meta_training_state"):
            self.model.set_meta_training_state(999, 999) 
        return self.model(x, targets)

@logger.catch
def run_neck_ablation(exp, ckpt_path, batch_size=8, fp16=False):
    rank = get_local_rank()
    torch.cuda.set_device(rank)
    
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{rank}")
    model.load_state_dict(ckpt["model"])
    model.cuda(rank).eval()

    logger.info("\n" + "="*50)
    logger.info("PHASE 1: EVALUATING FULL TDE-YOLOX (GLOBAL NECK ACTIVE)")
    logger.info("="*50)
    
    eval_on = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    eval_on.per_class_AP = True
    active_model = NeckAblationWrapper(model, disable_global=False)
    *_, active_summary = eval_on.evaluate(active_model, False, fp16, None, None, exp.test_size)
    active_model.remove_hooks()
    
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: EVALUATING LOCAL YOLOX (GLOBAL NECK SEVERED)")
    logger.info("="*50)
    
    eval_off = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    eval_off.per_class_AP = True
    severed_model = NeckAblationWrapper(model, disable_global=True)
    *_, severed_summary = eval_off.evaluate(severed_model, False, fp16, None, None, exp.test_size)
    severed_model.remove_hooks()

    # --- REPORT GENERATION ---
    logger.info("\n" + "="*50)
    logger.info("TRIBRID NECK CAUSAL IMPACT REPORT")
    logger.info("="*50)
    
    def extract_ap(summary_str, class_name):
        import re
        match = re.search(rf"\|\s*{class_name}\s*\|\s*([\d.]+)\s*\|", summary_str)
        return float(match.group(1)) if match else 0.0

    classes = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
    print(f"{'Class':<15} | {'Local-Only AP':<15} | {'Global+Local AP':<15} | {'Delta (Neck Lift)':<15}")
    print("-" * 65)
    
    for cls in classes:
        ap_off = extract_ap(severed_summary, cls)
        ap_on = extract_ap(active_summary, cls)
        delta = ap_on - ap_off
        delta_str = f"\033[92m+{delta:.2f}\033[0m" if delta > 0.5 else f"\033[91m{delta:.2f}\033[0m" if delta < -0.5 else f"{delta:.2f}"
        print(f"{cls:<15} | {ap_off:<15.2f} | {ap_on:<15.2f} | {delta_str:<15}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Neck Ablation")
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    setup_logger("YOLOX_outputs/ablation", filename="neck_ablation.txt", mode="a")
    run_neck_ablation(get_exp(args.exp_file), args.ckpt, args.batch_size, args.fp16)