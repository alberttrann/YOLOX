#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import re
import itertools
import torch
import torch.nn as nn
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import get_local_rank, setup_logger
from yolox.models.tribrid_neck import C2f_Tribrid

class TDEAblationWrapper(nn.Module):
    """
    Omni-Ablation Wrapper.
    Dynamically severs the requested components (T, D, or E) using PyTorch hooks
    and method overriding, without modifying the underlying weights.
    """
    def __init__(self, model, disable_t=False, disable_d=False, disable_e=False):
        super().__init__()
        self.model = model
        self.disable_t = disable_t
        self.disable_d = disable_d
        self.disable_e = disable_e
        self.hooks = []
        
        self._apply_ablations()

    def _apply_ablations(self):
        # --- ABLATE T: Test-Time Training ---
        self.original_get_prob = self.model._get_ttt_probability
        if self.disable_t:
            # Force TTT probability to 0.0, bypassing the adaptation loop
            self.model._get_ttt_probability = lambda: 0.0
            logger.warning("  [-] ABLATION ACTIVE: TTT-Stem Disabled (T=OFF)")
        else:
            # Force to 1.0 to ensure full adaptation during inference
            self.model._get_ttt_probability = lambda: 1.0
            logger.info("  [+] COMPONENT ACTIVE: TTT-Stem (T=ON)")

        # --- ABLATE D: DeepSeek Tribrid Neck (Global Context) ---
        if self.disable_d:
            def force_zero_global(module, input, output):
                # Returns pure zeros, neutralizing the sparse global branch
                return torch.zeros_like(output)
            
            for m in self.model.modules():
                if isinstance(m, C2f_Tribrid):
                    self.hooks.append(m.global_mixer.register_forward_hook(force_zero_global))
            logger.warning("  [-] ABLATION ACTIVE: Tribrid Global Neck Disabled (D=OFF)")
        else:
            logger.info("  [+] COMPONENT ACTIVE: Tribrid Global Neck (D=ON)")

        # --- ABLATE E: Engram Memory Bank ---
        if self.disable_e:
            def force_zero_gate(module, input, output):
                # Forces uncertainty gate to 0.0, falling back entirely to Convs
                return torch.zeros_like(output)
            
            for gate in self.model.head.uncertainty_gates:
                self.hooks.append(gate.register_forward_hook(force_zero_gate))
            logger.warning("  [-] ABLATION ACTIVE: Engram Memory Bank Disabled (E=OFF)")
        else:
            logger.info("  [+] COMPONENT ACTIVE: Engram Memory Bank (E=ON)")

    def restore(self):
        """Clean up hooks and monkey-patches to leave model in pristine state."""
        self.model._get_ttt_probability = self.original_get_prob
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def forward(self, x, targets=None):
        # Bypass set_meta_training_state locally since we control _get_ttt_probability directly
        return self.model(x, targets)

@logger.catch
def run_omni_ablation(exp, ckpt_path, batch_size=8, fp16=False):
    rank = get_local_rank()
    torch.cuda.set_device(rank)
    
    # 1. Load Base Model
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{rank}")
    model.load_state_dict(ckpt["model"])
    model.cuda(rank).eval()

    # ==========================================
    # PHASE 1: EVALUATE FULL MODEL (BASELINE)
    # ==========================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: EVALUATING FULL TDE-YOLOX (T=ON, D=ON, E=ON)")
    logger.info("="*60)
    
    eval_full = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
    eval_full.per_class_AP = True
    
    full_model_wrapper = TDEAblationWrapper(model, disable_t=False, disable_d=False, disable_e=False)
    *_, full_summary = eval_full.evaluate(full_model_wrapper, False, fp16, None, None, exp.test_size)
    full_model_wrapper.restore()

    # Generate all combinations of (disable_t, disable_d, disable_e)
    # Exclude (False, False, False) as it is the full model already evaluated
    configs = list(itertools.product([False, True], repeat=3))[1:]

    for disable_t, disable_d, disable_e in configs:
        # ==========================================
        # PHASE 2: EVALUATE ABLATED MODEL
        # ==========================================
        logger.info("\n" + "="*60)
        config_str = f"T={'OFF' if disable_t else 'ON'}, D={'OFF' if disable_d else 'ON'}, E={'OFF' if disable_e else 'ON'}"
        logger.info(f"PHASE 2: EVALUATING ABLATED YOLOX ({config_str})")
        logger.info("="*60)
        
        eval_ablated = exp.get_evaluator(batch_size, is_distributed=False, testdev=False, legacy=False)
        eval_ablated.per_class_AP = True
        
        ablated_model_wrapper = TDEAblationWrapper(model, disable_t=disable_t, disable_d=disable_d, disable_e=disable_e)
        *_, ablated_summary = eval_ablated.evaluate(ablated_model_wrapper, False, fp16, None, None, exp.test_size)
        ablated_model_wrapper.restore()
        
        # ==========================================
        # FORENSIC DELTA REPORT
        # ==========================================
        logger.info("\n" + "="*60)
        logger.info(f"OMNI-ABLATION IMPACT REPORT: ABLATING [{config_str}]")
        logger.info("="*60)
        
        def extract_ap(summary_str, class_name):
            match = re.search(rf"\|\s*{class_name}\s*\|\s*([\d.]+)\s*\|", summary_str)
            return float(match.group(1)) if match else 0.0

        classes = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
        
        print(f"{'Class':<15} | {'Ablated AP':<15} | {'Full Model AP':<15} | {'Delta':<20}")
        print("-" * 75)
        
        total_lift = 0
        for cls in classes:
            ap_off = extract_ap(ablated_summary, cls)
            ap_on = extract_ap(full_summary, cls)
            delta = ap_on - ap_off
            total_lift += delta
            
            if delta > 0.5:
                delta_str = f"\033[92m+{delta:.2f}\033[0m" 
            elif delta < -0.5:
                delta_str = f"\033[91m{delta:.2f}\033[0m" 
            else:
                delta_str = f"{delta:.2f}"
                
            print(f"{cls:<15} | {ap_off:<15.2f} | {ap_on:<15.2f} | {delta_str:<20}")
            
        print("-" * 75)
        print(f"Total Cumulative AP Lift provided by disabled components: {total_lift:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TDE-YOLOX Omni-Ablation")
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()
    setup_logger("YOLOX_outputs/ablation", filename="omni_ablation.txt", mode="a")
    
    run_omni_ablation(
        get_exp(args.exp_file), 
        args.ckpt, 
        args.batch_size, 
        args.fp16
    )