#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.2: Automated Multi-Domain X-Ray Scanner

import os
import sys
import glob
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict

# === WINDOWS UTF-8 ENCODING SHIELD ===
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.exp import get_exp
from yolox.utils import postprocess, load_ckpt
from yolox.data.data_augment import ValTransform

# Import diagnostic taps and visualization from probe_inference
from probe_inference import (
    TDEInferenceXRay,
    run_counterfactual_ablation,
    render_diagnostic_dashboard,
    Style,
    CATEGORY_MAP
)

class AutoDomainScanner:
    """Recursively locates valid test images across ACDC and BDD100K."""
    def __init__(self, acdc_root="D:/ACDC/rgb_anon", clean_root="D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"):
        self.acdc_root = Path(acdc_root)
        self.clean_root = Path(clean_root)

    def find_image_for_domain(self, domain):
        """Recursively finds the first valid image matching the requested domain."""
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        
        if domain.lower() == "clean":
            search_dir = self.clean_root
            if not search_dir.exists():
                print(f"{Style.YELLOW}Warning: Clean root not found at {search_dir}{Style.RESET}")
                return None
            for ext in patterns:
                for file_path in search_dir.rglob(ext):
                    if "test" not in str(file_path).lower() and file_path.is_file():
                        return str(file_path)
        else:
            # Domain is one of: fog, night, rain, snow
            domain_dir = self.acdc_root / domain.lower()
            if not domain_dir.exists():
                # Fallback: Check if user provided top-level ACDC directory containing domain folders
                domain_dir = self.acdc_root
            
            for ext in patterns:
                for file_path in domain_dir.rglob(ext):
                    # Ensure path corresponds to the requested domain
                    if domain.lower() in str(file_path).lower() and file_path.is_file():
                        return str(file_path)

        return None


def run_comparative_probe(exp, model, img_path, domain, save_dir, ckpt_tag, device="cuda"):
    """Runs full telemetry extraction on a single image and checkpoint."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"{Style.RED}Error loading image: {img_path}{Style.RESET}")
        return None

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mean_r, mean_g, mean_b = img_rgb[:, :, 0].mean(), img_rgb[:, :, 1].mean(), img_rgb[:, :, 2].mean()
    haze_intensity = (mean_r + mean_g + mean_b) / (3.0 * 255.0)

    preproc = ValTransform(legacy=False)
    img_tensor, _ = preproc(img_bgr, None, exp.test_size)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device).half()

    xray = TDEInferenceXRay(model)
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        m.set_meta_training_state(exp.max_epoch, exp.max_epoch)

    # Warmup
    with torch.no_grad():
        _ = model(img_tensor)
    xray.clear()

    # Timed Forward Pass
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
    torch.cuda.synchronize()
    fwd_time = (time.time() - t0) * 1000.0

    detections = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]

    ratio = min(exp.test_size[0] / orig_h, exp.test_size[1] / orig_w)
    if detections is not None and len(detections) > 0:
        detections[:, :4] /= ratio
        boxes_np = detections.cpu().numpy()
    else:
        boxes_np = np.zeros((0, 7))

    ablations = run_counterfactual_ablation(model, img_tensor, exp)

    # Telemetry Variables
    h_s = xray.taps.get("p5_entropy_raw", 0.0)
    tau_adapt = xray.taps.get("tau_adaptive", 0.0)
    g_p3 = xray.taps.get("engram_scale_0_mean_gate", 0.0)
    g_p4 = xray.taps.get("engram_scale_1_mean_gate", 0.0)
    g_p5 = xray.taps.get("engram_scale_2_mean_gate", 0.0)
    inner_l = xray.taps.get("ttt_inner_loss", 0.0)
    mask_r = xray.taps.get("ttt_mask_ratio", 0.0) * 100.0
    
    n_full = len(ablations.get('Full', []))
    n_no_ttt = len(ablations.get('No_TTT', []))
    n_no_eng = len(ablations.get('No_Engram', []))

    # Render Visual Dashboard
    img_name = Path(img_path).stem
    dash_save_path = os.path.join(save_dir, f"{ckpt_tag}_{domain}_{img_name}_DASHBOARD.png")
    render_diagnostic_dashboard(img_rgb, boxes_np, xray.taps, ablations, f"{ckpt_tag}_{domain}", dash_save_path)

    return {
        "domain": domain,
        "ckpt": ckpt_tag,
        "image": Path(img_path).name,
        "haze": haze_intensity,
        "h_s": h_s,
        "tau": tau_adapt,
        "mask_ratio": mask_r,
        "ttt_loss": inner_l,
        "gate_p3": g_p3,
        "gate_p4": g_p4,
        "gate_p5": g_p5,
        "n_full": n_full,
        "n_no_ttt": n_no_ttt,
        "n_no_eng": n_no_eng,
        "fwd_ms": fwd_time
    }


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX Auto-Recursive X-Ray Probe")
    parser.add_argument("-f", "--exp_file", type=str, default="exps/default/tde_yolox_s.py")
    parser.add_argument("-c36", "--ckpt_36", type=str, default="YOLOX_outputs/tde_yolox_s/epoch_36_ckpt.pth")
    parser.add_argument("-c49", "--ckpt_49", type=str, default="YOLOX_outputs/tde_yolox_s/epoch_49_ckpt.pth")
    parser.add_argument("--acdc_root", type=str, default="D:/ACDC/rgb_anon")
    parser.add_argument("--clean_root", type=str, default="D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images")
    parser.add_argument("--save_dir", type=str, default="./probe_results/auto_xray")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{Style.BOLD}{Style.WHITE}╔══════════════════════════════════════════════════════════════════════════════════════╗{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}║         TDE-YOLOX v3.2 AUTOMATED RECURSIVE ADVERSE X-RAY ENGINE                      ║{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}╚══════════════════════════════════════════════════════════════════════════════════════╝{Style.RESET}")

    # 1. Initialize Auto-Scanner
    scanner = AutoDomainScanner(acdc_root=args.acdc_root, clean_root=args.clean_root)
    domains = ["clean", "fog", "snow", "rain", "night"]

    print("\nScanning dataset directories for test images:")
    sample_images = {}
    for d in domains:
        img_path = scanner.find_image_for_domain(d)
        if img_path:
            sample_images[d] = img_path
            print(f"  • {d.upper():<7} -> {Style.GREEN}{img_path}{Style.RESET}")
        else:
            print(f"  • {d.upper():<7} -> {Style.RED}No valid image found!{Style.RESET}")

    if not sample_images:
        print(f"\n{Style.RED}Error: Zero images were located. Check your --acdc_root and --clean_root paths.{Style.RESET}")
        return

    # 2. Build Model & Base Configuration
    exp = get_exp(args.exp_file, None)
    exp.test_conf = 0.25
    exp.nmsthre = 0.65
    model = exp.get_model()
    model.to(device)
    model.half()
    model.eval()

    all_telemetry = []

    # 3. Execute Dual-Checkpoint Diagnostic Run
    for ckpt_tag, ckpt_path in [("E36", args.ckpt_36), ("E49", args.ckpt_49)]:
        if not os.path.exists(ckpt_path):
            print(f"\n{Style.RED}Skipping {ckpt_tag}: File not found at {ckpt_path}{Style.RESET}")
            continue

        print(f"\n{Style.BOLD}{Style.CYAN}================================================================================{Style.RESET}")
        print(f"{Style.BOLD}{Style.CYAN}   LOADING CHECKPOINT: [{ckpt_tag}] -> {ckpt_path}{Style.RESET}")
        print(f"{Style.BOLD}{Style.CYAN}================================================================================{Style.RESET}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_state = ckpt.get("ema", ckpt.get("model", ckpt))
        model = load_ckpt(model, ckpt_state)

        for d, path in sample_images.items():
            print(f"\nExecuting X-Ray on [{d.upper()}] with [{ckpt_tag}]...")
            t_res = run_comparative_probe(
                exp=exp,
                model=model,
                img_path=path,
                domain=d,
                save_dir=args.save_dir,
                ckpt_tag=ckpt_tag,
                device=device
            )
            if t_res:
                all_telemetry.append(t_res)

    # 4. Master Comparative Attribution Summary Matrix
    print(f"\n{Style.BOLD}{Style.WHITE}========================================================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}                   TDE-YOLOX v3.2 ADVERSE FORENSIC ATTRIBUTION MATRIX (E30 vs. E78)                                     {Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}========================================================================================================================{Style.RESET}")
    print(f"{'Domain':<7} | {'Ckpt':<5} | {'H(s) nats':<10} | {'Tau':<5} | {'TTT Mask%':<10} | {'Gate P3':<8} | {'Gate P5':<8} | {'Full':<5} | {'No_TTT':<7} | {'No_Eng':<7} | {'Driver Attribution':<20}")
    print("-" * 120)

    for r in all_telemetry:
        delta_ttt = r['n_full'] - r['n_no_ttt']
        delta_eng = r['n_full'] - r['n_no_eng']
        
        if delta_ttt > 0 and delta_eng > 0:
            attr = f"{Style.CYAN}Bimodal Synergy{Style.RESET}"
        elif delta_ttt > 0:
            attr = f"{Style.GREEN}TTT Driven (+{delta_ttt}){Style.RESET}"
        elif delta_eng > 0:
            attr = f"{Style.MAGENTA}Engram Driven (+{delta_eng}){Style.RESET}"
        elif r['n_full'] == 0:
            attr = f"{Style.RED}Detection Blindness{Style.RESET}"
        else:
            attr = f"{Style.DIM}Conv Stable{Style.RESET}"

        p5_g_str = f"{Style.RED}0.0000{Style.RESET}" if r['gate_p5'] < 1e-4 else f"{r['gate_p5']:.4f}"

        print(f"{r['domain'].upper():<7} | {r['ckpt']:<5} | {r['h_s']:>10.4f} | {r['tau']:>5.2f} | {r['mask_ratio']:>9.1f}% | {r['gate_p3']:>8.4f} | {p5_g_str:<8} | {r['n_full']:<5} | {r['n_no_ttt']:<7} | {r['n_no_eng']:<7} | {attr}")

    print(f"{Style.BOLD}{Style.WHITE}========================================================================================================================{Style.RESET}\n")
    print(f"Visual dashboard panels saved to: {Style.BOLD}{os.path.abspath(args.save_dir)}{Style.RESET}\n")


if __name__ == "__main__":
    main()