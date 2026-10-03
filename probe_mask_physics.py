#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.3: Stage 1 Feature Physics & Masking Diagnostic Engine

import os
import sys
import math
import argparse
from pathlib import Path

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
from yolox.utils import load_ckpt
from yolox.data.data_augment import ValTransform


class Style:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"


def simulate_masking_formulas(feat, ratio=0.75):
    """
    Evaluates 4 competing masking formulations on the exact same feature tensor:
      - Formula A (Legacy): var / (mean^2 + 1e-3)
      - Formula B (Standardized Variance): (var - mu) / (sigma + 1e-5)
      - Formula C (Spatial Gradient Energy): 2D spatial first-order feature differences
      - Formula D (Ground-Plane Aware Variance): Standardized variance with zenith sky prior
    """
    B, C, H, W = feat.shape
    f_fp32 = feat.float()
    
    # -------------------------------------------------------------------------
    # Formula A: Legacy Contrast (var / mean^2)
    # -------------------------------------------------------------------------
    var_a = torch.clamp(torch.var(f_fp32, dim=1, keepdim=True), min=0.0)
    mean_sq_a = torch.clamp(torch.mean(f_fp32, dim=1, keepdim=True).pow(2), min=1e-3)
    contrast_a = var_a / mean_sq_a
    v4_a = F.avg_pool2d(contrast_a, kernel_size=4, stride=4)
    v8_a = F.avg_pool2d(contrast_a, kernel_size=8, stride=8)
    vf_a = v4_a + F.interpolate(v8_a, size=(H // 4, W // 4), mode='nearest')
    scores_flat_a = vf_a.view(B, -1)
    q_a = torch.quantile(scores_flat_a, ratio, dim=-1, keepdim=True).view(B, 1, 1, 1)
    mask_a = (vf_a >= q_a).to(feat.dtype)
    mask_a_up = F.interpolate(mask_a, size=(H, W), mode='nearest')

    # -------------------------------------------------------------------------
    # Formula B: Standardized Variance ((var - mu) / sigma)
    # -------------------------------------------------------------------------
    var_b = torch.clamp(torch.var(f_fp32, dim=1, keepdim=True), min=0.0)
    v4_b = F.avg_pool2d(var_b, kernel_size=4, stride=4)
    v8_b = F.avg_pool2d(var_b, kernel_size=8, stride=8)
    vf_b = v4_b + F.interpolate(v8_b, size=(H // 4, W // 4), mode='nearest')
    v_mean_b = vf_b.mean(dim=[-2, -1], keepdim=True)
    v_std_b = torch.clamp(vf_b.std(dim=[-2, -1], keepdim=True), min=1e-5)
    vf_b_norm = (vf_b - v_mean_b) / v_std_b
    scores_flat_b = vf_b_norm.view(B, -1)
    q_b = torch.quantile(scores_flat_b, ratio, dim=-1, keepdim=True).view(B, 1, 1, 1)
    mask_b = (vf_b_norm >= q_b).to(feat.dtype)
    mask_b_up = F.interpolate(mask_b, size=(H, W), mode='nearest')

    # -------------------------------------------------------------------------
    # Formula C: Spatial Gradient Energy (Edge Contours |dF/dx| + |dF/dy|)
    # -------------------------------------------------------------------------
    dx = torch.abs(f_fp32[:, :, :, 1:] - f_fp32[:, :, :, :-1])
    dy = torch.abs(f_fp32[:, :, 1:, :] - f_fp32[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    grad_energy = (dx + dy).mean(dim=1, keepdim=True)
    v4_c = F.avg_pool2d(grad_energy, kernel_size=4, stride=4)
    v8_c = F.avg_pool2d(grad_energy, kernel_size=8, stride=8)
    vf_c = v4_c + F.interpolate(v8_c, size=(H // 4, W // 4), mode='nearest')
    scores_flat_c = vf_c.view(B, -1)
    q_c = torch.quantile(scores_flat_c, ratio, dim=-1, keepdim=True).view(B, 1, 1, 1)
    mask_c = (vf_c >= q_c).to(feat.dtype)
    mask_c_up = F.interpolate(mask_c, size=(H, W), mode='nearest')

    # -------------------------------------------------------------------------
    # Formula D: Ground-Plane Modulated Variance (Suppresses Zenith Sky)
    # -------------------------------------------------------------------------
    # Sigmoid vertical prior: smooth transition attenuating top 20% zenith sky
    y_coords = torch.linspace(0.0, 1.0, steps=H // 4, device=feat.device).view(1, 1, -1, 1)
    zenith_suppression = torch.sigmoid(12.0 * (y_coords - 0.20))
    vf_d = vf_b_norm * zenith_suppression
    scores_flat_d = vf_d.view(B, -1)
    q_d = torch.quantile(scores_flat_d, ratio, dim=-1, keepdim=True).view(B, 1, 1, 1)
    mask_d = (vf_d >= q_d).to(feat.dtype)
    mask_d_up = F.interpolate(mask_d, size=(H, W), mode='nearest')

    return {
        "Formula_A_Legacy": (mask_a_up[0, 0].cpu().numpy(), vf_a[0, 0].cpu().numpy()),
        "Formula_B_Standardized": (mask_b_up[0, 0].cpu().numpy(), vf_b_norm[0, 0].cpu().numpy()),
        "Formula_C_SpatialGradient": (mask_c_up[0, 0].cpu().numpy(), vf_c[0, 0].cpu().numpy()),
        "Formula_D_GroundPlane": (mask_d_up[0, 0].cpu().numpy(), vf_d[0, 0].cpu().numpy()),
    }


def analyze_feature_elevation_bands(feat):
    """Divides feature tensor into 4 vertical elevation bands and profiles energy."""
    B, C, H, W = feat.shape
    f_fp32 = feat.float()
    
    bands = {
        "Zenith_Sky [0-25%]":    f_fp32[:, :, 0:H//4, :],
        "Horizon_Skyline [25-50%]": f_fp32[:, :, H//4:H//2, :],
        "Road_Vehicles [50-75%]":   f_fp32[:, :, H//2:3*H//4, :],
        "Near_Asphalt [75-100%]":   f_fp32[:, :, 3*H//4:H, :]
    }
    
    stats = {}
    for name, band in bands.items():
        mean_val = band.mean().item()
        var_val = torch.var(band, dim=1).mean().item()
        l2_val = torch.norm(band, p=2, dim=1).mean().item()
        stats[name] = {"mean": mean_val, "var": var_val, "l2": l2_val}
        
    return stats


def calculate_road_to_sky_ratio(mask_np):
    """
    Computes percentage of active mask pixels residing in:
      - Sky Zone: Top 40% of canvas
      - Road Zone: Bottom 60% of canvas
    """
    H, W = mask_np.shape
    cutoff = int(H * 0.40)
    sky_active = mask_np[:cutoff, :].sum()
    road_active = mask_np[cutoff:, :].sum()
    total_active = max(1.0, sky_active + road_active)
    
    sky_pct = (sky_active / total_active) * 100.0
    road_pct = (road_active / total_active) * 100.0
    ratio = (road_active + 1e-4) / (sky_active + 1e-4)
    return sky_pct, road_pct, ratio


def render_comparison_figure(img_rgb, masks_dict, elevation_stats, domain_tag, save_path):
    fig = plt.figure(figsize=(26, 16))
    fig.patch.set_facecolor('#121212')

    # Panel 1: Original Image with Elevation Bands
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#121212')
    ax1.imshow(img_rgb)
    H_img, W_img = img_rgb.shape[:2]
    ax1.axhline(y=H_img * 0.25, color='cyan', linestyle='--', alpha=0.7, label='25% Horizon')
    ax1.axhline(y=H_img * 0.50, color='yellow', linestyle='--', alpha=0.7, label='50% Road Plane')
    ax1.axhline(y=H_img * 0.75, color='orange', linestyle='--', alpha=0.7, label='75% Near Field')
    ax1.set_title(f"Input Scene [{domain_tag.upper()}] with Elevation Bands", color='white', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', facecolor='#222', edgecolor='none', labelcolor='white')
    ax1.axis('off')

    # Panels 2-5: The 4 Masking Formulas
    plot_slots = [
        (2, "Formula_A_Legacy", "Formula A (Legacy: Var / Mean²)", "#e74c3c"),
        (3, "Formula_B_Standardized", "Formula B (Standardized Variance)", "#3498db"),
        (4, "Formula_C_SpatialGradient", "Formula C (Spatial Gradient Energy)", "#f39c12"),
        (5, "Formula_D_GroundPlane", "Formula D (Ground-Plane Aware Variance)", "#2ecc71")
    ]

    ratios_summary = []

    for slot_idx, key, title_label, border_color in plot_slots:
        ax = fig.add_subplot(2, 3, slot_idx)
        ax.set_facecolor('#1e1e1e')
        mask_np, score_map = masks_dict[key]
        
        # Overlay mask on image
        mask_resized = cv2.resize(mask_np, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        ax.imshow(img_rgb, alpha=0.55)
        ax.imshow(mask_resized, cmap='autumn', alpha=0.65)
        
        sky_p, road_p, r_val = calculate_road_to_sky_ratio(mask_np)
        ratios_summary.append((title_label, sky_p, road_p, r_val))
        
        status = "TARGETING ROAD" if r_val > 1.2 else ("TARGETING SKY" if r_val < 0.8 else "NEUTRAL")
        ax.set_title(f"{title_label}\nRoad: {road_p:.1f}% | Sky: {sky_p:.1f}% [{status}]", color='white', fontsize=12, fontweight='bold')
        ax.axis('off')

    # Panel 6: Elevation Band Energy Profiler & Road-to-Sky Ratios
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#1e1e1e')

    band_names = list(elevation_stats.keys())
    vars_vals = [elevation_stats[k]["var"] for k in band_names]
    means_vals = [elevation_stats[k]["mean"] for k in band_names]

    y_pos = np.arange(len(band_names))
    width = 0.35

    ax6.barh(y_pos - width/2, vars_vals, width, label='Channel Variance (σ²)', color='#e74c3c', alpha=0.85)
    ax6.barh(y_pos + width/2, means_vals, width, label='Mean Activation (μ)', color='#3498db', alpha=0.85)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(band_names, color='white', fontsize=10, fontweight='bold')
    ax6.set_xlabel("Activation Magnitude", color='white', fontsize=11)
    ax6.set_title("Stage 1 Dark2 Physical Energy by Elevation Band", color='white', fontsize=13, fontweight='bold')
    ax6.tick_params(colors='white')
    ax6.legend(facecolor='#222', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    plt.close()

    return ratios_summary


def probe_single_image(exp, model, img_path, domain_tag, save_dir, device="cuda"):
    print(f"\n{Style.BOLD}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}   STAGE 1 MASKING PHYSICS PROBE: [{domain_tag.upper()}]{Style.RESET}")
    print(f"{Style.BOLD}================================================================================{Style.RESET}")
    print(f"  • Image: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"{Style.RED}Error: Cannot read image at {img_path}{Style.RESET}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preproc = ValTransform(legacy=False)
    img_tensor, _ = preproc(img_bgr, None, exp.test_size)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device).half()

    m = model.module if hasattr(model, "module") else model
    
    # 1. Forward through Stem Focus to reach Dark2 input
    with torch.no_grad():
        x_stem = m.backbone.backbone.stem(img_tensor)
        # 2. Forward through Dark2 backbone stage to extract clean_target
        clean_target = m.backbone.backbone.dark2.backbone_stage(x_stem)

    print(f"  • Stage 1 Dark2 Feature Shape: {list(clean_target.shape)} (Stride 4)")

    # Profile physical elevation bands
    elevation_stats = analyze_feature_elevation_bands(clean_target)
    print(f"\n  {Style.BOLD}Vertical Elevation Band Energy Profile:{Style.RESET}")
    for b_name, b_s in elevation_stats.items():
        print(f"    - {b_name:<26} | Mean: {b_s['mean']:+8.4f} | Channel Var: {Style.YELLOW}{b_s['var']:8.4e}{Style.RESET} | L2: {b_s['l2']:6.2f}")

    # Simulate and compare 4 masking formulations
    masks_dict = simulate_masking_formulas(clean_target, ratio=0.75)

    img_name = Path(img_path).stem
    save_path = os.path.join(save_dir, f"{domain_tag}_{img_name}_MASK_PHYSICS.png")
    ratios = render_comparison_figure(img_rgb, masks_dict, elevation_stats, domain_tag, save_path)

    print(f"\n  {Style.BOLD}Road-to-Sky Concentration Audit (Bottom 60% vs. Top 40%):{Style.RESET}")
    for title, sky_p, road_p, r_val in ratios:
        status_color = Style.GREEN if r_val > 1.2 else (Style.RED if r_val < 0.8 else Style.YELLOW)
        print(f"    • {title:<42} -> Road: {Style.BOLD}{road_p:>5.1f}%{Style.RESET} | Sky: {sky_p:>5.1f}% | Ratio: {status_color}{r_val:>5.2f}{Style.RESET}")

    print(f"\n{Style.GREEN}✔ Diagnostic visual dashboard saved to: {Style.BOLD}{save_path}{Style.RESET}")


def auto_find_images():
    """Locates sample images across adverse domains and clean BDD."""
    acdc_root = Path("D:/ACDC/rgb_anon")
    clean_root = Path("D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images")
    
    samples = {}
    for d in ["fog", "snow", "rain", "night"]:
        d_dir = acdc_root / d
        if d_dir.exists():
            for p in d_dir.rglob("*.png"):
                samples[d] = str(p)
                break

    if clean_root.exists():
        for p in clean_root.rglob("*.jpg"):
            if "test" not in str(p).lower():
                samples["clean"] = str(p)
                break
                
    return samples


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX Stage 1 Feature Masking Physics Probe")
    parser.add_argument("-f", "--exp_file", type=str, default="exps/default/tde_yolox_s.py")
    parser.add_argument("-c", "--ckpt", type=str, default="YOLOX_outputs/tde_yolox_s/epoch_49_ckpt.pth")
    parser.add_argument("-i", "--image", type=str, default=None, help="Explicit path to single image")
    parser.add_argument("--domain", type=str, default="adverse")
    parser.add_argument("--save_dir", type=str, default="./probe_results/mask_physics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = get_exp(args.exp_file, None)
    exp.test_conf = 0.001
    
    model = exp.get_model()
    model.to(device).half().eval()

    if os.path.exists(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        ckpt_state = ckpt.get("ema", ckpt.get("model", ckpt))
        model = load_ckpt(model, ckpt_state)
    else:
        print(f"{Style.YELLOW}Warning: Checkpoint not found at {args.ckpt}. Probing with base initialized weights.{Style.RESET}")

    if args.image:
        probe_single_image(exp, model, args.image, args.domain, args.save_dir, device)
    else:
        # Auto-discover images across domains
        samples = auto_find_images()
        if not samples:
            print(f"{Style.RED}Error: Zero sample images could be automatically found.{Style.RESET}")
            return
        for domain, path in samples.items():
            probe_single_image(exp, model, path, domain, args.save_dir, device)


if __name__ == "__main__":
    main()