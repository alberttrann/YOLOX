#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.2: Automated 5-Milestone Longitudinal Diagnostic Engine

import os
import sys
import math
import argparse
from collections import OrderedDict
import cv2
import numpy as np

# === WINDOWS UTF-8 ENCODING SHIELD ===
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# =====================================

import torch
import torch.nn.functional as F

from yolox.exp import get_exp
from yolox.utils import postprocess, load_ckpt
from yolox.data.data_augment import ValTransform

class Style:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

CATEGORY_MAP = {
    1: "car", 2: "bus", 3: "truck", 4: "person", 
    5: "rider", 6: "bike", 7: "motor", 8: "traffic light", 9: "traffic sign"
}


def safe_norm(t):
    return torch.norm(t.float(), p=2).item()


def analyze_prototypes(P):
    """Analyzes S^127 Hyperspherical Geometry for a single prototype bank."""
    P = P.float()
    row_norms = torch.norm(P, p=2, dim=-1)
    mean_norm = row_norms.mean().item()
    min_norm = row_norms.min().item()
    max_norm = row_norms.max().item()

    P_norm = F.normalize(P, p=2, dim=-1)
    sim_matrix = torch.matmul(P_norm, P_norm.t())
    eye = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[~eye]

    mean_cos = off_diag.mean().item()
    clamped_cos = torch.clamp(off_diag, -1.0, 1.0)
    mean_angle = torch.acos(clamped_cos).mean().item() * (180.0 / math.pi)

    return {
        "mean_norm": mean_norm,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "mean_cos": mean_cos,
        "mean_angle": mean_angle
    }


def audit_static_checkpoint(ckpt_path):
    print(f"  • Reading state tensors from: {os.path.basename(ckpt_path)}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    meta = {
        "start_epoch": ckpt.get("start_epoch", "N/A"),
        "best_ap": ckpt.get("best_ap", 0.0),
        "curr_ap": ckpt.get("curr_ap", 0.0),
        "curr_iter": ckpt.get("curr_iter", 0)
    }

    # Extract model and EMA state dicts
    model_sd = ckpt.get("model", ckpt)
    ema_sd = ckpt.get("ema", None)

    # 1. GroupNorm Gamma Drift in Dark2
    gn_weights = []
    for k, v in model_sd.items():
        if "dark2" in k and "gn.weight" in k:
            gn_weights.append(v.float().mean().item())
    dark2_mean_gamma = float(np.mean(gn_weights)) if gn_weights else 0.0

    # 2. TTT-Dark2 Step Sizes
    ttt_lrs = []
    for k, v in model_sd.items():
        if "ttt_lrs" in k:
            ttt_lrs.append(F.softplus(v.float()).item())
    mean_ttt_lr = float(np.mean(ttt_lrs)) if ttt_lrs else 0.0

    # 3. Scale-AttnRes Query Vector Norms
    attnres_norms = {}
    for name in ["attnres_n3", "attnres_n4", "attnres_n5"]:
        for k, v in model_sd.items():
            if name in k and k.endswith(".w"):
                attnres_norms[name] = safe_norm(v)

    # 4. Prototype Geometry across Banks
    proto_stats = {}
    for b_idx in [0, 1, 2]:
        key = f"head.memory_banks.{b_idx}.prototypes"
        for k in model_sd.keys():
            if key in k:
                proto_stats[f"bank_{b_idx}"] = analyze_prototypes(model_sd[k])
                break

    # 5. EMA Prototype Radius Deviation
    ema_proto_dev = 0.0
    if ema_sd is not None:
        devs = []
        for b_idx in [0, 1, 2]:
            key = f"head.memory_banks.{b_idx}.prototypes"
            for k in ema_sd.keys():
                if key in k:
                    P_ema = ema_sd[k].float()
                    r_norms = torch.norm(P_ema, p=2, dim=-1)
                    devs.append((r_norms - 1.0).abs().max().item())
        ema_proto_dev = float(np.max(devs)) if devs else 0.0

    # 6. Distance Penalties (Omega)
    omegas = []
    for b_idx in [0, 1, 2]:
        for k, v in model_sd.items():
            if f"head.memory_banks.{b_idx}.omega" in k:
                omegas.append(v.float().item())
                break

    return {
        "meta": meta,
        "dark2_gamma": dark2_mean_gamma,
        "ttt_lr": mean_ttt_lr,
        "attnres_norms": attnres_norms,
        "proto_stats": proto_stats,
        "ema_proto_dev": ema_proto_dev,
        "omegas": omegas
    }


def audit_live_inference(exp, ckpt_path, img_path, device="cuda"):
    exp.test_conf = 0.25
    exp.nmsthre = 0.65
    model = exp.get_model()
    model.to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_state = ckpt.get("ema", None) if (isinstance(ckpt, dict) and ckpt.get("ema") is not None) else ckpt.get("model", ckpt)
    model = load_ckpt(model, ckpt_state)

    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        m.set_meta_training_state(exp.max_epoch, exp.max_epoch)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return {"error": "Image not found"}
    orig_h, orig_w = img_bgr.shape[:2]

    preproc = ValTransform(legacy=False)
    img_tensor, _ = preproc(img_bgr, None, exp.test_size)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

    # Use FP16 if CUDA
    if device.type == "cuda":
        model.half()
        img_tensor = img_tensor.half()

    # Counterfactual Attribution Passes
    with torch.no_grad():
        # 1. Full
        out_full = m(img_tensor)
        dets_full = postprocess(out_full, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        n_full = len(dets_full) if dets_full is not None else 0
        conf_full = dets_full[:, 4].mean().item() if n_full > 0 else 0.0

        # 2. No TTT
        orig_dark2 = m.backbone.backbone.dark2.forward
        m.backbone.backbone.dark2.forward = lambda x_in, run_ttt=True: (m.backbone.backbone.dark2.backbone_stage(x_in), torch.tensor(0.0, device=img_tensor.device))
        out_no_ttt = m(img_tensor)
        dets_no_ttt = postprocess(out_no_ttt, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        n_no_ttt = len(dets_no_ttt) if dets_no_ttt is not None else 0
        m.backbone.backbone.dark2.forward = orig_dark2

        # 3. No Engram
        orig_bank_fwds = [b.forward for b in m.head.memory_banks]
        for b in m.head.memory_banks:
            def make_dummy(target_dim):
                return lambda a, h, obj, z, epoch=0, h_s=0.0: (
                    torch.zeros((h.shape[0], target_dim, h.shape[2], h.shape[3]), device=h.device, dtype=h.dtype),
                    torch.zeros((h.shape[0], 1, h.shape[2], h.shape[3]), device=h.device, dtype=h.dtype),
                    torch.zeros((h.shape[0], h.shape[2] * h.shape[3], 9), device=h.device, dtype=h.dtype)
                )
            b.forward = make_dummy(b.latent_dim)

        out_no_engram = m(img_tensor)
        dets_no_engram = postprocess(out_no_engram, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        n_no_engram = len(dets_no_engram) if dets_no_engram is not None else 0
        for b, orig_f in zip(m.head.memory_banks, orig_bank_fwds):
            b.forward = orig_f

    return {
        "n_full": n_full,
        "conf_full": conf_full,
        "delta_ttt": n_full - n_no_ttt,
        "delta_engram": n_full - n_no_engram
    }


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX 5-Milestone Longitudinal Diagnostic Suite")
    parser.add_argument("-f", "--exp_file", type=str, default="exps/default/tde_yolox_s.py")
    parser.add_argument("-d", "--ckpt_dir", type=str, default="YOLOX_outputs/tde_yolox_s")
    parser.add_argument("-i", "--image", type=str, default="D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/100k/val/b1c66a42-6f7d68ca.jpg")
    parser.add_argument("--save_path", type=str, default="probe_results/longitudinal_audit_summary.txt")
    args = parser.parse_args()

    # Define the 5 critical milestone epochs
    milestone_targets = [
        ("M1 (E10 Warmup)", 10),
        ("M2 (E24 PEAK)", 24),
        ("M3 (E40 Sag Onset)", 40),
        ("M4 (E64 Sag Floor)", 64),
        ("M5 (E78 Terminal)", 78),
    ]

    resolved_milestones = []
    for label, ep in milestone_targets:
        exact_p = os.path.join(args.ckpt_dir, f"epoch_{ep}_ckpt.pth")
        if os.path.exists(exact_p):
            resolved_milestones.append((label, ep, exact_p))
        else:
            # Fallback search
            if ep == 24 and os.path.exists(os.path.join(args.ckpt_dir, "best_ckpt.pth")):
                resolved_milestones.append((label, ep, os.path.join(args.ckpt_dir, "best_ckpt.pth")))
            elif ep == 78 and os.path.exists(os.path.join(args.ckpt_dir, "latest_ckpt.pth")):
                resolved_milestones.append((label, ep, os.path.join(args.ckpt_dir, "latest_ckpt.pth")))

    if not resolved_milestones:
        print(f"{Style.RED}Error: No milestone checkpoints found in {args.ckpt_dir}!{Style.RESET}")
        sys.exit(1)

    print(f"\n{Style.BOLD}{Style.WHITE}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}   TDE-YOLOX LONGITUDINAL SYSTEMIC AUDIT (5-MILESTONE TIMELINE)                 {Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}================================================================================{Style.RESET}")
    print(f"  Checkpoints Located: {len(resolved_milestones)} / 5 milestones")
    for lbl, ep, p in resolved_milestones:
        print(f"    • {lbl:<22} -> {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = get_exp(args.exp_file, None)

    # 1. Execute Static Audits
    static_results = OrderedDict()
    live_results = OrderedDict()

    import cv2
    has_image = os.path.exists(args.image)

    for lbl, ep, p in resolved_milestones:
        print(f"\n{Style.CYAN}==> Auditing Milestone: [{lbl}]...{Style.RESET}")
        static_results[lbl] = audit_static_checkpoint(p)
        if has_image:
            live_results[lbl] = audit_live_inference(exp, p, args.image, device=device)

    # =========================================================================
    # COMPILE COMPREHENSIVE COMPARATIVE TABLE
    # =========================================================================
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    out_lines = []

    def out_print(s):
        print(s)
        out_lines.append(s)

    out_print("\n" + "=" * 115)
    out_print("                           TDE-YOLOX 5-MILESTONE LONGITUDINAL COMPARATIVE MATRIX")
    out_print("=" * 115)
    
    headers = [lbl for lbl, _, _ in resolved_milestones]
    header_str = f"{'Metric / Diagnostic Dimension':<40} | " + " | ".join([f"{h:^12}" for h in headers])
    divider_str = "-" * 40 + "-+-" + "-+-".join(["-" * 12 for _ in headers])
    out_print(header_str)
    out_print(divider_str)

    # 1. Validation Performance History
    cur_aps = [f"{static_results[h]['meta']['curr_ap']:.4f}" if isinstance(static_results[h]['meta']['curr_ap'], float) else str(static_results[h]['meta']['curr_ap']) for h in headers]
    out_print(f"{'Validation mAP (curr_ap)':<40} | " + " | ".join([f"{v:^12}" for v in cur_aps]))

    # 2. Stage 1 GroupNorm Gamma Drift
    gammas = [f"{static_results[h]['dark2_gamma']:.4f}" for h in headers]
    out_print(f"{'Dark2 GN Mean Gamma (Weight)':<40} | " + " | ".join([f"{v:^12}" for v in gammas]))

    # 3. TTT Meta-LR
    ttt_lrs = [f"{static_results[h]['ttt_lr']:.5f}" for h in headers]
    out_print(f"{'TTT Mean Inner LR (effective)':<40} | " + " | ".join([f"{v:^12}" for v in ttt_lrs]))

    # 4. Prototypes S^127 Norm Drift (Bank 0 / P3)
    p3_norms = [f"{static_results[h]['proto_stats'].get('bank_0', {}).get('mean_norm', 0.0):.5f}" for h in headers]
    out_print(f"{'P3 Prototype Mean Norm (Target 1.0)':<40} | " + " | ".join([f"{v:^12}" for v in p3_norms]))

    # 5. Prototypes Angular Separation (Bank 0 / P3)
    p3_angles = [f"{static_results[h]['proto_stats'].get('bank_0', {}).get('mean_angle', 0.0):.2f}°" for h in headers]
    out_print(f"{'P3 Mean Inter-Class Angle (Target 90°)':<40} | " + " | ".join([f"{v:^12}" for v in p3_angles]))

    # 6. Prototypes Angular Separation (Bank 2 / P5 - Buses/Trucks)
    p5_angles = [f"{static_results[h]['proto_stats'].get('bank_2', {}).get('mean_angle', 0.0):.2f}°" for h in headers]
    out_print(f"{'P5 Mean Inter-Class Angle (Target 90°)':<40} | " + " | ".join([f"{v:^12}" for v in p5_angles]))

    # 7. ModelEMA S^127 Deflation Deviation
    ema_devs = [f"{static_results[h]['ema_proto_dev']:.4f}" for h in headers]
    out_print(f"{'ModelEMA S^127 Max Deflation Dev':<40} | " + " | ".join([f"{v:^12}" for v in ema_devs]))

    # 8. ScaleAttnRes Stride 8 Highway Norm
    n3_w = [f"{static_results[h]['attnres_norms'].get('attnres_n3', 0.0):.4f}" for h in headers]
    out_print(f"{'ScaleAttnRes N3 Query Norm ||w||':<40} | " + " | ".join([f"{v:^12}" for v in n3_w]))

    # 9. Engram Distance Penalty Omega (P3)
    om_p3 = [f"{static_results[h]['omegas'][0]:.3f}" if len(static_results[h]['omegas']) > 0 else "N/A" for h in headers]
    out_print(f"{'Engram Omega Distance Penalty (P3)':<40} | " + " | ".join([f"{v:^12}" for v in om_p3]))

    # 10. Live Attribution (If test image was provided)
    if has_image:
        out_print(divider_str)
        dets = [str(live_results[h].get("n_full", "N/A")) for h in headers]
        out_print(f"{'Live Image Object Count (conf=0.25)':<40} | " + " | ".join([f"{v:^12}" for v in dets]))

        confs = [f"{live_results[h].get('conf_full', 0.0):.2f}" for h in headers]
        out_print(f"{'Live Image Mean Confidence':<40} | " + " | ".join([f"{v:^12}" for v in confs]))

        d_ttt = [f"{live_results[h].get('delta_ttt', 0):+d}" for h in headers]
        out_print(f"{'Attribution: Delta TTT (boxes)':<40} | " + " | ".join([f"{v:^12}" for v in d_ttt]))

        d_eng = [f"{live_results[h].get('delta_engram', 0):+d}" for h in headers]
        out_print(f"{'Attribution: Delta Engram (boxes)':<40} | " + " | ".join([f"{v:^12}" for v in d_eng]))

    out_print("=" * 115)

    with open(args.save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\n{Style.GREEN}✔ Longitudinal Diagnostic Matrix saved to: {Style.BOLD}{args.save_path}{Style.RESET}\n")


if __name__ == "__main__":
    main()