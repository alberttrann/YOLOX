#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Forensic Static Checkpoint Archaeology Scanner for TDE-YOLOX v3.5/v3.6

import argparse
import os
import math
import glob
import re
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate


def unwrap_state_dict(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 1. Extract Model State
    model_state = ckpt.get("model", ckpt)
    while isinstance(model_state, dict) and "model" in model_state and isinstance(model_state["model"], dict):
        model_state = model_state["model"]
    model_state = {k[7:] if k.startswith("module.") else k: v for k, v in model_state.items()}

    # 2. Extract ModelEMA State (if present)
    ema_state = ckpt.get("ema", None)
    if ema_state is not None:
        while isinstance(ema_state, dict) and "ema" in ema_state and isinstance(ema_state["ema"], dict):
            ema_state = ema_state["ema"]
        ema_state = {k[7:] if k.startswith("module.") else k: v for k, v in ema_state.items()}

    metadata = {
        "epoch": ckpt.get("start_epoch", ckpt.get("epoch", 0)),
        "best_ap": ckpt.get("best_ap", 0.0),
        "curr_ap": ckpt.get("curr_ap", 0.0),
        "has_ema": ema_state is not None
    }
    return model_state, ema_state, metadata


# =========================================================================
# AUDIT 1: HYPERSPHERICAL MANIFOLD HEALTH ON S^127 (576 MODES)
# =========================================================================
def audit_prototypes(state_dict, scale_names=("P3 (Stride 8)", "P4 (Stride 16)", "P5 (Stride 32)")):
    results = []
    for k in range(3):
        key = f"head.memory_banks.{k}.prototypes"
        if key not in state_dict:
            continue
        
        P = state_dict[key].float()  # [576, 128]
        num_prototypes, dim = P.shape
        num_classes = 9
        num_modes = num_prototypes // num_classes

        # Unit Hypersphere Verification
        norms = torch.norm(P, p=2, dim=-1)
        norm_dev = torch.max(torch.abs(norms - 1.0)).item()

        # Normalize strictly
        P_norm = F.normalize(P, p=2, dim=-1, eps=1e-5)
        G = torch.matmul(P_norm, P_norm.t())  # [576, 576] Gram Matrix

        # Intra-Class Mode Dispersion
        class_ids = torch.arange(num_classes).repeat_interleave(num_modes)
        same_class_mask = class_ids.unsqueeze(0) == class_ids.unsqueeze(1)
        eye_mask = torch.eye(num_prototypes, dtype=torch.bool)
        
        intra_mask = same_class_mask & (~eye_mask)
        intra_sims = G[intra_mask]
        intra_mean = intra_sims.mean().item()
        intra_max = intra_sims.max().item()

        # Inter-Class Orthogonality (Collision Risk)
        diff_class_mask = class_ids.unsqueeze(0) != class_ids.unsqueeze(1)
        inter_sims = G[diff_class_mask]
        inter_mean = inter_sims.mean().item()
        inter_max = inter_sims.max().item()
        
        # Violating Pairs (cos > 0.80 across different categories)
        severe_collisions = (inter_sims > 0.80).sum().item() // 2  # symmetric

        status = "HEALTHY"
        if inter_mean > 0.35 or severe_collisions > 50:
            status = "COLLAPSED (High Cross-Class Collision!)"
        elif inter_mean > 0.20 or severe_collisions > 0:
            status = "WARNING (Clustering)"

        results.append([
            scale_names[k],
            f"||p|| dev: {norm_dev:.1e}",
            f"μ = {intra_mean:.3f} (max: {intra_max:.3f})",
            f"μ = {inter_mean:.3f} (max: {inter_max:.3f})",
            f"{severe_collisions} pairs",
            status
        ])

    headers = ["Bank Scale", "Unit Length Dev", "Intra-Class Mode Sim", "Inter-Class Orthogonality", "Pairs (cos > 0.80)", "Manifold Status"]
    return tabulate(results, headers=headers, tablefmt="fancy_grid")


# =========================================================================
# AUDIT 2: SCALE-ATTNRES DEPTH ROUTING HIGHWAYS
# =========================================================================
def audit_scale_attnres(state_dict):
    results = []
    for k, name in enumerate(["n3", "n4", "n5"]):
        key = f"backbone.attnres_{name}.w"
        if key not in state_dict:
            continue
        w = state_dict[key].float()
        dim = w.shape[0]
        w_norm = torch.norm(w, p=2).item()
        w_mean = w.mean().item()
        w_std = w.std().item()

        # Simulate Depth Softmax with synthetic uniform inputs
        # logits = w_safe * 1.0 / sqrt(dim)
        w_safe = torch.clamp(w, -5.0, 5.0)
        num_sources = 2 if name == "n3" else 3
        K_sim = torch.ones(num_sources, dim)  # [M, C]
        scale = dim ** -0.5
        sim_logits = torch.matmul(K_sim, w_safe) * scale  # [M]
        sim_weights = F.softmax(sim_logits, dim=0).numpy().round(3).tolist()

        polarization = "BALANCED"
        if max(sim_weights) > 0.90:
            polarization = "ONE-HOT COLLAPSED (Pathway Severed!)"
        elif max(sim_weights) > 0.70:
            polarization = "HIGHLY POLARIZED"

        results.append([
            f"ScaleAttnRes {name.upper()}",
            f"{w_norm:.4f}",
            f"μ={w_mean:.4f}, σ={w_std:.4f}",
            str(sim_weights),
            polarization
        ])

    headers = ["Highway", "||w||_2 Norm", "Weight Stats", "Simulated Source Softmax", "Routing Status"]
    return tabulate(results, headers=headers, tablefmt="fancy_grid")


# =========================================================================
# AUDIT 3: HEAD PREDICTION BIASES & PHYSICAL OPERATORS
# =========================================================================
def audit_head_biases_and_scalars(state_dict):
    results = []
    
    # 1. Objectness Prediction Biases
    for k in range(3):
        key = f"head.obj_preds.{k}.bias"
        if key in state_dict:
            b = state_dict[key].float().numpy()[0]
            prob = 1.0 / (1.0 + np.exp(-b))
            results.append([f"Obj Pred P{k+3} Bias", f"{b:.3f}", f"Implied Prior Prob = {prob:.4f}", "Healthy" if prob >= 0.05 else "WARNING (Low Prior)"])

    # 2. Classification Prediction Biases
    for k in range(3):
        key = f"head.cls_preds.{k}.bias"
        if key in state_dict:
            b = state_dict[key].float().numpy()
            results.append([f"Cls Pred P{k+3} Bias", f"μ={b.mean():.3f}, σ={b.std():.3f}", f"min: {b.min():.3f}, max: {b.max():.3f}", "Healthy"])

    # 3. Dynamic BQSA Gating Parameters
    if "backbone.p4_bqsa.pool_alpha" in state_dict:
        raw_alpha = state_dict["backbone.p4_bqsa.pool_alpha"].item()
        alpha = 1.0 / (1.0 + math.exp(-raw_alpha))
        results.append(["BQSA Pool Alpha", f"raw={raw_alpha:.3f}", f"σ(alpha) = {alpha:.3f} (MaxPool weight)", "Hybrid Active"])

    if "backbone.p4_bqsa.k_sigma" in state_dict:
        raw_ks = state_dict["backbone.p4_bqsa.k_sigma"].item()
        ks = math.log(1.0 + math.exp(raw_ks)) if raw_ks < 20 else raw_ks
        results.append(["BQSA K-Sigma", f"raw={raw_ks:.3f}", f"softplus(k_sigma) = {ks:.3f}", "Healthy"])

    # 4. Geodesic Anchors in Engram Bank
    for param in ["w_hf", "omega"]:
        key = f"head.memory_banks.0.{param}"
        if key in state_dict:
            val = state_dict[key].item()
            results.append([f"Engram Bank 0: {param}", f"{val:.4f}", f"Scalar norm = {abs(val):.4f}", "Healthy" if abs(val) < 5.0 else "EXPLODED"])

    headers = ["Parameter", "Raw Value / Stats", "Physical Translation", "Diagnostic"]
    return tabulate(results, headers=headers, tablefmt="fancy_grid")


# =========================================================================
# AUDIT 4: MODEL EMA DIVERGENCE (LIVE vs EMA WEIGHTS)
# =========================================================================
def audit_ema_divergence(model_state, ema_state):
    if ema_state is None:
        return "ModelEMA state not present in checkpoint (evaluated on live model weights)."

    modules = {
        "Stem FocusGN": lambda k: "stem" in k,
        "Dark2 TTT Stage": lambda k: "dark2" in k,
        "Dark3-5 Convs": lambda k: any(f"dark{i}" in k for i in [3, 4, 5]),
        "BQSA Neck": lambda k: any(c in k for c in ["p5_dense_attn", "p4_bqsa", "p3_bqsa", "lateral", "reduce", "down"]),
        "ScaleAttnRes": lambda k: "attnres" in k,
        "Engram Memory Banks": lambda k: "memory_banks" in k and "prototypes" in k,
        "Head Convs & Preds": lambda k: any(c in k for c in ["cls_convs", "reg_convs", "cls_preds", "reg_preds", "obj_preds"]),
    }

    results = []
    for mod_name, filter_fn in modules.items():
        keys = [k for k in model_state.keys() if filter_fn(k) and k in ema_state]
        if not keys:
            continue
        
        diff_sq_sum = 0.0
        norm_sq_sum = 0.0
        for k in keys:
            m_w = model_state[k].float()
            e_w = ema_state[k].float()
            diff_sq_sum += torch.sum((m_w - e_w) ** 2).item()
            norm_sq_sum += torch.sum(m_w ** 2).item()

        divergence = math.sqrt(diff_sq_sum) / max(1e-6, math.sqrt(norm_sq_sum))
        
        status = "TRACKING (Healthy)"
        if divergence > 0.50:
            status = "SEVERELY DIVERGED!"
        elif divergence > 0.30:
            status = "LAGGING"

        results.append([mod_name, len(keys), f"{divergence:.4f}", status])

    headers = ["Subsystem", "Tensor Count", "L2 Rel Divergence (||θ - θ_ema|| / ||θ||)", "Synchronization Status"]
    return tabulate(results, headers=headers, tablefmt="fancy_grid")


def scan_single_checkpoint(ckpt_path):
    print("=" * 100)
    print(f"STATIC ARCHEOLOGY PROBE: {os.path.abspath(ckpt_path)}")
    print("=" * 100)

    model_state, ema_state, meta = unwrap_state_dict(ckpt_path)
    print(f"Metadata: Epoch={meta['epoch']}, Best AP={meta['best_ap']:.4f}, Curr AP={meta['curr_ap']:.4f}, EMA Present={meta['has_ema']}\n")

    eval_target = ema_state if meta["has_ema"] else model_state
    target_name = "ModelEMA Shadow State" if meta["has_ema"] else "Live Model State"
    print(f"--- 1. Probing Hyperspherical Prototypes (Target: {target_name}) ---")
    print(audit_prototypes(eval_target))

    print(f"\n--- 2. Probing ScaleAttnRes Depth Highways (Target: {target_name}) ---")
    print(audit_scale_attnres(eval_target))

    print(f"\n--- 3. Probing Head Biases & Operator Anchors (Target: {target_name}) ---")
    print(audit_head_biases_and_scalars(eval_target))

    print(f"\n--- 4. Probing Live vs. ModelEMA Parameter Divergence ---")
    print(audit_ema_divergence(model_state, ema_state))
    print("=" * 100 + "\n")


def scan_trajectory(ckpt_paths):
    # Sort paths by epoch number if possible
    def extract_epoch(p):
        m = re.search(r"epoch_(\d+)", p)
        return int(m.group(1)) if m else 0

    sorted_paths = sorted(ckpt_paths, key=extract_epoch)
    
    print("=" * 110)
    print(f"MULTI-EPOCH TRAJECTORY AUDIT: {len(sorted_paths)} Checkpoints")
    print("=" * 110)

    trajectory_rows = []
    for p in sorted_paths:
        try:
            m_state, e_state, meta = unwrap_state_dict(p)
            ep = meta["epoch"]
            ap = meta["curr_ap"]
            target = e_state if meta["has_ema"] else m_state

            # P3 Cos Sim Mean
            p3_sim = 0.0
            if "head.memory_banks.0.prototypes" in target:
                P = F.normalize(target["head.memory_banks.0.prototypes"].float(), p=2, dim=-1)
                G = torch.matmul(P, P.t())
                eye = torch.eye(P.shape[0], dtype=torch.bool)
                p3_sim = G[~eye].mean().item()

            # P5 Cos Sim Mean
            p5_sim = 0.0
            if "head.memory_banks.2.prototypes" in target:
                P = F.normalize(target["head.memory_banks.2.prototypes"].float(), p=2, dim=-1)
                G = torch.matmul(P, P.t())
                eye = torch.eye(P.shape[0], dtype=torch.bool)
                p5_sim = G[~eye].mean().item()

            # N3 Norm
            n3_norm = target["backbone.attnres_n3.w"].float().norm().item() if "backbone.attnres_n3.w" in target else 0.0

            # Obj P4 Bias
            obj_p4 = target["head.obj_preds.1.bias"].float().numpy()[0] if "head.obj_preds.1.bias" in target else 0.0

            # EMA Div
            ema_div = 0.0
            if e_state is not None:
                d_sum, n_sum = 0.0, 0.0
                for k in ["backbone.lateral_c5.conv.weight", "head.cls_convs.0.0.conv.weight"]:
                    if k in m_state and k in e_state:
                        d_sum += torch.sum((m_state[k].float() - e_state[k].float()) ** 2).item()
                        n_sum += torch.sum(m_state[k].float() ** 2).item()
                ema_div = math.sqrt(d_sum) / max(1e-6, math.sqrt(n_sum))

            trajectory_rows.append([
                f"Epoch {ep}",
                f"{ap:.3f}",
                f"{p3_sim:.3f}",
                f"{p5_sim:.3f}",
                f"{n3_norm:.3f}",
                f"{obj_p4:.3f} (p={1/(1+np.exp(-obj_p4)):.3f})",
                f"{ema_div:.4f}",
                os.path.basename(p)
            ])
        except Exception as e:
            print(f"Skipping {p}: {e}")

    headers = ["Checkpoint", "Eval AP", "P3 Proto Sim", "P5 Proto Sim", "AttnRes N3 ||w||", "Obj P4 Prior", "Key EMA Div", "File"]
    print(tabulate(trajectory_rows, headers=headers, tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX Static Archaeology Scanner")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="Path to single checkpoint")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing epoch_*_ckpt.pth files")
    parser.add_argument("--pattern", type=str, default="epoch_*_ckpt.pth", help="Checkpoint glob pattern")
    args = parser.parse_args()

    if args.ckpt:
        scan_single_checkpoint(args.ckpt)
    elif args.dir:
        search_pattern = os.path.join(args.dir, args.pattern)
        found_ckpts = glob.glob(search_pattern)
        if not found_ckpts:
            print(f"No checkpoints found matching {search_pattern}")
            return
        scan_trajectory(found_ckpts)
    else:
        # Default fallback to current output directory
        default_dir = "YOLOX_outputs/tde_yolox_s"
        if os.path.exists(default_dir):
            found_ckpts = glob.glob(os.path.join(default_dir, "epoch_*_ckpt.pth"))
            if found_ckpts:
                scan_trajectory(found_ckpts)
            else:
                print(f"No epoch checkpoints in {default_dir}. Provide -c or --dir.")
        else:
            print("Please provide a checkpoint path (-c) or directory (--dir).")


if __name__ == "__main__":
    main()