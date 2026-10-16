#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Forensic Checkpoint & Activation Probe for TDE-YOLOX v3.5/v3.6

import argparse
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from yolox.exp import get_exp


def unwrap_ckpt(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema" in ckpt and ckpt["ema"] is not None:
        state = ckpt["ema"]
        src = "ModelEMA Shadow Weights"
    elif "model" in ckpt and ckpt["model"] is not None:
        state = ckpt["model"]
        src = "Standard Model Weights"
    else:
        state = ckpt
        src = "Raw State Dict"
    
    # Strip DDP prefix
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    return state, src, ckpt.get("best_ap", 0.0), ckpt.get("start_epoch", 0)


# =========================================================================
# MODULE A: STATIC WEIGHT & GEOMETRY AUDIT
# =========================================================================
def audit_static_weights(state1, state2, name1="Epoch 6", name2="Epoch 15"):
    print("=" * 100)
    print(f"STATIC TENSOR ARCHAEOLOGY: {name1} vs {name2}")
    print("=" * 100)

    rows = []

    # 1. Objectness Prediction Biases (Did background penalty crush detection?)
    for k in range(3):
        key = f"head.obj_preds.{k}.bias"
        if key in state1 and key in state2:
            b1 = state1[key].float().numpy()
            b2 = state2[key].float().numpy()
            p1 = 1.0 / (1.0 + np.exp(-b1[0]))
            p2 = 1.0 / (1.0 + np.exp(-b2[0]))
            rows.append([f"Obj Pred P{k+3} Bias (Logit / Prob)", 
                         f"{b1[0]:.3f} (p={p1:.4f})", 
                         f"{b2[0]:.3f} (p={p2:.4f})", 
                         f"Δ logit = {b2[0] - b1[0]:.3f}"])

    # 2. Classification Prediction Biases
    for k in range(3):
        key = f"head.cls_preds.{k}.bias"
        if key in state1 and key in state2:
            b1 = state1[key].float().numpy()
            b2 = state2[key].float().numpy()
            rows.append([f"Cls Pred P{k+3} Mean Bias", f"{b1.mean():.3f} [min: {b1.min():.2f}]", 
                         f"{b2.mean():.3f} [min: {b2.min():.2f}]", f"Δ = {b2.mean() - b1.mean():.3f}"])

    # 3. Stage 1 GroupNorm Affine Scales
    gn_weights1 = [v for k, v in state1.items() if ("dark2" in k and "gn." in k and "weight" in k)]
    gn_weights2 = [v for k, v in state2.items() if ("dark2" in k and "gn." in k and "weight" in k)]
    if gn_weights1 and gn_weights2:
        gw1 = torch.cat([w.flatten() for w in gn_weights1]).float()
        gw2 = torch.cat([w.flatten() for w in gn_weights2]).float()
        rows.append(["Dark2 GroupNorm Gamma (Scale)", 
                     f"μ={gw1.mean():.3f}, σ={gw1.std():.3f}", 
                     f"μ={gw2.mean():.3f}, σ={gw2.std():.3f}", 
                     f"Range: [{gw2.min():.2f}, {gw2.max():.2f}]"])

    # 4. Scale-AttnRes Highway Depth Weights
    for k, s_name in enumerate(["n3", "n4", "n5"]):
        key = f"backbone.attnres_{s_name}.w"
        if key in state1 and key in state2:
            w1 = state1[key].float()
            w2 = state2[key].float()
            rows.append([f"ScaleAttnRes {s_name.upper()} Weight Norm", 
                         f"||w||={w1.norm().item():.3f}, μ={w1.mean().item():.3f}", 
                         f"||w||={w2.norm().item():.3f}, μ={w2.mean().item():.3f}", 
                         f"Δ ||w|| = {w2.norm().item() - w1.norm().item():.3f}"])

    # 5. Geodesic Parameters in Engram Bank
    for param_name in ["w_hf", "v_q", "v_k", "omega"]:
        key = f"head.memory_banks.0.{param_name}"
        if key in state1 and key in state2:
            val1 = state1[key].float()
            val2 = state2[key].float()
            rows.append([f"Engram Bank 0: {param_name}", 
                         f"norm={val1.norm().item():.3f}", 
                         f"norm={val2.norm().item():.3f}", 
                         f"Δ = {val2.norm().item() - val1.norm().item():.3f}"])

    print(tabulate(rows, headers=["Architectural Component", name1, name2, "Diagnostics"], tablefmt="fancy_grid"))

    # 6. Prototypes Hyperspherical Orthogonality Audit (S^127 Collapse Check)
    print("\n" + "=" * 100)
    print("HYPERSPHERICAL PROTOTYPE MANIFOLD AUDIT (S^127)")
    print("=" * 100)
    
    proto_rows = []
    for scale_idx in range(3):
        key = f"head.memory_banks.{scale_idx}.prototypes"
        if key in state1 and key in state2:
            P1 = state1[key].float()  # [576, 128]
            P2 = state2[key].float()
            
            P1_norm = F.normalize(P1, p=2, dim=-1)
            P2_norm = F.normalize(P2, p=2, dim=-1)

            G1 = torch.matmul(P1_norm, P1_norm.t())  # [576, 576]
            G2 = torch.matmul(P2_norm, P2_norm.t())

            eye = torch.eye(G1.shape[0])
            off_diag_1 = G1[eye == 0]
            off_diag_2 = G2[eye == 0]

            proto_rows.append([
                f"Scale P{scale_idx+3} Prototypes (576 modes)",
                f"Cos Sim: μ={off_diag_1.mean():.3f}, max={off_diag_1.max():.3f}",
                f"Cos Sim: μ={off_diag_2.mean():.3f}, max={off_diag_2.max():.3f}",
                "COLLAPSED (Degenerate)!" if off_diag_2.mean() > 0.60 else "Healthy (Separated)"
            ])
            
    print(tabulate(proto_rows, headers=["Scale Bank", name1, name2, "Geometric Status"], tablefmt="fancy_grid"))


# =========================================================================
# MODULE B: DYNAMIC RUNTIME ACTIVATION PROBE
# =========================================================================
def probe_runtime_activations(exp, state_dict, name="Checkpoint", epoch=0, use_fp16=True):
    print("\n" + "=" * 100)
    print(f"DYNAMIC IN-FLIGHT RUNTIME PROBE: {name} (Evaluated at Epoch {epoch})")
    print("=" * 100)

    model = exp.get_model()
    
    # Synchronize epoch state to evaluate the exact dynamic gates of that epoch
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        m.set_meta_training_state(epoch, exp.max_epoch)
        
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    
    # DTYPE SYNC: Convert model and input to the exact same precision
    if use_fp16:
        model.half()
    else:
        model.float()
    model.eval()

    # Load 1 real validation batch
    val_loader = exp.get_eval_loader(batch_size=16, is_distributed=False)
    inps, _, info_imgs, ids = next(iter(val_loader))
    inps = inps.cuda()
    inps = inps.half() if use_fp16 else inps.float()

    telemetry = {
        "gates": []
    }

    hooks = []
    def hook_engram_bank(module, inputs, outputs):
        gate = outputs[1].detach().float().cpu()
        telemetry["gates"].append(gate)
    
    for bank in model.head.memory_banks:
        hooks.append(bank.register_forward_hook(hook_engram_bank))

    with torch.no_grad():
        outputs = model(inps)

    for h in hooks:
        h.remove()

    obj_confs = outputs[..., 4].float().cpu().flatten()
    cls_confs = outputs[..., 5:].max(dim=-1).values.float().cpu().flatten()
    combined = (obj_confs * cls_confs).flatten()

    total_anchors = len(combined)
    surviving_01 = (combined >= 0.01).sum().item()
    surviving_001 = (combined >= 0.001).sum().item()

    all_gates = torch.cat([g.flatten() for g in telemetry["gates"]]) if telemetry["gates"] else torch.tensor([0.0])

    runtime_rows = [
        ["Total Evaluated Anchor Candidates", f"{total_anchors:,}"],
        ["Mean Raw Objectness Prob (Sigmoid)", f"{obj_confs.mean():.4f} (Max: {obj_confs.max():.4f})"],
        ["Mean Top-1 Category Prob", f"{cls_confs.mean():.4f} (Max: {cls_confs.max():.4f})"],
        ["Mean Combined Score (obj * cls)", f"{combined.mean():.5f}"],
        ["Anchors Passing standard test_conf >= 0.01", f"{surviving_01:,} ({surviving_01/total_anchors*100:.2f}%)"],
        ["Anchors Passing adverse test_conf >= 0.001", f"{surviving_001:,} ({surviving_001/total_anchors*100:.2f}%)"],
        ["Engram Gate Activity (g_final)", f"μ={all_gates.mean():.4f}, max={all_gates.max():.4f}, min={all_gates.min():.4f}"]
    ]

    print(tabulate(runtime_rows, headers=["Dynamic Metric", "Observation"], tablefmt="fancy_grid"))

    if surviving_01 < 50:
        print("\nCRITICAL FORENSIC VERDICT: The Confidence Floor has collapsed!")
        print(f"Only {surviving_01} anchors out of {total_anchors:,} passed the 0.01 threshold!")
        print("This proves that NMS discarded valid boxes because objectness was pushed too low.")


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX Checkpoint Probe")
    parser.add_argument("-f", "--exp_file", default="exps/default/tde_yolox_s.py", type=str)
    parser.add_argument("--ckpt1", type=str, required=True, help="Path to Peak checkpoint (e.g. epoch_6_ckpt.pth)")
    parser.add_argument("--ckpt2", type=str, required=True, help="Path to Collapsed checkpoint (e.g. epoch_15_ckpt.pth)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Run probe in FP16 precision")
    args = parser.parse_args()

    exp = get_exp(args.exp_file)

    print("Loading Checkpoint 1:", args.ckpt1)
    state1, src1, ap1, ep1 = unwrap_ckpt(args.ckpt1)
    print(f"  -> Extracted {src1} (Saved at Epoch {ep1}, AP: {ap1:.3f})")

    print("\nLoading Checkpoint 2:", args.ckpt2)
    state2, src2, ap2, ep2 = unwrap_ckpt(args.ckpt2)
    print(f"  -> Extracted {src2} (Saved at Epoch {ep2}, AP: {ap2:.3f})\n")

    # 1. Run Static Weight Diagnostics
    name1 = f"Epoch {ep1} (Peak)"
    name2 = f"Epoch {ep2} (Collapsed)"
    audit_static_weights(state1, state2, name1=name1, name2=name2)

    # 2. Run Dynamic In-Flight Probing with exact epoch metadata
    probe_runtime_activations(exp, state1, name=name1, epoch=ep1, use_fp16=args.fp16)
    probe_runtime_activations(exp, state2, name=name2, epoch=ep2, use_fp16=args.fp16)


if __name__ == "__main__":
    main()