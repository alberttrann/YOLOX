#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Forensic Dynamic In-Flight Hook Telemetry Engine for TDE-YOLOX v3.5/v3.6

import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from yolox.exp import get_exp
from yolox.utils.checkpoint import load_ckpt


class InFlightTelemetryCollector:
    def __init__(self):
        self.hooks = []
        self.data = {
            "stem_out": None,
            "cma_gamma_delta": None,
            "cma_beta_delta": None,
            "cma_cov_violations": None,
            "cma_channel_stds": None,
            "p5_entropy_ratio": None,
            "p4_block_scores": None,
            "p4_pool_mask_ratio": None,
            "p4_topk_idx": None,
            "attnres_n3_weights": None,
            "attnres_n4_weights": None,
            "attnres_n5_weights": None,
            "engram_uncertainty": [],
            "engram_chordal_dist": [],
            "engram_gate_final": [],
            "cls_feat_pre_norm": [],
            "cls_feat_post_norm": [],
            "proto_assist_mag": [],
            "raw_conv_mag": []
        }

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def register_telemetry_hooks(model, collector):
    m = model.module if hasattr(model, "module") else model
    collector.clear()

    # -------------------------------------------------------------
    # 1. STEM HOOK (Koschmieder DC Airlight Centering)
    # -------------------------------------------------------------
    def hook_stem(module, inp, out):
        with torch.no_grad():
            f = out.float()
            collector.data["stem_mean"] = f.mean(dim=[-2, -1]).mean().item()
            collector.data["stem_std"] = f.std(dim=[-2, -1]).mean().item()

    collector.hooks.append(m.backbone.backbone.stem.register_forward_hook(hook_stem))

    # -------------------------------------------------------------
    # 2. TTT-CMA ADAPTIVE STAGE HOOK (Stage 1 Dark2)
    # -------------------------------------------------------------
    def hook_ttt(module, inp, out):
        with torch.no_grad():
            adapted_out, cma_loss = out
            f = adapted_out.float()
            B, C, H, W = f.shape
            
            # Covariance Matrix Violation Ratio (|Σ_ij| > 0.30)
            f_sub = f[:, :, ::2, ::2]
            N = float(f_sub.shape[-2] * f_sub.shape[-1])
            mu = f_sub.mean(dim=[-2, -1], keepdim=True)
            f_centered = (f_sub - mu).reshape(B, C, -1)
            sigma = torch.bmm(f_centered, f_centered.transpose(1, 2)) / max(1.0, N - 1.0)
            
            eye = torch.eye(C, device=f.device).unsqueeze(0)
            off_diag = torch.abs(sigma * (1.0 - eye))
            violations = (off_diag > 0.30).float().sum().item() / max(1.0, float(B * C * (C - 1)))
            collector.data["cma_cov_violations"] = violations

            # Channel Energy Variance
            var_diag = torch.diagonal(sigma, dim1=-2, dim2=-1)
            stds = torch.sqrt(torch.clamp(var_diag, min=1e-5)).cpu().numpy().flatten()
            collector.data["cma_channel_stds"] = (stds.min(), stds.mean(), stds.max())

    collector.hooks.append(m.backbone.backbone.dark2.register_forward_hook(hook_ttt))

    # -------------------------------------------------------------
    # 3. P5 DENSE ATTENTION HOOK (Optical Shannon Entropy)
    # -------------------------------------------------------------
    def hook_p5(module, inp, out):
        with torch.no_grad():
            _, _, h_ratio = out
            collector.data["p5_entropy_ratio"] = h_ratio.float().cpu().item()

    collector.hooks.append(m.backbone.p5_dense_attn.register_forward_hook(hook_p5))

    # -------------------------------------------------------------
    # 4. P4 BQSA ROUTER HOOK (Block Saliency & Gumbel Selection)
    # -------------------------------------------------------------
    def hook_p4_bqsa(module, inp, out):
        with torch.no_grad():
            p4_out, topk_idx, _ = out
            collector.data["p4_topk_idx"] = topk_idx.cpu().numpy()

    collector.hooks.append(m.backbone.p4_bqsa.register_forward_hook(hook_p4_bqsa))

    # -------------------------------------------------------------
    # 5. SCALE-ATTNRES HIGHWAY HOOKS (Depth Softmax Ratios)
    # -------------------------------------------------------------
    def make_hook_attnres(name):
        def hook_attnres(module, inp, out):
            with torch.no_grad():
                sources = inp[0]
                K = torch.stack([module.norm(s) for s in sources], dim=0)
                w_safe = torch.clamp(module.w, -5.0, 5.0)
                logits = torch.einsum('c, m b c h w -> m b h w', w_safe, K) * module.scale
                weights = F.softmax(logits.float(), dim=0)  # [M, B, H, W]
                mean_weights = weights.mean(dim=[-3, -2, -1]).cpu().numpy()
                collector.data[f"attnres_{name}_weights"] = mean_weights.round(3).tolist()
        return hook_attnres

    collector.hooks.append(m.backbone.attnres_n3.register_forward_hook(make_hook_attnres("n3")))
    collector.hooks.append(m.backbone.attnres_n4.register_forward_hook(make_hook_attnres("n4")))
    collector.hooks.append(m.backbone.attnres_n5.register_forward_hook(make_hook_attnres("n5")))

    # -------------------------------------------------------------
    # 6. ENGRAM MEMORY BANK HOOKS (P3, P4, P5 Identity Restoration)
    # -------------------------------------------------------------
    def make_hook_engram(bank_idx):
        def hook_engram(module, inp, out):
            with torch.no_grad():
                mem_spatial, gate_spatial, raw_scores = out
                g = gate_spatial.float().cpu().flatten()
                collector.data["engram_gate_final"].append(g)
        return hook_engram

    for i, bank in enumerate(m.head.memory_banks):
        collector.hooks.append(bank.register_forward_hook(make_hook_engram(i)))


def probe_dynamic_inference(exp, ckpt_path, epoch=0, use_fp16=True, batch_size=16):
    print("=" * 100)
    print(f"DYNAMIC IN-FLIGHT TELEMETRY PROBE: {ckpt_path}")
    print(f"Resolution: {exp.test_size} | Precision: {'FP16' if use_fp16 else 'FP32'} | Epoch: {epoch}")
    print("=" * 100)

    # 1. Instantiate Model
    model = exp.get_model()
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        m.set_meta_training_state(epoch, exp.max_epoch)

    # 2. Load Checkpoint (EMA Priority)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"] is not None:
        print("  -> Loading ModelEMA shadow weights...")
        ckpt_state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt and ckpt["model"] is not None:
        print("  -> Loading Standard model weights...")
        ckpt_state = ckpt["model"]
    else:
        ckpt_state = ckpt

    while isinstance(ckpt_state, dict) and "model" in ckpt_state and isinstance(ckpt_state["model"], dict):
        ckpt_state = ckpt_state["model"]

    model = load_ckpt(model, ckpt_state)
    model.cuda()
    if use_fp16:
        model.half()
    model.eval()

    # 3. Setup Telemetry Hooks
    collector = InFlightTelemetryCollector()
    register_telemetry_hooks(model, collector)

    # 4. Load Real Batch from Val DataLoader
    val_loader = exp.get_eval_loader(batch_size=batch_size, is_distributed=False)
    inps, targets, info_imgs, ids = next(iter(val_loader))
    inps = inps.cuda()
    if use_fp16:
        inps = inps.half()

    # 5. Execute Forward Pass
    with torch.no_grad():
        outputs = model(inps)

    # Unregister hooks immediately
    collector.clear()

    # 6. Parse and Report In-Flight Telemetry
    obj_confs = outputs[..., 4].float().cpu().flatten()
    cls_confs = outputs[..., 5:].max(dim=-1).values.float().cpu().flatten()
    combined_scores = (obj_confs * cls_confs).flatten()

    total_anchors = len(combined_scores)
    pass_01 = (combined_scores >= 0.01).sum().item()
    pass_001 = (combined_scores >= 0.001).sum().item()
    pass_0001 = (combined_scores >= 0.0001).sum().item()

    # Aggregate Engram Gates
    all_gates = torch.cat(collector.data["engram_gate_final"]) if collector.data["engram_gate_final"] else torch.tensor([0.0])
    gate_mean = all_gates.mean().item()
    gate_max = all_gates.max().item()
    gate_p99 = torch.quantile(all_gates, 0.99).item()

    # Group Summary Tables
    telemetry_table = [
        # Group 1: Sensor & Stem
        ["Group 1: FocusGN Stem Mean (DC Offset)", f"{collector.data.get('stem_mean', 0.0):.4f}", "Zero-Centered DC" if abs(collector.data.get('stem_mean', 0.0)) < 0.10 else "DC Drift"],
        ["Group 1: FocusGN Stem Std Dev", f"{collector.data.get('stem_std', 0.0):.4f}", "Healthy Dynamic Range"],
        
        # Group 2: CMA-TTT Stage 1
        ["Group 2: CMA Covariance Violation Ratio (|Σ_ij| > 0.30)", f"{collector.data.get('cma_cov_violations', 0.0)*100:.2f}%", "Healthy (<10%)" if collector.data.get('cma_cov_violations', 0.0) < 0.10 else "High Cross-Channel Noise"],
        ["Group 2: CMA Channel Energy Band [Min, Mean, Max]", f"[{collector.data.get('cma_channel_stds', (0,0,0))[0]:.2f}, {collector.data.get('cma_channel_stds', (0,0,0))[1]:.2f}, {collector.data.get('cma_channel_stds', (0,0,0))[2]:.2f}]", "Healthy [0.5, 2.0]"],

        # Group 3: Neck BQSA & AttnRes
        ["Group 3: P5 Optical Entropy Ratio (H(s) / ln(N))", f"{collector.data.get('p5_entropy_ratio', 0.0):.4f}", "Atmospheric Haze Indicator"],
        ["Group 3: ScaleAttnRes N3 Highway Softmax [P3, C3]", f"{collector.data.get('attnres_n3_weights')}", "Balanced" if max(collector.data.get('attnres_n3_weights', [1.0])) < 0.85 else "SEVERELY POLARIZED"],
        ["Group 3: ScaleAttnRes N4 Highway Softmax [P4, Down_N3, C4]", f"{collector.data.get('attnres_n4_weights')}", "Balanced" if max(collector.data.get('attnres_n4_weights', [1.0])) < 0.85 else "SEVERELY POLARIZED"],
        ["Group 3: ScaleAttnRes N5 Highway Softmax [P5, Down_N4, C5]", f"{collector.data.get('attnres_n5_weights')}", "Balanced" if max(collector.data.get('attnres_n5_weights', [1.0])) < 0.85 else "SEVERELY POLARIZED"],

        # Group 4: Head & Memory Bank
        ["Group 4: Engram Gate Activity (g_final) Mean", f"{gate_mean:.5f}", "Dormant (<0.02)" if gate_mean < 0.02 else "Active"],
        ["Group 4: Engram Gate Activity (g_final) 99th Percentile", f"{gate_p99:.4f}", "Selective Spike" if gate_p99 > 0.10 else "Permanently Closed"],
        ["Group 4: Engram Gate Activity (g_final) Max", f"{gate_max:.4f}", "Full Range Accessible" if gate_max > 0.50 else "Restricted"],

        # Group 5: Output Anchors & Confidences
        ["Group 5: Total Evaluated Grid Anchors", f"{total_anchors:,}", "16:9 Rule-of-64 Grid"],
        ["Group 5: Mean Raw Objectness Prob σ(obj)", f"{obj_confs.mean():.4f} (Max: {obj_confs.max():.4f})", "Prior Scale"],
        ["Group 5: Mean Top-1 Class Prob", f"{cls_confs.mean():.4f} (Max: {cls_confs.max():.4f})", "Category Certainty"],
        ["Group 5: Mean Combined Confidence (obj * cls)", f"{combined_scores.mean():.6f}", "Overall Score"],
        ["Group 5: Anchors Passing test_conf >= 0.01", f"{pass_01:,} ({pass_01/total_anchors*100:.2f}%)", "Healthy" if pass_01 > 500 else "COLLAPSED CONFIDENCE"],
        ["Group 5: Anchors Passing adverse test_conf >= 0.001", f"{pass_001:,} ({pass_001/total_anchors*100:.2f}%)", "Adverse Retained"],
        ["Group 5: Anchors Passing ultra-low conf >= 0.0001", f"{pass_001:,} ({pass_0001/total_anchors*100:.2f}%)", "Residual Bounding Box Floor"],
    ]

    print(tabulate(telemetry_table, headers=["Pipeline Subsystem / Telemetry Layer", "Observation", "Diagnostic Meaning"], tablefmt="fancy_grid"))
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX Dynamic Hook Telemetry Profiler")
    parser.add_argument("-f", "--exp_file", default="exps/default/tde_yolox_s.py", type=str)
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Checkpoint to probe dynamically")
    parser.add_argument("-e", "--epoch", type=int, default=0, help="Epoch context to set in model")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for probe pass")
    parser.add_argument("--fp16", action="store_true", default=True, help="Run probe in FP16 precision")
    args = parser.parse_args()

    exp = get_exp(args.exp_file)
    probe_dynamic_inference(exp, args.ckpt, epoch=args.epoch, use_fp16=args.fp16, batch_size=args.batch_size)


if __name__ == "__main__":
    main()