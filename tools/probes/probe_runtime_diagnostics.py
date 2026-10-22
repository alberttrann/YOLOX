#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
TDE-YOLOX v3.6 Real-World Runtime Diagnostic Telemetry Probe
Hooks into intermediate layers during inference on REAL validation images to measure:
  - Atmospheric contrast ratio (h_ratio) on real pixels
  - Effective memory gate activation (g_final) across pyramid levels
  - Stage 1 TTT affine modulation (gamma, beta) after test-time adaptation
  - Live vs. ModelEMA parameter divergence
"""

import argparse
import torch
import torch.nn.functional as F
from yolox.exp import get_exp
from yolox.utils import load_ckpt


def parse_args():
    parser = argparse.ArgumentParser("TDE-YOLOX Runtime Diagnostic Probe")
    parser.add_argument("-f", "--exp_file", default="exps/default/tde_yolox_s.py", type=str)
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/tde_yolox_s/epoch_25_ckpt.pth", type=str)
    parser.add_argument("-n", "--num-batches", default=10, type=int, help="number of batches to inspect")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    return parser.parse_args()


class RuntimeTelemetry:
    def __init__(self):
        self.h_ratio_records = []
        self.gate_records = {0: [], 1: [], 2: []} # P3, P4, P5
        self.gamma_records = []
        self.beta_records = []

    def clear(self):
        self.h_ratio_records.clear()
        for k in self.gate_records:
            self.gate_records[k].clear()
        self.gamma_records.clear()
        self.beta_records.clear()


def run_probe():
    args = parse_args()
    print("\n" + "=" * 80)
    print(" " * 20 + "TDE-YOLOX RUNTIME DIAGNOSTIC TELEMETRY PROBE")
    print("=" * 80)
    print(f" • Target Checkpoint: {args.ckpt}")
    print(f" • Experiment File  : {args.exp_file}\n")

    exp = get_exp(args.exp_file)
    val_loader = exp.get_eval_loader(batch_size=args.batch_size, is_distributed=False)

    # 1. Compare Live vs ModelEMA Parameter Divergence
    raw_ckpt = torch.load(args.ckpt, map_location=args.device)
    if "model" in raw_ckpt and "ema" in raw_ckpt and raw_ckpt["ema"] is not None:
        live_state = raw_ckpt["model"]
        ema_state = raw_ckpt["ema"]

        divergences = []
        proto_divergences = []
        for k in live_state:
            if k in ema_state:
                p_live = live_state[k].float()
                p_ema = ema_state[k].float()
                diff = (p_live - p_ema).norm().item()
                divergences.append(diff)
                if "prototypes" in k:
                    proto_divergences.append(diff)

        avg_div = sum(divergences) / max(1, len(divergences))
        avg_proto_div = sum(proto_divergences) / max(1, len(proto_divergences)) if proto_divergences else 0.0

        print("=" * 80)
        print(" 1. LIVE MODEL vs. MODELEMA SHADOW PARAMETER DIVERGENCE")
        print("=" * 80)
        print(f" • Overall Mean L2 Weight Divergence : {avg_div:.6f}")
        print(f" • Prototype Specific L2 Divergence  : {avg_proto_div:.6f}")
        if avg_proto_div < 1e-4:
            print("   [OK] Prototypes in EMA are in sync with live weights.")
        else:
            print("   [!] Notice: Prototypes differ between EMA and live model.")
        print("=" * 80 + "\n")

    # 2. Instantiate Model and Attach Diagnostic Hooks
    model = exp.get_model().to(args.device).eval()
    ckpt_state = raw_ckpt.get("model", raw_ckpt)
    model = load_ckpt(model, ckpt_state)

    telemetry = RuntimeTelemetry()

    # Hook A: P5 Atmospheric Contrast Ratio (h_ratio)
    def hook_p5(module, input, output):
        # output is (p5_feat, saliency, h_ratio)
        h_ratio = output[2]
        telemetry.h_ratio_records.extend(h_ratio.detach().cpu().flatten().tolist())

    model.backbone.p5_dense_attn.register_forward_hook(hook_p5)

    # Hook B: Memory Banks Final Gate (g_final) across P3, P4, P5
    def make_gate_hook(scale_idx):
        def hook_bank(module, input, output):
            # output is (mem_spatial, gate_spatial, raw_class_scores)
            gate = output[1]
            telemetry.gate_records[scale_idx].extend(gate.detach().cpu().flatten().tolist())
        return hook_bank

    for k in range(3):
        model.head.memory_banks[k].register_forward_hook(make_gate_hook(k))

    # Hook C: TTT Adaptive Stage (Gamma and Beta after inner adaptation)
    def hook_ttt(module, input, output):
        # We read adapted affine parameters directly from aligner
        gamma = module.aligner.gamma.data.detach().cpu().flatten()
        beta = module.aligner.beta.data.detach().cpu().flatten()
        telemetry.gamma_records.append(gamma)
        telemetry.beta_records.append(beta)

    model.backbone.backbone.dark2.register_forward_hook(hook_ttt)

    # 3. Feed Real Validation Batches
    print(f"Running inference telemetry over {args.num_batches} real validation batches ({args.num_batches * args.batch_size} images)...")

    with torch.no_grad():
        for b_idx, (imgs, _, info_imgs, ids) in enumerate(val_loader):
            if b_idx >= args.num_batches:
                break
            imgs = imgs.to(args.device).float()
            _ = model(imgs)

    # 4. Compute and Print Empirical Findings
    print("\n" + "=" * 80)
    print(" 2. EMPIRICAL RUNTIME TELEMETRY ON REAL VALIDATION PIXELS")
    print("=" * 80)

    # Telemetry Metric 1: h_ratio
    if telemetry.h_ratio_records:
        h_t = torch.tensor(telemetry.h_ratio_records)
        print(f" • Atmospheric Contrast Ratio (h_ratio):")
        print(f"     Mean   : {h_t.mean().item():.4f}")
        print(f"     Min/Max: [{h_t.min().item():.4f}, {h_t.max().item():.4f}]")
        print(f"     Std    : {h_t.std().item():.4f}")
        if h_t.mean().item() > 0.60:
            print("     [!] ALERT: h_ratio is high on clean images! The contrast ratio may not be distinguishing clean from fog.")
        elif h_t.mean().item() <= 0.35:
            print("     [OK] Clean Daylight Detected: h_ratio is low (<= 0.35), successfully identifying clean weather.")

    # Telemetry Metric 2: Memory Gate (g_final) across scales
    print("\n • Real-World Memory Gate Injection Strength (g_final in [0.0, 0.70]):")
    scale_names = ["P3 (Stride 8 - Micro)", "P4 (Stride 16 - Medium)", "P5 (Stride 32 - Macro)"]
    for k in range(3):
        if telemetry.gate_records[k]:
            g_t = torch.tensor(telemetry.gate_records[k])
            non_zero_ratio = (g_t > 0.05).float().mean().item() * 100
            print(f"   [{scale_names[k]}]:")
            print(f"     Mean Gate   : {g_t.mean().item():.5f}")
            print(f"     Max Gate    : {g_t.max().item():.5f}")
            print(f"     Active (>0.05): {non_zero_ratio:.2f}% of all spatial tokens")
            if g_t.mean().item() > 0.15:
                print(f"     [!] High Gate Alert: Memory is intervening heavily on clean images!")
            elif g_t.mean().item() <= 0.05:
                print(f"     [OK] Clean Conv Priority: Gate remains near zero on clean data.")

    # Telemetry Metric 3: TTT Affine Modulation (Gamma / Beta)
    if telemetry.gamma_records:
        gammas = torch.stack(telemetry.gamma_records)
        betas = torch.stack(telemetry.beta_records)
        print(f"\n • Stage 1 TTT Dynamic Modulation:")
        print(f"     Gamma (Channel Scale): Mean = {gammas.mean().item():.4f} | Min = {gammas.min().item():.4f} | Max = {gammas.max().item():.4f}")
        print(f"     Beta  (Channel Shift): Mean = {betas.mean().item():.4f} | Min = {betas.min().item():.4f} | Max = {betas.max().item():.4f}")
        if gammas.min().item() < 0.20:
            print("     [!] Channel Compression Warning: Some channels have gamma < 0.20.")
        else:
            print("     [OK] Healthy Channel Transmission: Gamma scales are well-behaved.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_probe()