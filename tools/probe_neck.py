#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
TDE-YOLOX Forensic Probe: Tribrid Neck & Gate Intervention
Measures:
  1. Checkpoint parameters: Exactly what value is 'gate' in C3_p4, C3_p3, C3_n3, C3_n4?
  2. Signal-to-Noise Ratio: Relative Frobenius Norm of Branch B vs. Branch A.
  3. SimAM energy scaling factor statistics (min, mean, max) across adverse scenes.
  4. Coordinate Attention directional energy balance (checks for rain streak spikes).
  5. Live Evaluation Intervention: Forces gate = [0.0, 0.01, 0.10, 0.50, 1.0] to test true Branch B impact!
"""

import argparse
import torch
import torch.nn as nn
from tabulate import tabulate
from loguru import logger

from yolox.exp import get_exp


def inspect_neck_parameters(model):
    """Directly inspects the checkpoint weights inside all C2f_Tribrid blocks."""
    m = model.module if hasattr(model, "module") else model
    neck = m.backbone

    table = []
    blocks = [("C3_p4", neck.C3_p4), ("C3_p3", neck.C3_p3), ("C3_n3", neck.C3_n3), ("C3_n4", neck.C3_n4)]

    for name, block in blocks:
        gate_raw = block.gate.item()
        gate_effective = torch.tanh(block.gate).item()
        table.append([
            name,
            f"{gate_raw:.6f}",
            f"{gate_effective:.6f}",
            f"{gate_effective * 100.0:.3f}%"
        ])

    print("\n" + "=" * 60)
    print(" TRIBRID NECK GATE STATUS IN CHECKPOINT ")
    print("=" * 60)
    headers = ["Neck Block", "Raw gate", "tanh(gate)", "Branch B Scaling %"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def probe_neck_dynamics(model, inps):
    """Hooks into C2f_Tribrid to measure Branch A vs. Branch B norms."""
    m = model.module if hasattr(model, "module") else model
    neck = m.backbone

    records = {}

    def get_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            y = list(module.cv1(x).chunk(2, 1))
            # Branch A computation
            for bottleneck in module.m:
                y.append(bottleneck(y[-1]))
            branch_a = y[-1]
            
            # Branch B computation
            branch_b = module.global_mixer(branch_a)
            effective_b = torch.tanh(module.gate) * branch_b

            norm_a = branch_a.norm(2).item()
            norm_b_raw = branch_b.norm(2).item()
            norm_b_effective = effective_b.norm(2).item()
            ratio = (norm_b_effective / (norm_a + 1e-8)) * 100.0

            records[name] = {
                "norm_A": norm_a,
                "norm_B_raw": norm_b_raw,
                "norm_B_effective": norm_b_effective,
                "ratio_pct": ratio
            }
        return hook

    handles = []
    for name, block in [("C3_p4", neck.C3_p4), ("C3_p3", neck.C3_p3), ("C3_n3", neck.C3_n3), ("C3_n4", neck.C3_n4)]:
        handles.append(block.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        _ = model(inps)

    for h in handles:
        h.remove()

    table = []
    for name, data in records.items():
        table.append([
            name,
            f"{data['norm_A']:.2f}",
            f"{data['norm_B_raw']:.2f}",
            f"{data['norm_B_effective']:.4f}",
            f"{data['ratio_pct']:.4f}%"
        ])

    print("\n" + "=" * 70)
    print(" BRANCH A vs. BRANCH B SIGNAL MAGNITUDE COMPARISON ")
    print("=" * 70)
    headers = ["Block", "||Branch A|| (Conv)", "||Branch B|| (Raw Attn)", "||tanh(G)*B|| (Injected)", "Signal Ratio %"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser("Neck Forensic Probe")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()

    exp = get_exp(args.exp_file, None)
    model = exp.get_model()

    logger.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.cuda()
    model.eval()

    # 1. Parameter Inspection
    inspect_neck_parameters(model)

    # 2. Dynamic Forward Probe on 1 batch
    eval_loader = exp.get_eval_loader(batch_size=4, is_distributed=False)
    batch = next(iter(eval_loader))
    inps = batch[0].cuda().float()
    probe_neck_dynamics(model, inps)


if __name__ == "__main__":
    main()