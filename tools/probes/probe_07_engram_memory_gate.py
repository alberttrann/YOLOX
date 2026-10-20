#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from yolox.models.engram_head import AntiCollisionEngramBank

def run_probe():
    print("=" * 70)
    print("PROBE 07: MEMORY GATE POLARITY, SCALE PARITY, & OBJECTNESS BYPASS")
    print("=" * 70)

    bank_p3 = AntiCollisionEngramBank(num_classes=9, num_modes=64, latent_dim=128, temperature=32.0).cuda().eval()
    bank_p4 = AntiCollisionEngramBank(num_classes=9, num_modes=64, latent_dim=128, temperature=24.0).cuda().eval()
    bank_p5 = AntiCollisionEngramBank(num_classes=9, num_modes=64, latent_dim=128, temperature=16.0).cuda().eval()

    weak_obj_mask = torch.ones(1, 1, 18, 32, device="cuda") * 0.02 # Faint object (2% obj score)

    # 1. Test Case A: Confident Prototype Match (Anchor and Local aligned to prototype 0)
    with torch.no_grad():
        proto_target = bank_p5.prototypes[0].view(1, 128, 1, 1).expand(1, 128, 18, 32).clone()
        anchor_confident = proto_target
        h_local_confident = proto_target
        _, gate_confident, _ = bank_p5(anchor_confident, h_local_confident, weak_obj_mask, epoch=50, h_s=0.5)

    avg_gate_conf = gate_confident.mean().item()
    print(f" • Gate Output on High-Certainty Target (Weak Objectness 0.02): {avg_gate_conf:.4f}")
    assert avg_gate_conf > 0.30, f"CRITICAL INVERSION: Gate stayed locked on confident target! Gate: {avg_gate_conf}"
    print(" [PASS] Confident prototype matches cleanly bypass weak objectness and open memory gate.")

    # 2. Test Case B: Baseline Negative Logit Parity Across Scales (Orthogonal Random Inputs)
    with torch.no_grad():
        anchor_random = torch.randn(1, 128, 18, 32, device="cuda")
        ortho_query = torch.randn(1, 128, 18, 32, device="cuda")
        _, _, scores_p3 = bank_p3(anchor_random, ortho_query, weak_obj_mask, h_s=0.5)
        _, _, scores_p4 = bank_p4(anchor_random, ortho_query, weak_obj_mask, h_s=0.5)
        _, _, scores_p5 = bank_p5(anchor_random, ortho_query, weak_obj_mask, h_s=0.5)

    mean_p3 = scores_p3.mean().item()
    mean_p4 = scores_p4.mean().item()
    mean_p5 = scores_p5.mean().item()

    print(f" • Baseline Negative Logits -> P3: {mean_p3:.3f} | P4: {mean_p4:.3f} | P5: {mean_p5:.3f}")
    assert abs(mean_p3 - mean_p5) < 0.35, (
        f"CRITICAL ASYMMETRY: P3 and P5 baseline logits are mismatched! "
        f"Diff: {abs(mean_p3 - mean_p5):.3f} (Stride 8 will trigger false-positive explosion)"
    )
    print(" [PASS] Mode-normalization decoupling guarantees logit energy parity across pyramid levels.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()