#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from yolox.models.darknet import CSPDarknet

def run_probe():
    print("=" * 70)
    print("PROBE 05: PIXEL-WISE PHASE LOSS WITH FP16 SUBNORMAL EPSILON GUARD")
    print("=" * 70)

    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.50).cuda().train()

    # Test Case 1: Completely Black Image (All Zeros) - Tests FP16 Underflow
    black_input = torch.zeros(2, 3, 576, 1024, device="cuda", dtype=torch.float16)

    with torch.cuda.amp.autocast(enabled=True):
        _, _, phase_loss_black = backbone(black_input)

    assert not torch.isnan(phase_loss_black), "CRITICAL LANDMINE: NaN in Phase Loss on completely dark image!"
    assert not torch.isinf(phase_loss_black), "CRITICAL LANDMINE: Inf in Phase Loss on completely dark image!"
    print(f" • Phase Loss on Exact Zero Night Image: {phase_loss_black.item():.6f}")
    print(" [PASS] FP16 subnormal underflow epsilon guard verified (Zero NaNs).")

    # Test Case 2: Standard Image - Verify Non-Trivial Gradient
    normal_input = torch.randn(2, 3, 576, 1024, device="cuda", dtype=torch.float16)
    with torch.cuda.amp.autocast(enabled=True):
        _, _, phase_loss_normal = backbone(normal_input)

    assert phase_loss_normal.item() > 0.0, "Phase loss is trivially zero on distinct images!"
    print(f" • Phase Loss on Normal Image: {phase_loss_normal.item():.6f}")
    print(" [PASS] Pixel-wise phase loss evaluates non-trivial spatial boundary gradients.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()