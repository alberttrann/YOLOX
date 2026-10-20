#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from yolox.models.tribrid_neck import P5ExclusiveDenseAttention, RotaryEmbedding2D

def run_probe():
    print("=" * 70)
    print("PROBE 02: P5 STEP-ZERO IDENTITY & CONTINUOUS 2D-RoPE EQUIVARIANCE")
    print("=" * 70)

    # 1. Test P5 Step-0 Identity Pass-Through
    p5_block = P5ExclusiveDenseAttention(dim=512).cuda().eval()
    x = torch.randn(2, 512, 18, 32, device="cuda")

    with torch.no_grad():
        out, _, _ = p5_block(x)
        diff = torch.abs(out - x).max().item()

    print(f" • P5 Step-0 Max Output Difference: {diff:.6f}")
    assert diff < 0.05, f"Step-0 identity violated! Difference is {diff:.4f} (Expected < 0.05)"
    print(" [PASS] P5 Attention initializes as a clean identity pass-through.")

    # 2. Test Continuous 2D-RoPE Aspect Invariance
    rope = RotaryEmbedding2D(dim=128).cuda()
    
    # Coordinates at low resolution vs high resolution
    q_low, k_low = torch.ones(1, 4, 16*28, 128, device="cuda"), torch.ones(1, 4, 16*28, 128, device="cuda")
    q_high, k_high = torch.ones(1, 4, 20*36, 128, device="cuda"), torch.ones(1, 4, 20*36, 128, device="cuda")

    out_q_low, _ = rope(q_low, k_low, H=16, W=28)
    out_q_high, _ = rope(q_high, k_high, H=20, W=36)

    # Spatial interpolation to check phase angle coherence across grid resizing
    map_low = out_q_low.view(1, 4, 16, 28, 128).permute(0, 1, 4, 2, 3).reshape(512, 16, 28)
    map_high = out_q_high.view(1, 4, 20, 36, 128).permute(0, 1, 4, 2, 3).reshape(512, 20, 36)
    map_high_interp = F.interpolate(map_high.unsqueeze(0), size=(16, 28), mode="bilinear", align_corners=False).squeeze(0)

    sim = F.cosine_similarity(map_low.flatten(), map_high_interp.flatten(), dim=0).item()
    print(f" • Continuous 2D-RoPE Scale Phase Correlation: {sim:.5f}")
    assert sim > 0.98, f"RoPE phase angles drifted across scales! Correlation: {sim:.4f} (Expected > 0.98)"
    print(" [PASS] Continuous 2D-RoPE preserves physical phase geometry across canvas scales.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()