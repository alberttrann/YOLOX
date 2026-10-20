#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time
import torch
from yolox.models.tribrid_neck import C2f_BQSA_P4, C2f_BQSA_P3

def run_probe():
    print("=" * 70)
    print("PROBE 03: P4 VECTORIZED TARGET RASTERIZER & P3 ZERO-DELTA BYPASS CHECK")
    print("=" * 70)

    p4_block = C2f_BQSA_P4(c1=512, c2=256).cuda()
    p3_block = C2f_BQSA_P3(c1=256, c2=128).cuda()

    # 1. Test Padded Target Hazard
    # Image with ZERO objects (all 120 slots are zero-padded)
    padded_targets = torch.zeros(2, 120, 5, device="cuda")
    
    # Rasterize on 9x16 grid
    target_mask = p4_block._rasterize_gt_blocks(padded_targets, batch_size=2, num_h=9, num_w=16, device="cuda")
    
    assert target_mask.sum().item() == 0.0, (
        f"CRITICAL LANDMINE DETECTED: Block (0,0) was activated by padded zero-boxes! "
        f"Target mask sum: {target_mask.sum().item()}"
    )
    print(" [PASS] Padded targets are strictly masked out. Block (0,0) is unpoisoned.")

    # 2. Benchmark Rasterizer GPU Vectorization (Check for CPU-GPU stalls)
    test_targets = torch.randn(16, 120, 5, device="cuda") * 200.0
    test_targets[:, :, 3:5] = torch.clamp(test_targets[:, :, 3:5].abs(), 10.0, 100.0)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(50):
        _ = p4_block._rasterize_gt_blocks(test_targets, batch_size=16, num_h=9, num_w=16, device="cuda")
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) / 50.0 * 1000.0

    print(f" • Vectorized 16-Batch Rasterization Latency: {elapsed_ms:.3f} ms")
    assert elapsed_ms < 1.0, f"Rasterizer is too slow ({elapsed_ms:.2f} ms). Suspected CPU-GPU sync stall!"
    print(" [PASS] Rasterizer runs fully vectorized on GPU with zero synchronization barriers.")

    # 3. Test P3 Zero-Delta Bypass
    assert p3_block.proj.bias is None, "CRITICAL LANDMINE: p3_block.proj has bias=True! Zero-delta bypass will leak bias!"
    
    zero_context = torch.zeros(2, 64, 72, 128, device="cuda")
    proj_out = p3_block.proj(zero_context)
    assert (proj_out == 0.0).all(), "p3_block.proj(0) != 0! Bias leak detected!"
    print(" [PASS] P3 Zero-Delta bypass verified: proj(0) is strictly 0.0000.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()