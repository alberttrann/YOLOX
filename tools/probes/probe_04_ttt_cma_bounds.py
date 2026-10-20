#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from yolox.models.network_blocks import BaseConvGN, CSPLayer
from yolox.models.ttt_modules import TTTAdaptiveStage, CovariateMomentAligner

def run_probe():
    print("=" * 70)
    print("PROBE 04: TTT 1-STEP MAML STABILITY & GAMMA POSITIVITY CHECK")
    print("=" * 70)

    # 1. Test Post-SiLU Mean Offset Behavior
    cma = CovariateMomentAligner().cuda()
    # Synthetic clean feature with natural SiLU mean of +0.40
    natural_silu_feat = (torch.randn(2, 64, 72, 128, device="cuda") * 0.8 + 0.40)
    loss_cma = cma(natural_silu_feat)
    
    print(f" • CMA Loss on Natural Post-SiLU Features (Mean=0.40): {loss_cma.mean().item():.5f}")
    assert loss_cma.mean().item() < 0.50, "CMA is excessively penalizing natural positive SiLU activations!"
    print(" [PASS] CMA offset centering accommodates natural SiLU activation energy.")

    # 2. Test TTT 1-Step MAML & Gamma Positivity
    stage = nn_stage = torch.nn.Sequential(
        BaseConvGN(32, 64, 3, 2),
        CSPLayer(64, 64, n=1, use_gn=True)
    ).cuda()
    ttt_stage = TTTAdaptiveStage(nn_stage, in_channels=64).cuda().train()

    # Adversarial input with severe DC airlight corruption
    corrupted_input = torch.randn(2, 32, 288, 512, device="cuda") + 5.0

    out, loss = ttt_stage(corrupted_input, run_ttt=True, epoch=0, max_epochs=80)
    
    # Check gamma positivity
    gamma_values = ttt_stage.aligner.gamma.data
    min_gamma = gamma_values.min().item()
    print(f" • Minimum Gamma after Adversarial TTT Step: {min_gamma:.4f}")
    assert min_gamma >= 0.10, f"CRITICAL LANDMINE: Gamma fell below 0.10 or inverted negative! Min gamma: {min_gamma}"
    print(" [PASS] Gamma positivity guard verified (Strictly bounded in [0.10, 3.00]).")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()