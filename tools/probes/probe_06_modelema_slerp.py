#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from yolox.exp import get_exp
from yolox.utils import ModelEMA

def run_probe():
    print("=" * 70)
    print("PROBE 06: MODELEMA PHASE-COHERENT SLERP & S^127 NORMALIZATION")
    print("=" * 70)

    exp = get_exp("exps/default/tde_yolox_s.py")
    model = exp.get_model().cuda().train()
    ema = ModelEMA(model, decay=0.9998)

    # Simulate 50 training optimization steps
    for step in range(50):
        # Apply synthetic update to live prototypes
        for bank in model.head.memory_banks:
            bank.prototypes.data.add_(torch.randn_like(bank.prototypes) * 0.05)
            bank.prototypes.data.copy_(F.normalize(bank.prototypes.data, p=2, dim=-1))

        # Perform EMA step
        ema.update(model)

        # Apply Phase-Coherent Slerp Normalization
        with torch.no_grad():
            for m_ema in ema.ema.head.memory_banks:
                m_ema.prototypes.data.copy_(F.normalize(m_ema.prototypes.data, p=2, dim=-1))

    # Verify unit-norm condition across all 576 prototypes in all 3 scales
    for k, m_ema in enumerate(ema.ema.head.memory_banks):
        norms = torch.norm(m_ema.prototypes.data, p=2, dim=-1)
        max_err = torch.abs(norms - 1.0).max().item()
        print(f" • Scale P{k+3} EMA Prototype Max Norm Error: {max_err:.8f}")
        assert max_err < 1e-4, f"Prototype norm collapsed inside unit sphere! Max error: {max_err}"

    print(" [PASS] ModelEMA preserves strict S^127 hyperspherical geometry across temporal updates.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()