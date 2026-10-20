#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import traceback

PROBES = [
    ("Probe 01: Multiscale Forward-Backward & AMP Gradient Sweeper", "tools.probes.probe_01_multiscale_shapes"),
    ("Probe 02: P5 Step-Zero Identity & Continuous 2D-RoPE Equivariance", "tools.probes.probe_02_p5_rope_identity"),
    ("Probe 03: P4 Vectorized Target Rasterizer & P3 Zero-Delta Bypass Check", "tools.probes.probe_03_bqsa_integrity"),
    ("Probe 04: TTT 1-Step MAML Stability & Gamma Positivity Check", "tools.probes.probe_04_ttt_cma_bounds"),
    ("Probe 05: Pixel-Wise Phase Loss with FP16 Subnormal Epsilon Guard", "tools.probes.probe_05_pixel_phase_loss"),
    ("Probe 06: ModelEMA Phase-Coherent Slerp & S^127 Normalization", "tools.probes.probe_06_modelema_slerp"),
    ("Probe 07: Memory Gate Polarity, Scale Parity, & Objectness Bypass", "tools.probes.probe_07_engram_memory_gate"),
    ("Probe 08: Decoupled Objectness Stem & Scale-Normalized Loss", "tools.probes.probe_08_decoupled_objectness"),
    ("Probe 09: MuSGD Parameter Group Partitioning & No-Decay Ledger", "tools.probes.probe_09_musgd_partition"),
]

def main():
    print("\n" + "#" * 80)
    print(" " * 15 + "TDE-YOLOX v3.6 PRE-FLIGHT VERIFICATION ORCHESTRATOR")
    print("#" * 80 + "\n")

    results = []
    for title, module_path in PROBES:
        try:
            mod = __import__(module_path, fromlist=["run_probe"])
            mod.run_probe()
            results.append((title, "PASSED", "\033[92mPASS\033[0m"))
        except Exception as e:
            traceback.print_exc()
            results.append((title, "FAILED", f"\033[91mFAIL ({e})\033[0m"))
            print(f"\n[ABORT] Critical invariant failure in: {title}\n")
            sys.exit(1)

    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL PRE-FLIGHT AUDIT SUMMARY")
    print("=" * 80)
    for title, _, status in results:
        print(f" {status} │ {title}")
    print("=" * 80)
    print("\n\033[92m>>> ALL RUNTIME INVARIANTS MATHEMATICALLY CERTIFIED. SAFE TO LAUNCH RETRAIN. <<<\033[0m\n")

if __name__ == "__main__":
    main()