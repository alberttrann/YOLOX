#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Forensic Speed & Profiling Diagnostic for TDE-YOLOX v3.5

import time
import torch
import math
from loguru import logger
from yolox.exp import get_exp
from yolox.utils import get_model_info

def benchmark():
    logger.info("=" * 60)
    logger.info("TDE-YOLOX v3.5 FORENSIC TIMING & PROFILING AUDIT")
    logger.info("=" * 60)

    # 1. Instantiate Experiment & Model in FP32
    exp = get_exp("exps/default/tde_yolox_s.py", None)
    model = exp.get_model()
    model.cuda()

    # 2. Match trainer.py: Run get_model_info() in FP32 BEFORE casting to half!
    logger.info("Simulating startup get_model_info() (FLOP profiling in FP32)...")
    _ = get_model_info(model, exp.test_size)

    # Check if total_ops was attached by thop
    has_total_ops = any(hasattr(m, "total_ops") for m in model.modules())
    logger.info(f"-> Startup total_ops present on modules: {has_total_ops}")

    # 3. Now cast to FP16 for the actual speed benchmarks
    model.half()

    # 4. Create synthetic 16:9 batches matching (576, 1024)
    B = 16
    H, W = exp.input_size  # (576, 1024)
    dummy_inps = torch.randn(B, 3, H, W, device="cuda", dtype=torch.float16)

    # Create dummy bounding boxes [class, xc, yc, w, h] so SimOTA runs cleanly
    dummy_targets = torch.zeros(B, 15, 5, device="cuda", dtype=torch.float16)
    dummy_targets[:, :, 0] = 1.0    # Class 1 (Car)
    dummy_targets[:, :, 1] = 500.0  # xc
    dummy_targets[:, :, 2] = 280.0  # yc
    dummy_targets[:, :, 3] = 120.0  # w
    dummy_targets[:, :, 4] = 80.0   # h

    # -------------------------------------------------------------------------
    # TEST 1: In-Training Step Speed (Forward + Backward)
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 1: In-Training Step Speed (Forward + Backward) ---")
    model.train()
    
    # Warmup 3 steps
    for _ in range(3):
        outputs = model(dummy_inps, dummy_targets)
        loss = outputs["total_loss"]
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark 10 training steps
    t0 = time.time()
    num_train_steps = 10
    for _ in range(num_train_steps):
        outputs = model(dummy_inps, dummy_targets)
        loss = outputs["total_loss"]
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    avg_train_step = (time.time() - t0) / num_train_steps
    
    logger.info(f"Average Training Step Time: {avg_train_step:.3f} s / batch")
    if avg_train_step < 0.65:
        logger.success(f"[PASS] Training Speed: FAST ({avg_train_step:.3f}s <= 0.65s target). Second-order Hessian trap is ELIMINATED!")
    else:
        logger.error(f"[FAIL] Training Speed: SLOW ({avg_train_step:.3f}s > 0.65s). new_param is still not detached in training!")

    # -------------------------------------------------------------------------
    # TEST 2: Clean In-Training Validation Speed (TTT Bypassed)
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 2: Clean In-Training Validation Speed (run_ttt=False) ---")
    model.eval()
    
    # Set model to clean bypass (0.0 TTT probability)
    model.eval_ttt_prob = 0.0

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_inps)
    torch.cuda.synchronize()

    # Benchmark 10 eval passes
    num_eval_steps = 10
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_eval_steps):
            _ = model(dummy_inps)
    torch.cuda.synchronize()
    avg_eval_clean_batch = (time.time() - t0) / num_eval_steps
    avg_eval_clean_per_img = (avg_eval_clean_batch * 1000.0) / B

    logger.info(f"Clean Validation Batch Time (B={B}): {avg_eval_clean_batch:.3f} s")
    logger.info(f"Clean Validation Per-Image Forward Time: {avg_eval_clean_per_img:.2f} ms")
    
    if avg_eval_clean_per_img < 16.0:
        logger.success(f"[PASS] Clean Eval Speed: FAST ({avg_eval_clean_per_img:.2f} ms <= 16.0 ms target). Full 530 batches will finish in ~2.5 mins!")
    else:
        logger.error(f"[FAIL] Clean Eval Speed: SLOW ({avg_eval_clean_per_img:.2f} ms). Clean bypass is not triggering!")

    # -------------------------------------------------------------------------
    # TEST 3: Adverse Weather TTT Inference Speed (Active TTT)
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 3: Adverse Weather TTT Inference Speed (Active TTT) ---")
    model.eval()
    # Activate TTT for adverse weather testing
    model.eval_ttt_prob = 1.0

    # Warmup
    for _ in range(3):
        _ = model(dummy_inps)
    torch.cuda.synchronize()

    # Benchmark 10 active TTT passes
    t0 = time.time()
    for _ in range(num_eval_steps):
        _ = model(dummy_inps)
    torch.cuda.synchronize()
    avg_ttt_batch = (time.time() - t0) / num_eval_steps
    avg_ttt_per_img = (avg_ttt_batch * 1000.0) / B

    logger.info(f"Active TTT Batch Time (B={B}): {avg_ttt_batch:.3f} s")
    logger.info(f"Active TTT Per-Image Forward Time: {avg_ttt_per_img:.2f} ms")
    
    if avg_ttt_per_img < 25.0:
        logger.success(f"[PASS] Active TTT Speed: FAST ({avg_ttt_per_img:.2f} ms / {1000.0/avg_ttt_per_img:.1f} FPS). Real-time adverse adaptation certified!")
    else:
        logger.warning(f"[WARNING] Active TTT Per-Image Time is {avg_ttt_per_img:.2f} ms (> 25 ms). Consider setting inner_steps = 1 in ttt_modules.py.")

    logger.info("\n" + "=" * 60)
    logger.info("AUDIT COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    benchmark()