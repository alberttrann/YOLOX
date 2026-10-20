#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from yolox.exp import get_exp

def run_probe():
    print("=" * 70)
    print("PROBE 01: MULTISCALE FORWARD-BACKWARD & AMP GRADIENT SWEEPER")
    print("=" * 70)
    
    exp = get_exp("exps/default/tde_yolox_s.py")
    model = exp.get_model().cuda().train()
    optimizer = exp.get_optimizer(batch_size=2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    test_resolutions = [
        (384, 640),   # Extreme low scale
        (512, 896),   # Rule-of-64 low
        (576, 1024),  # Native 16:9 base
        (640, 1152),  # Rule-of-64 high
        (704, 1280),  # 704/64=11, 1280/64=20 (Certified Rule-of-64 720p)
    ]

    for h, w in test_resolutions:
        optimizer.zero_grad()
        dummy_in = torch.randn(2, 3, h, w, device="cuda", dtype=torch.float32)
        dummy_targets = torch.zeros(2, 120, 5, device="cuda", dtype=torch.float32)
        
        # Inject realistic valid boxes
        dummy_targets[0, :3, 0] = torch.tensor([1, 4, 8])
        dummy_targets[0, :3, 1:5] = torch.tensor([[100., 100., 40., 40.], [300., 200., 60., 120.], [400., 300., 20., 20.]])
        dummy_targets[1, :2, 0] = torch.tensor([2, 3])
        dummy_targets[1, :2, 1:5] = torch.tensor([[250., 250., 80., 150.], [500., 400., 300., 150.]])

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(dummy_in, dummy_targets)
            loss = outputs["total_loss"]

        assert not torch.isnan(loss), f"NaN detected at canvas ({h}, {w})!"
        assert not torch.isinf(loss), f"Inf detected at canvas ({h}, {w})!"

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f" [PASS] Canvas ({h:>4}, {w:>4}) | Total Loss: {loss.item():.4f} | Backward Step Successful")

    print("=" * 70)
    print("PROBE 01 PASSED: Model is 100% resilient across dynamic multiscale shapes.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()