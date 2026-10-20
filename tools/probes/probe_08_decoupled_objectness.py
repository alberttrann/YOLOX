#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
from yolox.exp import get_exp

def run_probe():
    print("=" * 70)
    print("PROBE 08: DECOUPLED OBJECTNESS STEM & SCALE-NORMALIZED OBJECTNESS LOSS")
    print("=" * 70)

    exp = get_exp("exps/default/tde_yolox_s.py")
    head = exp.get_model().head.cuda().train()

    # 1. Verify Gradient Decoupling: Zero gradients to reg_conv from Objectness Loss
    dummy_x = [
        torch.randn(2, 128, 72, 128, device="cuda", requires_grad=True),
        torch.randn(2, 256, 36, 64, device="cuda", requires_grad=True),
        torch.randn(2, 512, 18, 32, device="cuda", requires_grad=True)
    ]
    p_anchors = [dummy_x[0], dummy_x[1], dummy_x[2]]

    head.zero_grad()
    inputs = (dummy_x, p_anchors, 0.5)
    
    # Forward pass
    dummy_targets = torch.zeros(2, 10, 5, device="cuda")
    dummy_targets[:, 0, 0] = 1.0
    dummy_targets[:, 0, 1:5] = torch.tensor([[100., 100., 50., 50.]])
    
    loss_tuple = head(inputs, dummy_targets, torch.zeros(2, 3, 576, 1024, device="cuda"))
    loss_obj = loss_tuple[3]

    # Backward strictly through objectness loss
    loss_obj.backward(retain_graph=True)

    # Inspect gradients on reg_conv vs obj_preds
    reg_conv_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.reg_convs.parameters())
    obj_conv_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.obj_preds.parameters())

    print(f" • Objectness Loss backpropagated into obj_preds: {obj_conv_has_grad}")
    print(f" • Objectness Loss backpropagated into reg_conv:  {reg_conv_has_grad}")

    assert not reg_conv_has_grad, "CRITICAL LEAK: 190,000 negative objectness anchors are still corrupting reg_conv!"
    assert obj_conv_has_grad, "obj_preds failed to receive objectness gradients!"
    print(" [PASS] Objectness branch is completely decoupled from the coordinate regression trunk.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()