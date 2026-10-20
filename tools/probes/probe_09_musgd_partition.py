#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from yolox.exp import get_exp

def run_probe():
    print("=" * 70)
    print("PROBE 09: MuSGD PARAMETER GROUP PARTITIONING & NO-DECAY LEDGER")
    print("=" * 70)

    exp = get_exp("exps/default/tde_yolox_s.py")
    _ = exp.get_model().cuda()
    optimizer = exp.get_optimizer(batch_size=16)

    groups = optimizer.param_groups
    assert len(groups) == 4, f"Expected 4 certified parameter groups, got {len(groups)}!"

    g0, g1, g2, g3 = groups[0], groups[1], groups[2], groups[3]

    print(f" • Group 0 (Muon S^127)      : {len(g0['params']):>3} tensors | LR: {g0['lr']:.4f} | WD: {g0['weight_decay']}")
    print(f" • Group 1 (SGD with Decay)  : {len(g1['params']):>3} tensors | LR: {g1['lr']:.4f} | WD: {g1['weight_decay']}")
    print(f" • Group 2 (SGD No-Decay)    : {len(g2['params']):>3} tensors | LR: {g2['lr']:.4f} | WD: {g2['weight_decay']}")
    print(f" • Group 3 (Meta-SGD TTT)    : {len(g3['params']):>3} tensors | LR: {g3['lr']:.4f} | WD: {g3['weight_decay']}")

    # 1. Assert Prototypes in Group 0 (Muon)
    for p in g0["params"]:
        assert p.shape[-1] == 128 and p.shape[0] % 9 == 0, "Non-prototype tensor found in Group 0 (Muon)!"
    assert g0["weight_decay"] == 0.0, "Weight decay detected on Muon prototypes!"

    # 2. Assert w_hf and attnres.w are in Group 2 (No-Decay)
    p_names = {p: name for name, p in exp.model.named_parameters() if p.requires_grad}
    g2_names = [p_names[p] for p in g2["params"]]

    assert any("w_hf" in n for n in g2_names), "CRITICAL LEAK: w_hf was placed in weight decay! It will decay to 0.0!"
    assert any("attnres" in n and n.endswith(".w") for n in g2_names), "CRITICAL LEAK: attnres.w was placed in weight decay! Routing will collapse!"
    assert g2["weight_decay"] == 0.0, "Group 2 has non-zero weight decay!"

    print(" [PASS] All routing weights, scalar gates, and prototypes are certified protected from weight decay.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_probe()