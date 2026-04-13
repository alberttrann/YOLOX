#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from yolox.exp import get_exp
from yolox.models.tribrid_neck import C2f_Tribrid

def inspect_model_internals(exp_file, ckpt_path, num_batches=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval()
    model.to(device)
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    print("\n" + "="*60)
    print("1. TRIBRID NECK GATE ANALYSIS (Global Context Strength)")
    print("="*60)
    
    gate_values = []
    for name, module in model.named_modules():
        if isinstance(module, C2f_Tribrid):
            # The gate is a pure linear scalar
            active_multiplier = module.gate.item()
            gate_values.append(active_multiplier)
            print(f"Module: {name:<20} | Active Multiplier (Linear): {active_multiplier:>7.4f}")
    
    avg_gate = np.mean(np.abs(gate_values))
    print("-" * 60)
    if avg_gate < 0.01:
        print("DIAGNOSIS: The Tribrid Neck is effectively SHUT OFF. (Local Convs Only)")
    elif avg_gate > 0.05:
        print("DIAGNOSIS: The Tribrid Neck is ACTIVE. Global context is modifying features.")
    else:
        print("DIAGNOSIS: The Tribrid Neck is MARGINAL. It provides a tiny residual hint.")

    print("\n" + "="*60)
    print("2. TTT-STEM META-LR ANALYSIS")
    print("="*60)
    
    try:
        ttt_stage = model.backbone.backbone.dark2
        # Apply the softplus to get the ACTUAL learning rate used in math
        raw_lr = ttt_stage.ttt_lr.item()
        actual_lr = F.softplus(torch.tensor(raw_lr)).item() + 1e-4
        print(f"Raw Meta-LR Parameter: {raw_lr:.6f}")
        print(f"Actual Math Meta-LR:   {actual_lr:.6f}")
    except Exception as e:
        print(f"Could not locate TTT LR. Error: {e}")

    print("\n" + "="*60)
    print(f"3. TTT-STEM FEATURE SHIFT ANALYSIS (Averaged over {num_batches} batches)")
    print("="*60)
    
    val_loader = exp.get_eval_loader(batch_size=4, is_distributed=False, legacy=False)
    
    total_mad = 0.0
    total_rel_shift = 0.0
    
    actual_backbone = model.backbone.backbone
    ttt_stage = actual_backbone.dark2
    
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(val_loader):
            if i >= num_batches: break
            imgs = imgs.to(device)
            
            stem_out = actual_backbone.stem(imgs)
            
            # Static Backbone
            static_features = ttt_stage(stem_out, run_ttt=False)
            
            # Adapted Backbone
            with torch.enable_grad():
                adapted_features = ttt_stage(stem_out, run_ttt=True)
            
            mad = torch.mean(torch.abs(adapted_features.detach() - static_features.detach())).item()
            rel_shift = mad / (torch.mean(torch.abs(static_features.detach())).item() + 1e-6)
            
            total_mad += mad
            total_rel_shift += rel_shift

    avg_mad = total_mad / num_batches
    avg_rel_shift = total_rel_shift / num_batches

    print(f"Mean Absolute Difference (MAD) caused by TTT: {avg_mad:.4f}")
    print(f"Relative Feature Shift: {avg_rel_shift * 100:.2f}%")
    print("-" * 60)
    
    if avg_mad < 0.001:
        print("DIAGNOSIS: TTT is mathematically DEAD. The inner loop is doing nothing.")
    elif avg_rel_shift > 0.05: 
        print("DIAGNOSIS: TTT is HIGHLY ACTIVE. It is radically altering the feature distribution.")
    else:
        print("DIAGNOSIS: TTT is applying a micro-calibration (Normal).")


    print("\n" + "="*60)
    print("4. ENGRAM HEAD ANALYSIS (Is Memory Injecting Identity?)")
    print("="*60)

    # --- Pytorch Hooks to capture internal Engram Tensors ---
    engram_data = {'gate': [], 'cls_feat': [], 'mem_feat': [], 'obj_out': []}
    
    def get_gate_hook():
        def hook(m, i, o): engram_data['gate'].append(o.detach())
        return hook
        
    def get_cls_hook():
        def hook(m, i, o): engram_data['cls_feat'].append(o.detach())
        return hook
        
    def get_mem_hook():
        def hook(m, i, o): engram_data['mem_feat'].append(o.detach())
        return hook

    def get_obj_hook():
        def hook(m, i, o): engram_data['obj_out'].append(o.detach())
        return hook

    handles = []
    # Attach hooks to the P5 scale (Index 2)
    handles.append(model.head.uncertainty_gates[2].register_forward_hook(get_gate_hook()))
    handles.append(model.head.cls_convs[2].register_forward_hook(get_cls_hook()))
    handles.append(model.head.memory_banks[2].register_forward_hook(get_mem_hook()))
    handles.append(model.head.obj_preds[2].register_forward_hook(get_obj_hook()))

    # Run one batch to trigger hooks
    with torch.no_grad():
        imgs, _, _, _ = next(iter(val_loader))
        imgs = imgs.to(device)
        _ = model(imgs)
    
    for h in handles: h.remove()

    # --- TRUE MAGNITUDE MATCHING CALCULATION ---
    scale_idx = 2 # P5 Scale
    gate_val = engram_data['gate'][0]        # [B, HW, 1]
    cls_f = engram_data['cls_feat'][0]       # [B, C, H, W]
    mem_raw = engram_data['mem_feat'][0]     # [B, HW, latent_dim]
    obj_out = engram_data['obj_out'][0]      # [B, 1, H, W]
    
    B, C, H, W = cls_f.shape
    cls_flat = cls_f.permute(0, 2, 3, 1).reshape(B, H*W, C)
    
    # 1. Re-inflate memory
    proj_weight = model.head.latent_projectors[scale_idx].weight.t()
    mem_inflated = F.linear(mem_raw, proj_weight) # [B, HW, C]
    
    # 2. Emulate the exact scaling logic from TDE_Head.forward()
    norm_conv = torch.norm(cls_flat, p=2, dim=-1, keepdim=True)
    norm_mem = torch.norm(mem_inflated, p=2, dim=-1, keepdim=True)
    memory_scaled = mem_inflated * (norm_conv / (norm_mem + 1e-6)) # This is the TRUE injected feature
    
    # 3. Filter by Objectness (We only care about Gate/Memory on actual objects, not sky)
    obj_scores = torch.sigmoid(obj_out).view(B, -1)
    obj_mask = obj_scores > 0.3 # Threshold for "Likely an object"
    
    if obj_mask.sum() == 0:
        print("Warning: No objects detected with >0.3 confidence in this sample batch.")
        print("Run the script again to sample a different batch.")
        return

    # Extract metrics only for object pixels
    valid_gates = gate_val.view(B, -1)[obj_mask]
    valid_conv = cls_flat[obj_mask]
    valid_mem_scaled = memory_scaled[obj_mask]

    # Calculate final magnitudes on Object Pixels
    conv_mag = torch.norm(valid_conv, dim=-1).mean().item()
    mem_mag = torch.norm(valid_mem_scaled, dim=-1).mean().item()
    
    avg_gate = valid_gates.mean().item()
    max_gate = valid_gates.max().item()
    min_gate = valid_gates.min().item()

    print(f"Scale P5 Average Uncertainty Gate (\u03B2) on Objects:  {avg_gate:.4f}")
    print(f"Scale P5 Max Uncertainty Gate on Objects:       {max_gate:.4f}")
    print(f"Scale P5 Min Uncertainty Gate on Objects:       {min_gate:.4f}\n")
    
    print(f"Avg L2 Norm of Convolution Features:      {conv_mag:.4f}")
    print(f"Avg L2 Norm of SCALED Memory Injection:   {mem_mag:.4f}")
    
    if conv_mag > 0:
        # The true injection ratio includes the gate multiplier
        true_injection_ratio = ((mem_mag * avg_gate) / conv_mag) * 100
        print(f"True Memory Injection Ratio (Memory * \u03B2 / Conv): {true_injection_ratio:.2f}%")
    print("-" * 60)
    
    if avg_gate < 0.05:
        print("DIAGNOSIS: Engram Gate is CLOSED. Model is relying purely on Convolutions.")
    elif avg_gate > 0.4:
        print("DIAGNOSIS: Engram Gate is HIGHLY ACTIVE. Model is heavily replacing Conv features with Prototypes.")
    else:
        print("DIAGNOSIS: Engram Gate is BLENDING. Model is augmenting Convs with Memory identity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TDE-YOLOX Internal Diagnostics")
    parser.add_argument("-f", "--exp_file", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("-n", "--num_batches", type=int, default=3, help="Batches to average TTT shift over")
    args = parser.parse_args()
    
    inspect_model_internals(args.exp_file, args.ckpt, args.num_batches)