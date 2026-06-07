#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#tools/inspect_internals_2.py: A comprehensive diagnostic tool for TDE-YOLOX internals, analyzing TTT dynamics, Tribrid gating, and Engram injection in a single pass. Provides actionable insights for researchers to understand and optimize the interplay of components.
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
    
    # Use strict=False to allow loading partially compatible checkpoints
    msg = model.load_state_dict(ckpt["model"], strict=False)
    if msg.missing_keys:
        logger.warning(f"Missing keys (using defaults): {msg.missing_keys}")
    if msg.unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint (ignored): {msg.unexpected_keys}")
    
    print("\n" + "="*70)
    print("1. TRIBRID NECK GATE ANALYSIS (Global Context Strength)")
    print("="*70)
    
    gate_values = []
    for name, module in model.named_modules():
        if isinstance(module, C2f_Tribrid):
            # Calculate weight of global context branch via softmax of the router's bias
            with torch.no_grad():
                # Access the final linear layer of the dynamic router
                router_bias = module.dynamic_router.router[2].bias
                weights = F.softmax(router_bias, dim=0)
                active_multiplier = weights[1].item()
            gate_values.append(active_multiplier)
            print(f"Module: {name:<20} | Bias-based Gate (Global): {active_multiplier:>7.4f}")
    
    if not gate_values:
        print("No C2f_Tribrid modules found.")
        return

    avg_gate = np.mean(gate_values)
    print("-" * 70)
    if avg_gate < 0.01:
        print("DIAGNOSIS: Tribrid Neck effectively SHUT OFF. (Local Convs Only)")
    elif avg_gate > 0.05:
        print("DIAGNOSIS: Tribrid Neck ACTIVE. Global context modifying features.")
    else:
        print("DIAGNOSIS: Tribrid Neck MARGINAL. Providing tiny residual hint.")

    print("\n" + "="*70)
    print("2. TTT-STEM META-LR ANALYSIS")
    print("="*70)
    
    try:
        ttt_stage = model.backbone.backbone.dark2
        if hasattr(ttt_stage, "ttt_lrs") and len(ttt_stage.ttt_lrs) > 0:
            lrs = [p.data for p in ttt_stage.ttt_lrs.values()]
            avg_raw_lr = torch.stack(lrs).mean().item()
            actual_lr = F.softplus(torch.tensor(avg_raw_lr)).item() + 1e-4
            print(f"Detected {len(lrs)} Meta-LR parameters in ParameterDict.")
            print(f"Average Raw Meta-LR: {avg_raw_lr:.6f}")
            print(f"Actual Math Meta-LR:   {actual_lr:.6f}")
        else:
            print("TTTAdaptiveStage found, but ttt_lrs is empty or missing.")
    except Exception as e:
        print(f"Could not locate TTT LR. Error: {e}")

    print("\n" + "="*70)
    print(f"3. PANOPTIC FEATURE SHIFT ANALYSIS (Averaged over {num_batches} batches)")
    print("="*70)
    
    val_loader = exp.get_eval_loader(batch_size=4, is_distributed=False, legacy=False)
    
    # Trackers
    total_mad_stage1 = 0.0
    total_rel_shift_stage1 = 0.0
    total_mad_head = 0.0
    total_rel_shift_head = 0.0
    total_mask_density = 0.0
    
    actual_backbone = model.backbone.backbone
    ttt_stage = actual_backbone.dark2
    
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(val_loader):
            if i >= num_batches: break
            imgs = imgs.to(device)
            
            # --- RUN 1: STATIC MODEL (TTT OFF) ---
            fpn_static = model.backbone(imgs)
            head_in_static = fpn_static[0] # P3 scale for baseline
            
            # Extract Stage 1 static features directly
            stem_out = actual_backbone.stem(imgs)
            stage1_static = ttt_stage(stem_out, run_ttt=False)

            # --- RUN 2: ADAPTIVE MODEL (TTT ON) ---
            # We must enable grad for the inner loop to run
            with torch.enable_grad():
                # Pass ttt_prob=1.0 to force the TTT loop to run for this measurement
                fpn_adapted = model.backbone(imgs, ttt_prob=1.0)
                head_in_adapted = fpn_adapted[0]
                
                # We extract the mask generated during this pass
                # _get_robust_variance_mask is deterministic, so calling it here matches the inner loop
                mask = ttt_stage._get_robust_variance_mask(stage1_static.detach())
                
                # Re-run just Stage 1 to get isolated adapted features
                stage1_adapted = ttt_stage(stem_out, run_ttt=True)

            # --- CALCULATE STAGE 1 SHIFT (The Source) ---
            mad_s1 = torch.mean(torch.abs(stage1_adapted.detach() - stage1_static.detach())).item()
            rel_shift_s1 = mad_s1 / (torch.mean(torch.abs(stage1_static.detach())).item() + 1e-6)
            total_mad_stage1 += mad_s1
            total_rel_shift_stage1 += rel_shift_s1
            
            # --- CALCULATE HEAD SHIFT (The Ripple Effect) ---
            mad_head = torch.mean(torch.abs(head_in_adapted.detach() - head_in_static.detach())).item()
            rel_shift_head = mad_head / (torch.mean(torch.abs(head_in_static.detach())).item() + 1e-6)
            total_mad_head += mad_head
            total_rel_shift_head += rel_shift_head
            
            # --- CALCULATE MASK DENSITY ---
            # How much of the image is the Indexer targeting?
            mask_density = mask.mean().item() * 100
            total_mask_density += mask_density

    # Averages
    avg_rel_s1 = total_rel_shift_stage1 / num_batches
    avg_rel_head = total_rel_shift_head / num_batches
    avg_mask = total_mask_density / num_batches

    print(f"Variance Masking Density:            {avg_mask:.2f}% of pixels targeted.")
    print(f"Stage 1 Relative Shift (Source):     {avg_rel_s1 * 100:.2f}%")
    print(f"Head Input Relative Shift (Ripple):  {avg_rel_head * 100:.2f}%")
    print("-" * 70)
    
    if avg_rel_s1 < 0.05:
        print("DIAGNOSIS (S1): TTT is applying a micro-calibration (Normal).")
    else:
        print("DIAGNOSIS (S1): TTT is HIGHLY ACTIVE. Radically altering feature distribution.")
        
    if avg_rel_head > (avg_rel_s1 * 0.1):
        print("DIAGNOSIS (Network): TTT Signal SURVIVES the PAFPN. The Head sees the adaptation.")
    else:
        print("DIAGNOSIS (Network): TTT Signal WASHED OUT. The deep backbone absorbs the adaptation.")


    print("\n" + "="*70)
    print("4. ENGRAM HEAD ANALYSIS (Identity Injection Dynamics)")
    print("="*70)

    # --- Hooks for Engram ---
    engram_data = {'gate': [], 'cls_feat': [], 'mem_feat': [], 'obj_out': []}
    
    def get_gate_hook(): return lambda m, i, o: engram_data['gate'].append(o.detach())
    def get_cls_hook(): return lambda m, i, o: engram_data['cls_feat'].append(o.detach())
    def get_mem_hook(): return lambda m, i, o: engram_data['mem_feat'].append(o.detach())
    def get_obj_hook(): return lambda m, i, o: engram_data['obj_out'].append(o.detach())

    handles = []
    # Attach to P5 (Index 2)
    handles.append(model.head.uncertainty_gates[2].register_forward_hook(get_gate_hook()))
    handles.append(model.head.cls_convs[2].register_forward_hook(get_cls_hook()))
    handles.append(model.head.memory_banks[2].register_forward_hook(get_mem_hook()))
    handles.append(model.head.obj_preds[2].register_forward_hook(get_obj_hook()))

    with torch.no_grad():
        imgs, _, _, _ = next(iter(val_loader))
        imgs = imgs.to(device)
        _ = model(imgs)
    
    for h in handles: h.remove()

    scale_idx = 0 
    
    handles = []
    # Attach to P3 (Index 0)
    handles.append(model.head.uncertainty_gates[scale_idx].register_forward_hook(get_gate_hook()))
    handles.append(model.head.cls_convs[scale_idx].register_forward_hook(get_cls_hook()))
    handles.append(model.head.memory_banks[scale_idx].register_forward_hook(get_mem_hook()))
    handles.append(model.head.obj_preds[scale_idx].register_forward_hook(get_obj_hook()))
    gate_val = engram_data['gate'][0]        
    cls_f = engram_data['cls_feat'][0]       
    mem_raw = engram_data['mem_feat'][0]     
    obj_out = engram_data['obj_out'][0]      
    
    B, C, H, W = cls_f.shape
    cls_flat = cls_f.permute(0, 2, 3, 1).reshape(B, H*W, C)
    
    proj_weight = model.head.latent_projectors[scale_idx].weight.t()
    mem_inflated = F.linear(mem_raw, proj_weight) 
    
    # Exact scaling math
    norm_conv = torch.norm(cls_flat, p=2, dim=-1, keepdim=True)
    norm_mem = torch.norm(mem_inflated, p=2, dim=-1, keepdim=True)
    memory_scaled = mem_inflated * (norm_conv / (norm_mem + 1e-6)) 
    
    obj_scores = torch.sigmoid(obj_out).view(B, -1)
    obj_mask = obj_scores > 0.01
    
    if obj_mask.sum() == 0:
        print("Warning: No objects >0.01 confidence in sample batch. Skipping Gate metrics.")
    else:
        valid_gates = gate_val.view(B, -1)[obj_mask]
        valid_conv = cls_flat[obj_mask]
        valid_mem_scaled = memory_scaled[obj_mask]

        conv_mag = torch.norm(valid_conv, dim=-1).mean().item()
        mem_mag = torch.norm(valid_mem_scaled, dim=-1).mean().item()
        
        avg_gate = valid_gates.mean().item()
        max_gate = valid_gates.max().item()

        print(f"Scale P5 Average Uncertainty Gate (\u03B2) on Objects: {avg_gate:.4f}")
        print(f"Scale P5 Max Uncertainty Gate on Objects:      {max_gate:.4f}\n")
        
        print(f"Avg L2 Norm of Convolution Features:      {conv_mag:.4f}")
        print(f"Avg L2 Norm of SCALED Memory Injection:   {mem_mag:.4f}")
        
        if conv_mag > 0:
            true_injection_ratio = ((mem_mag * avg_gate) / conv_mag) * 100
            print(f"True Memory Injection Ratio:              {true_injection_ratio:.2f}%")
        print("-" * 70)
        
        if avg_gate < 0.05:
            print("DIAGNOSIS: Engram Gate CLOSED. Relying purely on Convolutions.")
        elif avg_gate > 0.4:
            print("DIAGNOSIS: Engram Gate HIGHLY ACTIVE. Replacing Conv features with Prototypes.")
        else:
            print("DIAGNOSIS: Engram Gate BLENDING. Augmenting Convs with Memory identity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TDE-YOLOX Panoptic Diagnostics")
    parser.add_argument("-f", "--exp_file", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("-n", "--num_batches", type=int, default=3)
    args = parser.parse_args()
    
    inspect_model_internals(args.exp_file, args.ckpt, args.num_batches)