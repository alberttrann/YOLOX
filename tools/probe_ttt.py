#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
TDE-YOLOX Forensic Probe: TTT Stage 1 Telemetry
Measures:
  1. Inner-loop gradient norms ||∇_θ L_inner|| across GroupNorm layers
  2. Actual parameter shift magnitude: ||Δθ|| = ||η · ∇_θ L_inner||
  3. Feature representation drift: 1 - CosSim(F_initial, F_adapted)
  4. Inner loss reduction: L_inner(pre) vs. L_inner(post)
  5. Saliency mask coverage percentage on real adverse scenes
"""

import argparse
import os
import torch
import torch.nn.functional as F
from tabulate import tabulate
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import setup_logger


def probe_ttt_batch(model, inps, noise_std=0.08, mask_ratio=0.20):
    """Deep-dives into a single batch to extract TTT inner dynamics."""
    model.eval()
    
    # Access Darknet & dark2
    m = model.module if hasattr(model, "module") else model
    darknet = m.backbone.backbone
    dark2_stage = darknet.dark2  # TTTAdaptiveStage
    backbone_stage = dark2_stage.backbone_stage  # dark2 base
    # Diagnostic Override: Set effective step size to 2.0 to overcome gradient attenuation
    for k in dark2_stage.ttt_lrs:
        dark2_stage.ttt_lrs[k].data.fill_(2.0)

    # 1. Forward Stem
    with torch.no_grad():
        x_stem = darknet.stem(inps)

    # 2. Extract Stage 1 Initial Features & Saliency Mask
    with torch.no_grad():
        feat_initial = backbone_stage(x_stem)
        clean_target = feat_initial.detach()
        mask = dark2_stage._get_relative_contrast_mask(clean_target, ratio=mask_ratio)
        mask_coverage = (mask > 0.5).float().mean().item() * 100.0
        noise_map = torch.randn_like(clean_target) * noise_std

    # 3. Compute Inner Loss & Gradients
    backbone_params = dict(backbone_stage.named_parameters())
    backbone_buffers = dict(backbone_stage.named_buffers())
    proj_params = dict(dark2_stage.projector.named_parameters())

    adapt_backbone = {
        k: v for k, v in backbone_params.items()
        if ("gn" in k or "norm" in k) and v.is_floating_point()
    }
    fixed_state = {
        **{k: v for k, v in backbone_params.items() if k not in adapt_backbone},
        **backbone_buffers
    }

    # Enable grads for inner step
    with torch.enable_grad():
        x_curr = x_stem.detach()
        x_curr.requires_grad = True

        from torch.func import functional_call, grad

        def inner_loss_fn(p_adapt, p_proj):
            full_params = {**p_adapt, **fixed_state}
            feat = functional_call(backbone_stage, full_params, x_curr)
            feat_noisy = feat + noise_map
            feat_corrupted = feat_noisy * (1.0 - mask) + dark2_stage.projector.mask_token * mask
            rec = functional_call(dark2_stage.projector, p_proj, feat_corrupted)
            mse = F.mse_loss(rec, clean_target)
            cos = F.cosine_similarity(rec, clean_target, dim=1).mean()
            return mse + 0.1 * (1.0 - cos)

        initial_inner_loss = inner_loss_fn(adapt_backbone, proj_params).item()
        grads_backbone = grad(inner_loss_fn, argnums=0)(adapt_backbone, proj_params)

    # 4. Measure Parameter Shift & Layer-wise Dynamics
    table_data = []
    total_param_norm = 0.0
    total_shift_norm = 0.0

    updated_params = {**backbone_params, **backbone_buffers}

    for name, p in adapt_backbone.items():
        g = grads_backbone[name]
        g_norm = g.norm(2).item()
        p_norm = p.norm(2).item()

        lr_key = name.replace(".", "_")
        lr = dark2_stage.ttt_lrs[lr_key].item() if lr_key in dark2_stage.ttt_lrs else 0.02
        shift = lr * g
        shift_norm = shift.norm(2).item()
        rel_shift = (shift_norm / (p_norm + 1e-8)) * 100.0

        updated_params[name] = p - shift.to(p.dtype)

        total_param_norm += p_norm ** 2
        total_shift_norm += shift_norm ** 2

        table_data.append([
            name.split("backbone_stage.")[-1],
            f"{p_norm:.4f}",
            f"{g_norm:.4e}",
            f"{lr:.4f}",
            f"{shift_norm:.4e}",
            f"{rel_shift:.4f}%"
        ])

    # 5. Measure Post-Adaptation Feature Shift
    with torch.no_grad():
        feat_adapted = functional_call(backbone_stage, updated_params, x_stem)
        post_inner_loss = inner_loss_fn(
            {k: updated_params[k] for k in adapt_backbone}, proj_params
        ).item()
        cos_sim_features = F.cosine_similarity(feat_initial, feat_adapted, dim=1).mean().item()
        feature_drift = (1.0 - cos_sim_features) * 100.0

    total_param_norm = total_param_norm ** 0.5
    total_shift_norm = total_shift_norm ** 0.5
    global_rel_shift = (total_shift_norm / (total_param_norm + 1e-8)) * 100.0

    summary = {
        "mask_coverage": mask_coverage,
        "initial_inner_loss": initial_inner_loss,
        "post_inner_loss": post_inner_loss,
        "loss_delta": initial_inner_loss - post_inner_loss,
        "global_param_shift": global_rel_shift,
        "feature_drift_pct": feature_drift,
        "table": table_data
    }
    return summary


def main():
    parser = argparse.ArgumentParser("TTT Forensic Probe")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Experiment file (e.g., test_acdc_snow.py)")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint (best_ckpt.pth)")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size for inspection")
    args = parser.parse_args()

    exp = get_exp(args.exp_file, None)
    model = exp.get_model()

    logger.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict)
    model.cuda()
    model.eval()

    eval_loader = exp.get_eval_loader(args.batch_size, is_distributed=False)
    batch_iter = iter(eval_loader)

    logger.info("Executing TTT Forensic Probe across 3 consecutive batches...")
    for step in range(3):
        batch = next(batch_iter)
        inps = batch[0].cuda().float()

        res = probe_ttt_batch(model, inps)
        
        print("\n" + "=" * 80)
        print(f" BATCH {step + 1} TTT FORENSIC TELEMETRY ")
        print("=" * 80)
        print(f"Mask Coverage:             {res['mask_coverage']:.2f}% (Expected: ~20%)")
        print(f"Inner Loss (Pre-Step):     {res['initial_inner_loss']:.6f}")
        print(f"Inner Loss (Post-Step):    {res['post_inner_loss']:.6f} (Delta: {res['loss_delta']:+.6f})")
        print(f"Global Parameter Shift:    {res['global_param_shift']:.6f}%")
        print(f"Feature Map Drift (1-Cos): {res['feature_drift_pct']:.6f}%")
        print("\nLayer-by-Layer GroupNorm Parameter Dynamics:")
        headers = ["Layer Name", "||θ||", "||∇L||", "η (LR)", "||Δθ||", "Rel Shift %"]
        print(tabulate(res["table"], headers=headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()