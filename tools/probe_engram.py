#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
TDE-YOLOX Forensic Probe: Engram Memory Bank & Head Deep-Dive
Measures:
  1. Prototype Matrix Geometry: Pairwise Cosine Matrix of class prototypes in checkpoint
  2. Ground-Truth Prototype Retrieval Confusion Matrix across adverse splits
  3. Gating Telemetry (Uncertainty vs. Objectness) partitioned by Object Scale (Small, Med, Large)
  4. FiLM Modulation Injection Power: ||Gate * Mem|| / ||Cls_Feat||
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tabulate import tabulate
from loguru import logger

from yolox.exp import get_exp


CATEGORY_NAMES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]


def inspect_prototype_geometry(model):
    """Computes and displays the pairwise cosine similarity matrix of learned prototypes."""
    m = model.module if hasattr(model, "module") else model
    head = m.head

    print("\n" + "=" * 90)
    print(" ENGRAM PROTOTYPE GEOMETRY IN CHECKPOINT (P4 Level) ")
    print("=" * 90)

    # Inspect prototypes from P4 level (index 1)
    prototypes = head.memory_banks[1].prototypes.data
    p_norm = F.normalize(prototypes, p=2, dim=-1)
    cosine_matrix = torch.matmul(p_norm, p_norm.t()).cpu().numpy()

    # Format table
    table = []
    for i, row in enumerate(cosine_matrix):
        formatted_row = [CATEGORY_NAMES[i]] + [f"{val:+.2f}" if i != j else " 1.00 " for j, val in enumerate(row)]
        table.append(formatted_row)

    headers = ["Class"] + [c[:5] for c in CATEGORY_NAMES]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    # Compute mean off-diagonal correlation (Orthogonality check)
    off_diag = cosine_matrix[~np.eye(9, dtype=bool)]
    print(f"Mean Off-Diagonal Cosine Similarity: {off_diag.mean():+.4f} (Ideal Orthogonal: ~0.0000)")
    print(f"Max Off-Diagonal Collision:          {off_diag.max():+.4f} (Collision Risk if > 0.60)")


def probe_engram_inference(model, eval_loader, num_batches=5):
    """Hooks into TDE_Head to extract real-world retrieval, gating, and FiLM injection metrics."""
    model.eval()
    m = model.module if hasattr(model, "module") else model
    head = m.head

    scale_names = ["P3 (Small)", "P4 (Medium)", "P5 (Macro)"]
    
    # Storage
    gate_stats = {s: {"uncertainty": [], "objectness": [], "effective_gate": []} for s in scale_names}
    signal_ratios = {s: [] for s in scale_names}
    class_retrieval_hits = {c: {"correct": 0, "total": 0, "confused_with": []} for c in range(9)}

    logger.info(f"Probing Engram dynamics across {num_batches} adverse batches...")

    with torch.no_grad():
        for b_idx, batch in enumerate(eval_loader):
            if b_idx >= num_batches:
                break
            inps = batch[0].cuda().float()
            targets = batch[1].cuda()  # [B, max_labels, 5] (class, x, y, w, h)

            # Forward PAFPN
            fpn_outs = m.backbone(inps)

            for k, (x, stem, cls_conv, reg_conv) in enumerate(zip(fpn_outs, head.stems, head.cls_convs, head.reg_convs)):
                scale_name = scale_names[k]
                x_stem = stem(x)

                # Reg & Obj
                reg_feat = reg_conv(x_stem)
                obj_output = head.obj_preds[k](reg_feat)
                obj_mask = torch.sigmoid(obj_output).detach()  # [B, 1, H, W]

                # Cls & Latent
                cls_feat = cls_conv(x_stem)
                B, C, H, W = cls_feat.shape
                cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
                latent_vec = head.latent_projectors[k](cls_feat_flat)

                # Uncertainty & Gate
                uncertainty = head.uncertainty_gates[k](latent_vec)  # [B, HW, 1]
                obj_mask_flat = obj_mask.view(B, 1, -1).permute(0, 2, 1)  # [B, HW, 1]

                # Retrieval
                memory_gated, norm_logits = head.memory_banks[k](latent_vec, uncertainty, obj_mask_flat)
                
                # FiLM Modulation Norms
                memory_inflated = F.linear(memory_gated, head.latent_projectors[k].weight.t())
                memory_normed = head.memory_norms[k](memory_inflated)
                
                gate_spatial = uncertainty.view(B, H, W, 1).permute(0, 3, 1, 2)
                mem_spatial = memory_normed.view(B, H, W, C).permute(0, 3, 1, 2)

                injected_signal = gate_spatial * mem_spatial
                
                # Compute Norms
                norm_cls = cls_feat.norm(dim=1).mean().item()
                norm_inj = injected_signal.norm(dim=1).mean().item()
                ratio = (norm_inj / (norm_cls + 1e-8)) * 100.0
                signal_ratios[scale_name].append(ratio)

                # Collect Gate Values (where objectness > 0.1 to focus on candidate regions)
                fg_mask = (obj_mask_flat > 0.1).squeeze(-1)
                if fg_mask.sum() > 0:
                    gate_stats[scale_name]["uncertainty"].append(uncertainty[fg_mask].mean().item())
                    gate_stats[scale_name]["objectness"].append(obj_mask_flat[fg_mask].mean().item())
                    eff_gate = (uncertainty * obj_mask_flat)[fg_mask].mean().item()
                    gate_stats[scale_name]["effective_gate"].append(eff_gate)

                # Measure Retrieval Accuracy on Ground-Truth Target Locations
                for b_i in range(B):
                    valid_labels = targets[b_i][targets[b_i].sum(dim=-1) > 0]
                    for tgt in valid_labels:
                        cls_idx = int(tgt[0].item())
                        if cls_idx >= 9:
                            continue
                        cx, cy = tgt[1].item(), tgt[2].item()
                        
                        # Map cx, cy to grid coordinates
                        stride = head.strides[k]
                        gx = int(cx / stride)
                        gy = int(cy / stride)

                        if 0 <= gx < W and 0 <= gy < H:
                            token_idx = gy * W + gx
                            retrieved_class = norm_logits[b_i, token_idx].argmax().item()
                            class_retrieval_hits[cls_idx]["total"] += 1
                            if retrieved_class == cls_idx:
                                class_retrieval_hits[cls_idx]["correct"] += 1
                            else:
                                class_retrieval_hits[cls_idx]["confused_with"].append(retrieved_class)

    # 1. Print Multi-Scale Gating Table
    gate_table = []
    for s in scale_names:
        u_mean = np.mean(gate_stats[s]["uncertainty"]) if gate_stats[s]["uncertainty"] else 0.0
        o_mean = np.mean(gate_stats[s]["objectness"]) if gate_stats[s]["objectness"] else 0.0
        eff_mean = np.mean(gate_stats[s]["effective_gate"]) if gate_stats[s]["effective_gate"] else 0.0
        inj_ratio = np.mean(signal_ratios[s]) if signal_ratios[s] else 0.0

        gate_table.append([
            s,
            f"{u_mean:.4f}",
            f"{o_mean:.4f}",
            f"{eff_mean:.4f}",
            f"{inj_ratio:.2f}%"
        ])

    print("\n" + "=" * 80)
    print(" ENGRAM MULTI-SCALE GATING & INJECTION RATIO (Candidate Regions: Obj > 0.10) ")
    print("=" * 80)
    headers = ["Pyramid Scale", "Mean Uncertainty (U)", "Mean Objectness (Obj)", "Effective Gate (U*Obj)", "FiLM Injection Ratio %"]
    print(tabulate(gate_table, headers=headers, tablefmt="fancy_grid"))

    # 2. Print Class-by-Class Retrieval Accuracy Table
    class_table = []
    for c_i in range(9):
        total = class_retrieval_hits[c_i]["total"]
        correct = class_retrieval_hits[c_i]["correct"]
        acc = (correct / total * 100.0) if total > 0 else 0.0
        confused_list = class_retrieval_hits[c_i]["confused_with"]
        most_common_confusion = (
            CATEGORY_NAMES[max(set(confused_list), key=confused_list.count)] if confused_list else "None"
        )

        class_table.append([
            CATEGORY_NAMES[c_i],
            total,
            f"{acc:.2f}%",
            most_common_confusion
        ])

    print("\n" + "=" * 70)
    print(" ENGRAM CLASS PROTOTYPE RETRIEVAL ACCURACY ON ADVERSE TARGETS ")
    print("=" * 70)
    headers = ["Class", "Sampled Tokens", "Top-1 Retrieval Acc %", "Primary Class Confusion"]
    print(tabulate(class_table, headers=headers, tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser("Engram Deep-Dive Forensic Probe")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
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

    # 1. Inspect Prototype Geometry
    inspect_prototype_geometry(model)

    # 2. Probe Dynamic Inference on Adverse Dataset
    eval_loader = exp.get_eval_loader(args.batch_size, is_distributed=False)
    probe_engram_inference(model, eval_loader, num_batches=10)


if __name__ == "__main__":
    main()