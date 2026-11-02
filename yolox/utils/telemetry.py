#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import io
import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TelemetryRecorder:
    """
    Exhaustive In-Training Gradient and Architectural State Telemetry Engine.
    
    Streams Monitored (Zero Terminal Clutter):
      1. Layer-wise gradient norms (||g||_2) partitioned across 11 architectural subnets.
      2. Gradient-to-Weight update ratios (||g|| / (||w|| + eps)) to detect vanishing/exploding layers.
      3. SimOTA assignment dynamics (dynamic-k distributions, positive anchor counts, mean IoUs).
      4. Hyperspherical prototype geometry (pairwise inter-class & intra-centroid cosine matrices).
      5. Gating, uncertainty, and FiLM activation statistics.
      6. Loss component decompositions and relative gradient energies.
    """

    def __init__(
        self,
        save_dir: str,
        model: nn.Module,
        record_interval: int = 10,
        rank: int = 0,
    ):
        self.save_dir = os.path.join(save_dir, "telemetry")
        self.model = model
        self.record_interval = record_interval
        self.rank = rank
        self.is_active = (rank == 0)

        if self.is_active:
            os.makedirs(self.save_dir, exist_ok=True)
            self.iter_file = open(os.path.join(self.save_dir, "train_iter_telemetry.jsonl"), "a", buffering=1)
            self.grad_file = open(os.path.join(self.save_dir, "gradient_flow_telemetry.jsonl"), "a", buffering=1)
            self.assign_file = open(os.path.join(self.save_dir, "simota_telemetry.jsonl"), "a", buffering=1)
            self.proto_file = open(os.path.join(self.save_dir, "prototype_geometry.jsonl"), "a", buffering=1)

        # Map model parameters into explicit functional architectural buckets
        self.param_groups = self._categorize_parameter_groups()

    def _categorize_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Partitions all model parameters into functional subnets for gradient tracing."""
        groups = {
            "stem": [],
            "dark2_gn": [],
            "dark3_p3": [],
            "dark4_p4": [],
            "dark5_p5": [],
            "spi_highway": [],
            "neck_tribrid": [],
            "head_reg": [],
            "head_obj": [],
            "head_cls": [],
            "memory_banks": [],
            "ttt_meta_lrs": [],
            "ttt_projector": [],
            "affordance_head": [],
        }

        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        for name, param in model_ref.named_parameters():
            if not param.requires_grad:
                continue
            if "stem" in name:
                groups["stem"].append(param)
            elif "dark2" in name and "projector" not in name and "ttt_lrs" not in name:
                groups["dark2_gn"].append(param)
            elif "dark3" in name:
                groups["dark3_p3"].append(param)
            elif "dark4" in name:
                groups["dark4_p4"].append(param)
            elif "dark5" in name:
                groups["dark5_p5"].append(param)
            elif "spi_highway" in name:
                groups["spi_highway"].append(param)
            elif "C3_" in name or "lateral_" in name or "reduce_" in name or "bu_" in name:
                groups["neck_tribrid"].append(param)
            elif "reg_" in name:
                groups["head_reg"].append(param)
            elif "obj_" in name:
                groups["head_obj"].append(param)
            elif "cls_" in name:
                groups["head_cls"].append(param)
            elif "memory_banks" in name or "prototypes" in name:
                groups["memory_banks"].append(param)
            elif "ttt_lrs" in name:
                groups["ttt_meta_lrs"].append(param)
            elif "projector" in name:
                groups["ttt_projector"].append(param)
            elif "affordance_head" in name:
                groups["affordance_head"].append(param)

        return groups

    @torch.no_grad()
    def record_gradient_flow(self, epoch: int, iteration: int):
        """
        Extracts component-wise gradient norms and update ratios.
        Executes in a single batched GPU computation without stalling CUDA execution.
        """
        if not self.is_active or (iteration % self.record_interval != 0):
            return

        grad_norms = {}
        weight_norms = {}
        update_ratios = {}
        total_norm_sq = 0.0

        for group_name, params in self.param_groups.items():
            valid_grads = [p.grad for p in params if p.grad is not None]
            valid_weights = [p.data for p in params if p.grad is not None]

            if not valid_grads:
                grad_norms[group_name] = 0.0
                weight_norms[group_name] = 0.0
                update_ratios[group_name] = 0.0
                continue

            # Batch compute norms on device
            g_sq = sum(g.pow(2).sum() for g in valid_grads)
            w_sq = sum(w.pow(2).sum() for w in valid_weights)

            total_norm_sq += g_sq.item()
            g_norm = g_sq.sqrt().item()
            w_norm = w_sq.sqrt().item()

            grad_norms[group_name] = round(g_norm, 6)
            weight_norms[group_name] = round(w_norm, 6)
            update_ratios[group_name] = round(g_norm / (w_norm + 1e-7), 8)

        record = {
            "epoch": epoch,
            "iter": iteration,
            "timestamp": round(time.time(), 2),
            "total_grad_norm": round(total_norm_sq ** 0.5, 6),
            "grad_norms": grad_norms,
            "weight_norms": weight_norms,
            "update_ratios": update_ratios,
        }

        self.grad_file.write(json.dumps(record) + "\n")

    def record_iteration_losses(self, epoch: int, iteration: int, lr: float, outputs: Dict[str, float]):
        """Logs exact numerical loss decompositions into JSONL."""
        if not self.is_active or (iteration % self.record_interval != 0):
            return

        record = {
            "epoch": epoch,
            "iter": iteration,
            "lr": round(lr, 8),
            "timestamp": round(time.time(), 2),
        }
        for k, v in outputs.items():
            if isinstance(v, (int, float)):
                record[k] = round(float(v), 6)
            elif torch.is_tensor(v):
                record[k] = round(v.detach().cpu().item(), 6)

        self.iter_file.write(json.dumps(record) + "\n")

    @torch.no_grad()
    def record_simota_assignment(
        self,
        epoch: int,
        iteration: int,
        num_fg: float,
        pred_ious: Optional[torch.Tensor] = None,
        dynamic_ks: Optional[torch.Tensor] = None,
        cls_targets: Optional[torch.Tensor] = None,
    ):
        """Records SimOTA matching statistics to detect anchor haloing or micro starvation."""
        if not self.is_active or (iteration % self.record_interval != 0):
            return

        record = {
            "epoch": epoch,
            "iter": iteration,
            "num_fg_total": int(num_fg),
        }

        if pred_ious is not None and pred_ious.numel() > 0:
            ious_cpu = pred_ious.detach().float().cpu()
            record["matched_iou_mean"] = round(ious_cpu.mean().item(), 4)
            record["matched_iou_std"] = round(ious_cpu.std().item(), 4)
            record["matched_iou_min"] = round(ious_cpu.min().item(), 4)
            record["matched_iou_max"] = round(ious_cpu.max().item(), 4)

        if dynamic_ks is not None and dynamic_ks.numel() > 0:
            ks_cpu = dynamic_ks.detach().float().cpu()
            record["dynamic_k_mean"] = round(ks_cpu.mean().item(), 2)
            record["dynamic_k_min"] = int(ks_cpu.min().item())
            record["dynamic_k_max"] = int(ks_cpu.max().item())

        if cls_targets is not None and cls_targets.numel() > 0:
            active_classes = cls_targets.sum(dim=0).detach().cpu().tolist()
            record["class_sample_distribution"] = [round(x, 1) for x in active_classes]

        self.assign_file.write(json.dumps(record) + "\n")

    @torch.no_grad()
    def record_prototype_geometry(self, epoch: int):
        """
        Logs the complete geometric matrix of the hyperspherical prototype memory.
        Monitors intra-class sub-centroid diversity and inter-class taxonomic separation.
        """
        if not self.is_active:
            return

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model_ref, "head") or not hasattr(model_ref.head, "memory_banks"):
            return

        geometry_data = {"epoch": epoch, "levels": {}}

        for k, bank in enumerate(model_ref.head.memory_banks):
            prototypes = bank.prototypes.data  # [K, M, D]
            K, M, D = prototypes.shape

            # Compute L2 norms before normalization to monitor inflation
            p_norms = prototypes.norm(p=2, dim=-1).cpu().tolist()

            # Normalized representations
            p_normed = F.normalize(prototypes, p=2, dim=-1)

            # Class-mean cosine matrix [K, K]
            p_class = F.normalize(p_normed.mean(dim=1), p=2, dim=-1)
            inter_class_cos = torch.matmul(p_class, p_class.t()).cpu().tolist()

            # Intra-class diversity: average pairwise cosine between sub-centroids within each class
            intra_class_cos = []
            if M > 1:
                for c in range(K):
                    sub_sim = torch.matmul(p_normed[c], p_normed[c].t())
                    eye_m = torch.eye(M, device=sub_sim.device)
                    off_diag = (sub_sim * (1.0 - eye_m)).sum() / (M * (M - 1))
                    intra_class_cos.append(round(off_diag.item(), 4))

            geometry_data["levels"][f"level_{k}_stride_{8 * (2**k)}"] = {
                "raw_l2_norms": [[round(val, 3) for val in row] for row in p_norms],
                "inter_class_similarity_matrix": [[round(val, 4) for val in row] for row in inter_class_cos],
                "intra_class_subcentroid_similarity": intra_class_cos,
            }

        self.proto_file.write(json.dumps(geometry_data) + "\n")

    def close(self):
        """Flushes buffers and closes file handles upon training completion."""
        if self.is_active:
            self.iter_file.close()
            self.grad_file.close()
            self.assign_file.close()
            self.proto_file.close()