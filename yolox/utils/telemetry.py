#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TelemetryRecorder:
    """Low-interference telemetry for tdevNext vNext.2+.

    Schema v3 records five orthogonal streams:
      1. loss/runtime state (LR used vs next LR, AMP scale, optimizer-step status),
      2. pre/post-clip gradient flow by architectural group,
      3. exact group parameter updates at a sparse heavy interval + cheap sentinels,
      4. rich SimOTA assignment health by object scale, FPN level, class, IoU and Dynamic-K,
      5. prototype geometry and architectural gate/meta-LR state.

    The recorder is intentionally read-only with respect to model/optimizer state.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        save_dir: str,
        model: nn.Module,
        record_interval: int = 10,
        rank: int = 0,
        heavy_update_interval: Optional[int] = None,
    ):
        self.save_dir = os.path.join(save_dir, "telemetry")
        self.model = model
        self.record_interval = max(1, int(record_interval))
        self.heavy_update_interval = int(
            heavy_update_interval
            if heavy_update_interval is not None
            else max(100, self.record_interval * 10)
        )
        self.rank = rank
        self.is_active = rank == 0

        if self.is_active:
            os.makedirs(self.save_dir, exist_ok=True)
            self.iter_file = open(os.path.join(self.save_dir, "train_iter_telemetry.jsonl"), "a", buffering=1)
            self.grad_file = open(os.path.join(self.save_dir, "gradient_flow_telemetry.jsonl"), "a", buffering=1)
            self.assign_file = open(os.path.join(self.save_dir, "simota_telemetry.jsonl"), "a", buffering=1)
            self.proto_file = open(os.path.join(self.save_dir, "prototype_geometry.jsonl"), "a", buffering=1)
            self.update_file = open(os.path.join(self.save_dir, "parameter_update_telemetry.jsonl"), "a", buffering=1)
            self.arch_file = open(os.path.join(self.save_dir, "architecture_state_telemetry.jsonl"), "a", buffering=1)

        self.param_groups = self._categorize_parameter_groups()

    def should_record(self, iteration: int) -> bool:
        return self.is_active and iteration % self.record_interval == 0

    def should_record_heavy(self, iteration: int) -> bool:
        return self.is_active and iteration % self.heavy_update_interval == 0

    def _model_ref(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    @staticmethod
    def _jsonable(value: Any):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                item = value.item()
                if isinstance(item, float) and not math.isfinite(item):
                    return None
                return item
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): TelemetryRecorder._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TelemetryRecorder._jsonable(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _categorize_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """One-and-only-one bucket per trainable parameter."""
        keys = (
            "backbone_stem", "dark2", "dark3_p3", "dark4_p4", "dark5_p5",
            "spi_highway", "neck_csp", "neck_global_attention",
            "head_stem", "head_reg", "head_obj", "head_cls", "head_misc",
            "memory_banks", "memory_adapters", "ttt_meta_lrs", "ttt_projector",
            "affordance_head", "unassigned",
        )
        groups = {k: [] for k in keys}
        named_groups = {k: [] for k in keys}
        seen = set()
        for name, param in self._model_ref().named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone.backbone.stem."):
                group = "backbone_stem"
            elif "ttt_lrs." in name:
                group = "ttt_meta_lrs"
            elif "dark2.projector." in name:
                group = "ttt_projector"
            elif "dark2." in name:
                group = "dark2"
            elif "dark3." in name:
                group = "dark3_p3"
            elif "dark4." in name:
                group = "dark4_p4"
            elif "dark5." in name:
                group = "dark5_p5"
            elif "spi_highway." in name:
                group = "spi_highway"
            elif "global_mixer." in name:
                group = "neck_global_attention"
            elif any(s in name for s in ("C3_", "lateral_conv", "reduce_conv", "bu_conv")):
                group = "neck_csp"
            elif "head.memory_banks." in name:
                group = "memory_banks"
            elif any(s in name for s in (
                "latent_projectors.", "spatial_smoothers.",
                "memory_proj_back.", "memory_norms.",
            )):
                group = "memory_adapters"
            elif "head.stems." in name:
                group = "head_stem"
            elif "head.reg_" in name or "head.gamma_reg" in name:
                group = "head_reg"
            elif "head.obj_" in name:
                group = "head_obj"
            elif "head.cls_" in name:
                group = "head_cls"
            elif "affordance_head." in name:
                group = "affordance_head"
            elif "head." in name:
                group = "head_misc"
            else:
                group = "unassigned"

            assert id(param) not in seen, f"Duplicate telemetry parameter: {name}"
            groups[group].append(param)
            named_groups[group].append((name, param))
            seen.add(id(param))

        assert len(seen) == sum(len(v) for v in groups.values())
        self.named_param_groups = named_groups
        return groups

    @torch.no_grad()
    def capture_gradient_state(self, iteration: int):
        """Capture exact group gradient state at the current point in the step.

        Called once immediately after GradScaler.unscale_ (pre-clip) and once after
        clip_grad_norm_ (post-clip). Returns None outside telemetry intervals.
        """
        if not self.should_record(iteration):
            return None

        result = {"groups": {}, "total_sq": 0.0, "any_nonfinite": False}
        for group_name, named_params in self.named_param_groups.items():
            params = [p for _, p in named_params]
            weights = [p.detach() for p in params]
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if grads:
                nonzero_flags = torch.stack([torch.count_nonzero(g) > 0 for g in grads])
                nonzero_tensors = int(nonzero_flags.sum().item())
            else:
                nonzero_tensors = 0
            zero_tensors = len(grads) - nonzero_tensors

            if not weights:
                result["groups"][group_name] = {
                    "parameter_tensors": 0,
                    "with_grad": 0,
                    "nonzero_grad_tensors": 0,
                    "zero_grad_tensors": 0,
                    "grad_norm": 0.0,
                    "weight_norm": 0.0,
                    "gradient_to_weight_ratio": 0.0,
                    "grad_abs_max": 0.0,
                    "status": "empty",
                }
                continue

            w_sq = sum(w.float().double().square().sum() for w in weights)
            w_norm = float(w_sq.sqrt().item())
            if not grads:
                result["groups"][group_name] = {
                    "parameter_tensors": len(params),
                    "with_grad": 0,
                    "nonzero_grad_tensors": 0,
                    "zero_grad_tensors": 0,
                    "grad_norm": 0.0,
                    "weight_norm": w_norm,
                    "gradient_to_weight_ratio": 0.0,
                    "grad_abs_max": 0.0,
                    "status": "no_gradient",
                }
                continue

            finite_flags = torch.stack([torch.isfinite(g).all() for g in grads])
            finite = bool(finite_flags.all().item())
            if not finite:
                result["any_nonfinite"] = True
                result["groups"][group_name] = {
                    "parameter_tensors": len(params),
                    "with_grad": len(grads),
                    "nonzero_grad_tensors": nonzero_tensors,
                    "zero_grad_tensors": zero_tensors,
                    "grad_norm": None,
                    "weight_norm": w_norm,
                    "gradient_to_weight_ratio": None,
                    "grad_abs_max": None,
                    "status": "nonfinite_gradient",
                }
                continue

            g_sq = sum(g.float().double().square().sum() for g in grads)
            g_norm = float(g_sq.sqrt().item())
            g_abs_max = float(torch.stack([g.float().abs().max() for g in grads]).max().item())
            result["total_sq"] += g_norm * g_norm
            result["groups"][group_name] = {
                "parameter_tensors": len(params),
                "with_grad": len(grads),
                "nonzero_grad_tensors": nonzero_tensors,
                "zero_grad_tensors": zero_tensors,
                "grad_norm": g_norm,
                "weight_norm": w_norm,
                "gradient_to_weight_ratio": g_norm / (w_norm + 1e-12),
                "grad_abs_max": g_abs_max,
                "status": "finite",
            }

        result["total_grad_norm"] = None if result["any_nonfinite"] else result["total_sq"] ** 0.5
        if result["total_grad_norm"] and result["total_grad_norm"] > 0:
            denom = result["total_grad_norm"] ** 2
            for group in result["groups"].values():
                g = group.get("grad_norm")
                group["gradient_energy_fraction"] = None if g is None else (g * g) / denom
        else:
            for group in result["groups"].values():
                group["gradient_energy_fraction"] = 0.0 if group.get("grad_norm") is not None else None
        return result

    @torch.no_grad()
    def record_gradient_flow(
        self,
        epoch: int,
        iteration: int,
        pre_clip,
        post_clip,
        clip_threshold: float,
        clip_norm_returned: Optional[float],
    ):
        if pre_clip is None or not self.is_active:
            return

        pre_total = pre_clip.get("total_grad_norm")
        post_total = post_clip.get("total_grad_norm") if post_clip is not None else None
        if pre_total is None:
            clip_coefficient = None
        else:
            clip_coefficient = min(1.0, float(clip_threshold) / (float(pre_total) + 1e-12))

        groups = {}
        for name, pre in pre_clip["groups"].items():
            post = post_clip["groups"].get(name, {}) if post_clip is not None else {}
            pre_norm = pre.get("grad_norm")
            post_norm = post.get("grad_norm")
            groups[name] = {
                **pre,
                "post_clip_grad_norm": post_norm,
                "post_clip_gradient_to_weight_ratio": post.get("gradient_to_weight_ratio"),
                "post_to_pre_norm_ratio": (
                    None
                    if pre_norm in (None, 0.0) or post_norm is None
                    else post_norm / (pre_norm + 1e-12)
                ),
            }

        record = {
            "telemetry_schema": self.SCHEMA_VERSION,
            "epoch": epoch,
            "iter": iteration,
            "timestamp": round(time.time(), 2),
            "clip_threshold": float(clip_threshold),
            "clip_norm_returned": clip_norm_returned,
            "pre_clip_total_grad_norm": pre_total,
            "post_clip_total_grad_norm": post_total,
            "clip_coefficient": clip_coefficient,
            "was_clipped": bool(clip_coefficient is not None and clip_coefficient < 0.999999),
            "any_nonfinite_gradient": bool(pre_clip.get("any_nonfinite", False)),
            "groups": groups,
        }
        self.grad_file.write(json.dumps(record, allow_nan=False) + "\n")

    def _is_sentinel(self, name: str) -> bool:
        return (
            name.endswith(".dense_gate")
            or name.endswith(".ema.gamma")
            or name.endswith(".gate")
            or name in ("head.mem_gate", "head.gamma_reg")
            or "dark2.residual_scale" in name
            or "dark2.residual_bias" in name
            or name.endswith("global_mixer.3.qkv.weight")
            or (name.startswith("head.memory_banks.") and name.endswith("prototypes"))
        )

    @torch.no_grad()
    def snapshot_parameter_updates(self, iteration: int):
        """Snapshot cheap sentinels every record interval and full groups sparsely.

        Full-group snapshots are exact but clone ~all trainable parameters, so they are
        intentionally limited to heavy_update_interval (default >=100 iterations).
        """
        if not self.should_record(iteration):
            return None

        sentinels = {}
        for name, p in self._model_ref().named_parameters():
            if p.requires_grad and self._is_sentinel(name):
                sentinels[name] = (p, p.detach().clone())

        full_groups = None
        if self.should_record_heavy(iteration):
            full_groups = {}
            for group_name, named_params in self.named_param_groups.items():
                full_groups[group_name] = [
                    (name, p, p.detach().clone()) for name, p in named_params
                ]

        return {
            "sentinels": sentinels,
            "full_groups": full_groups,
            "heavy": full_groups is not None,
        }

    @torch.no_grad()
    def record_parameter_updates(
        self,
        epoch: int,
        iteration: int,
        snapshot,
        did_step: bool,
        lr_used: Optional[float] = None,
        clip_coefficient: Optional[float] = None,
    ):
        if snapshot is None or not self.is_active:
            return

        record = {
            "telemetry_schema": self.SCHEMA_VERSION,
            "epoch": epoch,
            "iter": iteration,
            "optimizer_step": bool(did_step),
            "lr_used": lr_used,
            "clip_coefficient": clip_coefficient,
            "mode": "full_group_exact" if snapshot["heavy"] else "sentinel_only",
            "sentinels": {},
            "groups": {},
        }

        for name, (p, before) in snapshot["sentinels"].items():
            after = p.detach().float()
            before_f = before.float()
            diff = after - before_f
            delta = float(diff.norm().item())
            base = float(before_f.norm().item())
            record["sentinels"][name] = {
                "actual_delta_l2": delta,
                "actual_relative_delta": delta / (base + 1e-12),
                "before_l2": base,
                "after_l2": float(after.norm().item()),
                "value_mean": float(after.mean().item()),
                "value_abs_max": float(after.abs().max().item()),
            }

        full_groups = snapshot.get("full_groups")
        if full_groups is not None:
            for group_name, items in full_groups.items():
                numel = sum(p.numel() for _name, p, _before in items)
                if not items:
                    record["groups"][group_name] = {
                        "parameter_elements": 0, "parameter_tensors": 0,
                        "changed_tensors": 0, "actual_delta_l2": 0.0,
                        "actual_relative_delta": 0.0, "update_rms_per_element": 0.0,
                        "max_abs_delta": 0.0, "before_l2": 0.0, "after_l2": 0.0,
                    }
                    continue
                diffs = []
                before_terms = []
                after_terms = []
                max_terms = []
                changed_terms = []
                for _name, p, before in items:
                    after = p.detach().float()
                    before_f = before.float()
                    diff = after - before_f
                    diffs.append(diff.double().square().sum())
                    before_terms.append(before_f.double().square().sum())
                    after_terms.append(after.double().square().sum())
                    max_terms.append(diff.abs().max())
                    changed_terms.append(torch.count_nonzero(diff) > 0)
                delta_sq = float(torch.stack(diffs).sum().item())
                before_sq = float(torch.stack(before_terms).sum().item())
                after_sq = float(torch.stack(after_terms).sum().item())
                max_abs_delta = float(torch.stack(max_terms).max().item())
                changed_tensors = int(torch.stack(changed_terms).sum().item())
                delta_l2 = delta_sq ** 0.5
                before_l2 = before_sq ** 0.5
                record["groups"][group_name] = {
                    "parameter_elements": numel,
                    "parameter_tensors": len(items),
                    "changed_tensors": changed_tensors,
                    "actual_delta_l2": delta_l2,
                    "actual_relative_delta": delta_l2 / (before_l2 + 1e-12),
                    "update_rms_per_element": (delta_sq / max(1, numel)) ** 0.5,
                    "max_abs_delta": max_abs_delta,
                    "before_l2": before_l2,
                    "after_l2": after_sq ** 0.5,
                }

        self.update_file.write(json.dumps(record, allow_nan=False) + "\n")

    def record_iteration_losses(
        self,
        epoch: int,
        iteration: int,
        lr_next: float,
        outputs: Dict[str, float],
        runtime: Optional[Dict[str, Any]] = None,
    ):
        if not self.should_record(iteration):
            return

        record = {
            "telemetry_schema": self.SCHEMA_VERSION,
            "epoch": epoch,
            "iter": iteration,
            "lr_next": float(lr_next),
            "timestamp": round(time.time(), 2),
        }
        if runtime:
            record.update(self._jsonable(runtime))
        for k, v in outputs.items():
            if isinstance(v, (int, float)):
                value = float(v)
                record[k] = value if math.isfinite(value) else None
            elif torch.is_tensor(v) and v.numel() == 1:
                value = float(v.detach().float().cpu().item())
                record[k] = value if math.isfinite(value) else None

        self.iter_file.write(json.dumps(record, allow_nan=False) + "\n")

    @torch.no_grad()
    def record_simota_assignment(self, epoch: int, iteration: int, stats: Dict[str, Any]):
        """Write rich assignment telemetry from TDE_Head.last_assignment_stats."""
        if not self.should_record(iteration) or not stats:
            return

        raw = self._jsonable(stats)
        nfg = int(raw.get("num_fg_total", 0) or 0)
        ngt = int(raw.get("num_gt_total", 0) or 0)

        record = {
            "telemetry_schema": self.SCHEMA_VERSION,
            "epoch": epoch,
            "iter": iteration,
            "timestamp": round(time.time(), 2),
            "assignment_mode": raw.get("assignment_mode"),
            "input_hw": raw.get("input_hw"),
            "num_fg_total": nfg,
            "num_gt_total": ngt,
            "fg_per_gt": nfg / max(1, ngt),
            "size_bin_basis": raw.get("size_bin_basis"),
            "size_thresholds_px": raw.get("size_thresholds_px"),
        }

        # Overall matched-IoU moments + range sanity.
        iou_count = int(raw.get("matched_iou_count", 0) or 0)
        iou_sum = float(raw.get("matched_iou_sum", 0.0) or 0.0)
        iou_sq = float(raw.get("matched_iou_sq_sum", 0.0) or 0.0)
        iou_mean = iou_sum / max(1, iou_count)
        iou_var = max(0.0, iou_sq / max(1, iou_count) - iou_mean * iou_mean)
        record["matched_iou"] = {
            "count": iou_count,
            "mean": iou_mean if iou_count else None,
            "std": math.sqrt(iou_var) if iou_count else None,
            "min": raw.get("matched_iou_min") if iou_count else None,
            "max": raw.get("matched_iou_max") if iou_count else None,
            "histogram_0_025_05_075_1": raw.get("matched_iou_hist"),
            "invalid_or_out_of_range": int(raw.get("invalid_matched_iou_count", 0) or 0),
        }

        # Micro / medium / macro assignment health.
        names = raw.get("size_bin_names") or ["micro", "medium", "macro"]
        gt = raw.get("size_gt_counts") or [0] * len(names)
        pos = raw.get("size_positive_counts") or [0] * len(names)
        covered = raw.get("size_covered_gt_counts") or [0] * len(names)
        zero = raw.get("size_zero_positive_gt_counts") or [0] * len(names)
        iou_sums = raw.get("size_iou_sum") or [0.0] * len(names)
        iou_counts = raw.get("size_iou_count") or [0] * len(names)
        record["by_size"] = {}
        for i, name in enumerate(names):
            g = int(gt[i])
            p = int(pos[i])
            c = int(covered[i])
            z = int(zero[i])
            ic = int(iou_counts[i])
            record["by_size"][name] = {
                "gt": g,
                "positive_anchors": p,
                "fg_per_gt": p / max(1, g),
                "covered_gt": c,
                "zero_positive_gt": z,
                "coverage_rate": c / max(1, g),
                "mean_matched_iou": float(iou_sums[i]) / max(1, ic) if ic else None,
            }

        # Positive routing and quality per FPN level.
        strides = raw.get("fpn_strides") or []
        fpn_pos = raw.get("fpn_positive_counts") or []
        fpn_iou_sum = raw.get("fpn_iou_sum") or []
        fpn_iou_count = raw.get("fpn_iou_count") or []
        record["by_fpn"] = {}
        for i, stride in enumerate(strides):
            count = int(fpn_iou_count[i]) if i < len(fpn_iou_count) else 0
            record["by_fpn"][f"stride_{int(stride)}"] = {
                "positive_anchors": int(fpn_pos[i]),
                "positive_fraction": int(fpn_pos[i]) / max(1, nfg),
                "mean_matched_iou": float(fpn_iou_sum[i]) / max(1, count) if count else None,
            }

        dyn_count = int(raw.get("dynamic_k_count", 0) or 0)
        dyn_sum = float(raw.get("dynamic_k_sum", 0.0) or 0.0)
        requested = int(raw.get("pre_conflict_positive_total", 0) or 0)
        conflicts = int(raw.get("conflict_anchor_total", 0) or 0)
        fallback = int(raw.get("geometry_fallback_gt_total", 0) or 0)
        record["dynamic_k"] = {
            "count": dyn_count,
            "mean": dyn_sum / max(1, dyn_count) if dyn_count else None,
            "min": raw.get("dynamic_k_min") if dyn_count else None,
            "max": raw.get("dynamic_k_max") if dyn_count else None,
            "histogram_k0_to_k10": raw.get("dynamic_k_hist"),
            "requested_positives_pre_conflict": requested,
            "final_positives": nfg,
            "retention_after_conflict": nfg / max(1, requested) if requested else None,
            "conflict_anchors": conflicts,
        }
        record["geometry"] = {
            "fallback_gt": fallback,
            "fallback_rate": fallback / max(1, ngt),
            "candidate_anchor_total_across_images": int(raw.get("candidate_anchor_total", 0) or 0),
            "images_with_gt": int(raw.get("images_with_gt", 0) or 0),
        }
        record["class_gt_counts"] = raw.get("class_gt_counts")
        record["class_positive_counts"] = raw.get("class_positive_counts")

        self.assign_file.write(json.dumps(record, allow_nan=False) + "\n")

    @torch.no_grad()
    def record_architecture_state(self, epoch: int, iteration: int):
        if not self.should_record(iteration):
            return

        model_ref = self._model_ref()
        record = {
            "telemetry_schema": self.SCHEMA_VERSION,
            "epoch": epoch,
            "iter": iteration,
            "timestamp": round(time.time(), 2),
            "controls": {},
        }
        for name, p in model_ref.named_parameters():
            interesting = (
                name.endswith(".dense_gate")
                or name.endswith(".ema.gamma")
                or name.endswith(".gate")
                or name in ("head.mem_gate", "head.gamma_reg")
                or "dark2.residual_scale" in name
                or "dark2.residual_bias" in name
            )
            if interesting:
                t = p.detach().float()
                record["controls"][name] = {
                    "mean": float(t.mean().item()),
                    "abs_mean": float(t.abs().mean().item()),
                    "abs_max": float(t.abs().max().item()),
                }

        # Convert raw meta-LR logits into actual current TTT inner learning rates.
        dark2 = None
        try:
            dark2 = model_ref.backbone.backbone.dark2
        except AttributeError:
            pass
        if dark2 is not None and hasattr(dark2, "ttt_lrs"):
            epoch_fraction = min(1.0, max(0.0, dark2.current_epoch / max(1, dark2.max_epochs)))
            lr_ceiling = dark2.min_meta_lr + (dark2.base_max_lr - dark2.min_meta_lr) * (
                0.5 * (1.0 + math.cos(math.pi * epoch_fraction))
            )
            meta = {}
            for name, raw_param in dark2.ttt_lrs.items():
                scale = float(torch.sigmoid(raw_param.detach().float()).item())
                actual = dark2.min_meta_lr + (lr_ceiling - dark2.min_meta_lr) * scale
                meta[name] = {
                    "sigmoid_fraction": scale,
                    "actual_inner_lr": actual,
                }
            record["ttt_meta_lr"] = {
                "lr_ceiling": lr_ceiling,
                "parameters": meta,
            }

        self.arch_file.write(json.dumps(record, allow_nan=False) + "\n")

    @torch.no_grad()
    def record_prototype_geometry(self, epoch: int):
        """Log full hyperspherical prototype geometry once per epoch."""
        if not self.is_active:
            return

        model_ref = self._model_ref()
        if not hasattr(model_ref, "head") or not hasattr(model_ref.head, "memory_banks"):
            return

        geometry_data = {
            "telemetry_schema": self.SCHEMA_VERSION,
            "epoch": epoch,
            "levels": {},
        }
        for k, bank in enumerate(model_ref.head.memory_banks):
            prototypes = bank.prototypes.data
            K, M, _D = prototypes.shape
            p_norms = prototypes.norm(p=2, dim=-1).cpu().tolist()
            p_normed = F.normalize(prototypes, p=2, dim=-1)
            p_class = F.normalize(p_normed.mean(dim=1), p=2, dim=-1)
            inter_class_cos = torch.matmul(p_class, p_class.t()).cpu().tolist()

            intra_class_cos = []
            if M > 1:
                for c in range(K):
                    sub_sim = torch.matmul(p_normed[c], p_normed[c].t())
                    eye_m = torch.eye(M, device=sub_sim.device)
                    off_diag = (sub_sim * (1.0 - eye_m)).sum() / (M * (M - 1))
                    intra_class_cos.append(round(off_diag.item(), 4))

            geometry_data["levels"][f"level_{k}_stride_{8 * (2**k)}"] = {
                "raw_l2_norms": [[round(val, 5) for val in row] for row in p_norms],
                "inter_class_similarity_matrix": [
                    [round(val, 5) for val in row] for row in inter_class_cos
                ],
                "intra_class_subcentroid_similarity": intra_class_cos,
            }

        self.proto_file.write(json.dumps(geometry_data, allow_nan=False) + "\n")

    def close(self):
        if self.is_active:
            for f in (
                self.iter_file,
                self.grad_file,
                self.assign_file,
                self.proto_file,
                self.update_file,
                self.arch_file,
            ):
                f.close()
