#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Pilot G7 — Decision-Frontier, Failure-Mode Decomposition, and Classification Oracle Audit.

Evaluates frozen native predictions vs. PG-SMA corrections vs. an ideal Classification Oracle.
Decomposes every ground-truth missed detection into:
  1. Localization-limited (no box reaches IoU >= 0.50)
  2. Objectness-limited (box IoU >= 0.50 exists, but obj < tau_conf prevents score >= tau_conf)
  3. Classification-limited (box and obj viable, but class score/rank failed)
  4. NMS-limited (candidate passed threshold before NMS, but was suppressed)
  5. True Positive (already successfully detected)

Run from repository root:
  python -m tools.audit_pgsma_decision_frontier `
      --checkpoint audits/pilot_b_simota/pilot_last.pth `
      --pgsma-checkpoint audits/pilot_c_simota_pgsma_fixed/pilot_last.pth `
      --eval-images 128 --conf 0.01 --nms 0.60 --device cuda `
      --json audits/pilot_g7_decision_frontier.json
"""

import argparse
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset

from tools._pgsma_pilot_core import atomic_json, file_digest, group_scale, greedy_recall_pairs
from tools.pgsma_bounded_pilot import safe_checkpoint_state
from yolox.exp import get_exp
from yolox.utils import postprocess

SIZES = ("lt4", "4to8", "8to16", "16to32", "ge32")


def safe_load_compatible(model, state_dict):
    """Load matching keys with identical shapes, ignoring architecture version discrepancies."""
    model_state = model.state_dict()
    compatible = {}
    mismatched = []
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                compatible[k] = v
            else:
                mismatched.append((k, list(v.shape), list(model_state[k].shape)))
    model.load_state_dict(compatible, strict=False)
    return compatible, mismatched


def bbox_iou_tensor(p_xyxy, g_xyxy):
    if not len(g_xyxy) or not len(p_xyxy):
        return p_xyxy.new_zeros((len(p_xyxy), len(g_xyxy)))
    lu = torch.maximum(p_xyxy[:, None, :2], g_xyxy[None, :, :2])
    rd = torch.minimum(p_xyxy[:, None, 2:], g_xyxy[None, :, 2:])
    inter = (rd - lu).clamp_min(0).prod(-1)
    a = (p_xyxy[:, 2:] - p_xyxy[:, :2]).clamp_min(0).prod(-1)
    b = (g_xyxy[:, 2:] - g_xyxy[:, :2]).clamp_min(0).prod(-1)
    return inter / (a[:, None] + b[None, :] - inter).clamp_min(1e-9)


def run_g7_audit(args):
    from exps.default.tde_yolox_s_pgsma import Exp
    exp = Exp()
    exp.data_num_workers = 0
    exp.ema = False
    device = torch.device(args.device)

    # 1. Load Native Model
    model = exp.get_model()
    _, native_weights = safe_checkpoint_state(args.checkpoint)
    loaded_native, mismatched_native = safe_load_compatible(model, native_weights)
    if mismatched_native:
        print(f"[*] Safely bypassed {len(mismatched_native)} mismatched keys in native checkpoint (e.g. legacy gate):")
        for k, src_s, tgt_s in mismatched_native[:4]:
            print(f"    - {k}: src {src_s} -> tgt {tgt_s}")

    model.to(device).eval()

    # 2. Extract PG-SMA weights if provided
    pgsma_weights = None
    mismatched_pgsma = []
    if args.pgsma_checkpoint:
        _, pgsma_raw = safe_checkpoint_state(args.pgsma_checkpoint)
        pgsma_weights = pgsma_raw

    # 3. Validation Loader Setup
    source_loader = exp.get_eval_loader(batch_size=1, is_distributed=False, testdev=False)
    dataset = source_loader.dataset
    cats = list(dataset.class_ids)

    # Deterministic subset selection
    rng = np.random.RandomState(args.seed)
    nval = len(dataset)
    subset_indices = sorted(rng.choice(nval, size=min(args.eval_images, nval), replace=False).tolist())

    subset_loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=source_loader.collate_fn,
    )

    failure_modes = {
        s: {
            "total_gt": 0,
            "tp_native": 0,
            "localization_limited": 0,
            "objectness_limited": 0,
            "classification_limited": 0,
            "nms_limited": 0,
        }
        for s in SIZES
    }

    coco_detections = {
        "native": [],
        "pgsma": [],
        "oracle_cls": [],
    }

    print(f"[*] Starting Pilot G7 Decision-Frontier Audit on {len(subset_indices)} images...", flush=True)

    selected_image_ids = []
    with torch.inference_mode():
        for step, batch in enumerate(subset_loader):
            imgs, _, info, ids = batch[:4]
            image_id = int(ids[0].item() if hasattr(ids[0], "item") else ids[0])
            selected_image_ids.append(image_id)
            ...
            imgs_t = imgs.to(device).float()
            image_id = int(ids[0].item() if hasattr(ids[0], "item") else ids[0])
            img_h = float(info[0][0])
            img_w = float(info[1][0])
            scale = min(float(exp.test_size[0]) / img_h, float(exp.test_size[1]) / img_w)

            # Ground truth extraction
            ann_ids = dataset.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
            anns = dataset.coco.loadAnns(ann_ids)
            gt_boxes_xyxy = []
            gt_classes = []
            gt_sizes = []
            for a in anns:
                cat_id = int(a["category_id"])
                if cat_id not in cats:
                    continue
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                gt_boxes_xyxy.append([x, y, x + w, y + h])
                gt_classes.append(cats.index(cat_id))
                gt_sizes.append(group_scale(min(w, h) * scale))

            gt_boxes_t = torch.tensor(gt_boxes_xyxy, device=device, dtype=torch.float32) * scale

            # --- Native Forward ---
            model.set_inference_mode("native")
            raw_native = model(imgs_t)  # [1, N, 5 + K] -> [cx, cy, w, h, obj_prob, cls_prob]

            # Parse decoded anchors
            pred_boxes_xyxy = raw_native[0, :, :4].clone()
            cx, cy = pred_boxes_xyxy[:, 0], pred_boxes_xyxy[:, 1]
            bw, bh = pred_boxes_xyxy[:, 2], pred_boxes_xyxy[:, 3]
            pred_boxes_xyxy[:, 0] = cx - bw / 2.0
            pred_boxes_xyxy[:, 1] = cy - bh / 2.0
            pred_boxes_xyxy[:, 2] = cx + bw / 2.0
            pred_boxes_xyxy[:, 3] = cy + bh / 2.0

            obj_prob = raw_native[0, :, 4]       # [N]
            cls_prob_nat = raw_native[0, :, 5:]  # [N, K]

            # Pairwise Anchor-to-GT IoUs
            ious = bbox_iou_tensor(pred_boxes_xyxy, gt_boxes_t)  # [N, G]

            # --- PG-SMA Forward (if compatible weights exist) ---
            raw_pgsma = None
            if pgsma_weights is not None:
                safe_load_compatible(model, pgsma_weights)
                model.set_inference_mode("pgsma")
                raw_pgsma = model(imgs_t)
                safe_load_compatible(model, native_weights)
                model.set_inference_mode("native")

            # --- G7-C: Classification Oracle ---
            # Max possible score = obj * 1.0 on matching class, 0.0 on competitors
            raw_oracle = raw_native.clone()
            oracle_cls_prob = raw_oracle[0, :, 5:]
            if len(gt_classes) > 0 and ious.numel() > 0:
                max_iou_per_anchor, matched_gt_for_anchor = ious.max(dim=1)
                valid_match_mask = max_iou_per_anchor >= 0.50
                if valid_match_mask.any():
                    matched_gt_classes = torch.tensor(gt_classes, device=device)[matched_gt_for_anchor[valid_match_mask]]
                    oracle_cls_prob[valid_match_mask] = 0.0
                    oracle_cls_prob[valid_match_mask, matched_gt_classes] = 1.0

            # --- G7-A: GT Failure Mode Decomposition ---
            det_native = postprocess(raw_native.clone(), len(cats), args.conf, args.nms, class_agnostic=False)[0]

            for g_idx, (gt_box, gt_cls, gt_sz) in enumerate(zip(gt_boxes_xyxy, gt_classes, gt_sizes)):
                failure_modes[gt_sz]["total_gt"] += 1
                if ious.shape[0] == 0 or ious.shape[1] == 0:
                    failure_modes[gt_sz]["localization_limited"] += 1
                    continue

                gt_ious = ious[:, g_idx]
                max_iou = float(gt_ious.max().item())

                # 1. Localization-limited
                if max_iou < 0.50:
                    failure_modes[gt_sz]["localization_limited"] += 1
                    continue

                candidates_mask = gt_ious >= 0.50
                cand_obj = obj_prob[candidates_mask]
                max_cand_obj = float(cand_obj.max().item())

                # 2. Objectness-limited (max possible score = obj * 1.0 < tau)
                if max_cand_obj < args.conf:
                    failure_modes[gt_sz]["objectness_limited"] += 1
                    continue

                cand_cls_score = cls_prob_nat[candidates_mask, gt_cls]
                composite_scores = cand_obj * cand_cls_score
                max_comp_score = float(composite_scores.max().item())

                # 3. Classification-limited (viable box + viable obj, but cls score dropped below tau)
                if max_comp_score < args.conf:
                    failure_modes[gt_sz]["classification_limited"] += 1
                    continue

                # 4 & 5. Post-NMS Survival (TP vs NMS-Suppression)
                detected_tp = False
                if det_native is not None and len(det_native) > 0:
                    boxes_det = det_native[:, :4] / scale
                    classes_det = det_native[:, 6].long()
                    det_ious = bbox_iou_tensor(boxes_det, torch.tensor([gt_box], device=device))[:, 0]
                    matched_det = (det_ious >= 0.50) & (classes_det == gt_cls)
                    if matched_det.any():
                        detected_tp = True

                if detected_tp:
                    failure_modes[gt_sz]["tp_native"] += 1
                else:
                    failure_modes[gt_sz]["nms_limited"] += 1

            # Accumulate Detections for COCO Metrics
            for mode_name, tensor_data in [("native", raw_native), ("pgsma", raw_pgsma), ("oracle_cls", raw_oracle)]:
                if tensor_data is None:
                    continue
                dets = postprocess(tensor_data.clone(), len(cats), args.conf, args.nms, class_agnostic=False)[0]
                if dets is not None and len(dets) > 0:
                    for row in dets.detach().cpu().tolist():
                        x1, y1, x2, y2, obj, cls_c, c_idx = row[:7]
                        c_idx = int(c_idx)
                        x1 = max(0.0, min(img_w, x1 / scale))
                        x2 = max(0.0, min(img_w, x2 / scale))
                        y1 = max(0.0, min(img_h, y1 / scale))
                        y2 = max(0.0, min(img_h, y2 / scale))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        coco_detections[mode_name].append({
                            "image_id": image_id,
                            "category_id": cats[c_idx],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(obj * cls_c),
                        })

    # --- Compute COCO Metrics ---
    coco_metrics = {}
    for mode_name in ["native", "pgsma", "oracle_cls"]:
        dets = coco_detections[mode_name]
        if not dets:
            coco_metrics[mode_name] = None
            continue
        with io.StringIO():
            coco_dt = dataset.coco.loadRes(dets)
            evaluator = COCOeval(dataset.coco, coco_dt, "bbox")
            # Use real COCO image IDs instead of subset_indices
            evaluator.params.imgIds = selected_image_ids
            evaluator.params.catIds = cats
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            s = evaluator.stats
            coco_metrics[mode_name] = {
                "AP50_95": float(s[0]),
                "AP50": float(s[1]),
                "AP75": float(s[2]),
                "AR100": float(s[8]),
                "AP_small": float(s[3]),
                "AP_medium": float(s[4]),
                "AP_large": float(s[5]),
            }

    report = {
        "status": "CHECKS_PASSED",
        "eval_images": len(subset_indices),
        "failure_mode_decomposition": failure_modes,
        "coco_metrics": coco_metrics,
        "theoretical_classification_headroom": {
            "delta_AP50_95": (
                coco_metrics["oracle_cls"]["AP50_95"] - coco_metrics["native"]["AP50_95"]
                if coco_metrics.get("oracle_cls") and coco_metrics.get("native") else None
            ),
            "delta_AR100": (
                coco_metrics["oracle_cls"]["AR100"] - coco_metrics["native"]["AR100"]
                if coco_metrics.get("oracle_cls") and coco_metrics.get("native") else None
            ),
            "delta_AP_small": (
                coco_metrics["oracle_cls"]["AP_small"] - coco_metrics["native"]["AP_small"]
                if coco_metrics.get("oracle_cls") and coco_metrics.get("native") else None
            ),
        },
    }

    atomic_json(args.json, report)

    print("\n" + "=" * 90)
    print("PILOT G7: DECISION-FRONTIER & FAILURE MODE DECOMPOSITION SUMMARY")
    print("=" * 90)
    print(f"{'Scale':<10} | {'Total GT':<9} | {'Native TP':<10} | {'Loc-Limited':<12} | {'Obj-Limited':<12} | {'Cls-Limited':<12} | {'NMS-Limited':<12}")
    print("-" * 90)
    for s in SIZES:
        fm = failure_modes[s]
        print(f"{s:<10} | {fm['total_gt']:<9} | {fm['tp_native']:<10} | {fm['localization_limited']:<12} | {fm['objectness_limited']:<12} | {fm['classification_limited']:<12} | {fm['nms_limited']:<12}")
    print("-" * 90)
    print("\nCOCO Metric Ceiling Comparison:")
    for m in ["native", "pgsma", "oracle_cls"]:
        if coco_metrics.get(m):
            print(f"  {m:<12}: AP50:95 = {coco_metrics[m]['AP50_95']:.4f} | AP50 = {coco_metrics[m]['AP50']:.4f} | AR100 = {coco_metrics[m]['AR100']:.4f} | AP_small = {coco_metrics[m]['AP_small']:.4f}")
    if report["theoretical_classification_headroom"]["delta_AP50_95"] is not None:
        print(f"\nTheoretical Max Classification Headroom: ΔAP50:95 = {report['theoretical_classification_headroom']['delta_AP50_95']:+.4f} | ΔAP_small = {report['theoretical_classification_headroom']['delta_AP_small']:+.4f}")
    print("=" * 90)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained native checkpoint")
    parser.add_argument("--pgsma-checkpoint", default=None, help="Trained PG-SMA checkpoint to compare (optional)")
    parser.add_argument("--eval-images", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--nms", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run_g7_audit(args)