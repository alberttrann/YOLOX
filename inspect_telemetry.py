#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
import math
import os
import sys

TELEMETRY_DIR = os.path.join("YOLOX_outputs", "tde_yolox_s", "telemetry")


def print_header(title):
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)


def analyze_weights_and_gradients():
    grad_file = os.path.join(TELEMETRY_DIR, "gradient_flow_telemetry.jsonl")
    if not os.path.exists(grad_file):
        print(f"[!] File not found: {grad_file}")
        return

    print_header("1. EMPIRICAL CHECK: DID WEIGHTS ACTUALLY UPDATE? (WEIGHT NORMS)")

    first_record = None
    last_record = None
    all_records = []

    with open(grad_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            all_records.append(data)

    if not all_records:
        print("[!] No records found in gradient_flow_telemetry.jsonl")
        return

    first_record = all_records[0]
    mid_record = all_records[len(all_records) // 2]
    last_record = all_records[-1]

    print(f"Total Logged Checkpoints: {len(all_records)}")
    print(f"First Record: Epoch {first_record['epoch']}, Iter {first_record['iter']}")
    print(f"Mid Record:   Epoch {mid_record['epoch']}, Iter {mid_record['iter']}")
    print(f"Last Record:  Epoch {last_record['epoch']}, Iter {last_record['iter']}\n")

    print(f"{'Subnet / Layer Group':<20} | {'Start ||W||':<12} | {'Mid ||W||':<12} | {'End ||W||':<12} | {'Net Change %':<12}")
    print("-" * 75)

    w_start = first_record.get("weight_norms", {})
    w_mid = mid_record.get("weight_norms", {})
    w_end = last_record.get("weight_norms", {})

    frozen_subnets = []
    active_subnets = []

    for k in w_start.keys():
        s = w_start.get(k, 0.0)
        m = w_mid.get(k, 0.0)
        e = w_end.get(k, 0.0)
        pct_change = ((e - s) / (s + 1e-7)) * 100.0 if s > 0 else 0.0

        if abs(pct_change) < 0.01:
            frozen_subnets.append(k)
        else:
            active_subnets.append(k)

        print(f"{k:<20} | {s:<12.4f} | {m:<12.4f} | {e:<12.4f} | {pct_change:>+10.2f}%")

    print("\n[DIAGNOSTIC VERDICT - WEIGHT UPDATES]:")
    if frozen_subnets:
        print(f"  [!] FROZEN / UNCHANGED SUB-NETWORKS (Change < 0.01%): {frozen_subnets}")
    if active_subnets:
        print(f"  [+] ACTIVELY UPDATING SUB-NETWORKS: {active_subnets}")

    # Gradient Magnitude Distribution
    print_header("2. SUBNET GRADIENT ENERGY & STARVATION ANALYSIS (MEAN ||G||)")
    print(f"{'Subnet':<20} | {'Mean ||G||':<12} | {'Max ||G||':<12} | {'Min ||G||':<12} | {'Status':<15}")
    print("-" * 75)

    subnet_grads = {k: [] for k in w_start.keys()}
    for r in all_records:
        g_dict = r.get("grad_norms", {})
        for k, v in g_dict.items():
            if k in subnet_grads:
                subnet_grads[k].append(v)

    for k, vals in subnet_grads.items():
        if not vals:
            continue
        mean_g = sum(vals) / len(vals)
        max_g = max(vals)
        min_g = min(vals)

        status = "HEALTHY"
        if mean_g < 1e-5:
            status = "STARVED / DEAD"
        elif mean_g > 10.0:
            status = "EXPLODING"

        print(f"{k:<20} | {mean_g:<12.6f} | {max_g:<12.6f} | {min_g:<12.6f} | {status:<15}")


def analyze_losses():
    loss_file = os.path.join(TELEMETRY_DIR, "train_iter_telemetry.jsonl")
    if not os.path.exists(loss_file):
        print(f"[!] File not found: {loss_file}")
        return

    print_header("3. LOSS TRAJECTORIES (START VS MID VS END)")
    records = []
    with open(loss_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return

    first = records[0]
    mid = records[len(records) // 2]
    last = records[-1]

    keys = [k for k in first.keys() if "loss" in k or k in ["total_loss", "lr", "ttt_prob"]]

    print(f"{'Metric / Loss':<18} | {'Epoch 1 (Start)':<15} | {'Epoch 7 (Mid)':<15} | {'Epoch 14 (End)':<15} | {'Delta':<12}")
    print("-" * 80)
    for k in keys:
        s = first.get(k, 0.0)
        m = mid.get(k, 0.0)
        e = last.get(k, 0.0)
        delta = e - s
        print(f"{k:<18} | {s:<15.4f} | {m:<15.4f} | {e:<15.4f} | {delta:>+10.4f}")


def analyze_simota():
    assign_file = os.path.join(TELEMETRY_DIR, "simota_telemetry.jsonl")
    if not os.path.exists(assign_file):
        print(f"[!] File not found: {assign_file}")
        return

    print_header("4. SIMOTA MATCHING & ANCHOR ASSIGNMENT TELEMETRY")
    records = []
    with open(assign_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("[!] No SimOTA records found.")
        return

    num_fgs = [r["num_fg_total"] for r in records if "num_fg_total" in r]
    ious = [r["matched_iou_mean"] for r in records if "matched_iou_mean" in r]
    d_ks = [r["dynamic_k_mean"] for r in records if "dynamic_k_mean" in r]

    if num_fgs:
        print(f"Matched Foregrounds per Batch (num_fg): Mean = {sum(num_fgs)/len(num_fgs):.1f}, Min = {min(num_fgs)}, Max = {max(num_fgs)}")
    if ious:
        print(f"Matched Anchor Mean IoU with GT:        Mean = {sum(ious)/len(ious):.4f}, Min = {min(ious):.4f}, Max = {max(ious):.4f}")
    if d_ks:
        print(f"Dynamic-K Value per Target:             Mean = {sum(d_ks)/len(d_ks):.2f}, Min = {min(d_ks):.2f}, Max = {max(d_ks):.2f}")


def analyze_prototypes():
    proto_file = os.path.join(TELEMETRY_DIR, "prototype_geometry.jsonl")
    if not os.path.exists(proto_file):
        print(f"[!] File not found: {proto_file}")
        return

    print_header("5. PROTOTYPE MANIFOLD GEOMETRY (EPOCH-LEVEL DRIFT)")
    records = []
    with open(proto_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("[!] No prototype records found.")
        return

    print(f"Total Logged Epochs: {len(records)}")
    first_epoch = records[0]
    last_epoch = records[-1]

    print(f"Comparing Epoch {first_epoch['epoch']} vs Epoch {last_epoch['epoch']}:")
    for lvl in first_epoch.get("levels", {}).keys():
        s_norms = first_epoch["levels"][lvl]["raw_l2_norms"]
        e_norms = last_epoch["levels"][lvl]["raw_l2_norms"]
        
        flat_s = [val for row in s_norms for val in row]
        flat_e = [val for row in e_norms for val in row]
        mean_s_norm = sum(flat_s) / len(flat_s)
        mean_e_norm = sum(flat_e) / len(flat_e)

        s_intra = first_epoch["levels"][lvl].get("intra_class_subcentroid_similarity", [])
        e_intra = last_epoch["levels"][lvl].get("intra_class_subcentroid_similarity", [])
        mean_s_intra = sum(s_intra) / len(s_intra) if s_intra else 0.0
        mean_e_intra = sum(e_intra) / len(e_intra) if e_intra else 0.0

        print(f"\n  [{lvl}]:")
        print(f"    Prototype Norm ||P||:  Start = {mean_s_norm:.3f} ──► End = {mean_e_norm:.3f}")
        print(f"    Intra-Class Cosine:    Start = {mean_s_intra:.3f} ──► End = {mean_e_intra:.3f}")


if __name__ == "__main__":
    if not os.path.exists(TELEMETRY_DIR):
        print(f"[!] Telemetry folder not found at: {TELEMETRY_DIR}")
        print("Please verify your path or current working directory.")
        sys.exit(1)

    analyze_weights_and_gradients()
    analyze_losses()
    analyze_simota()
    analyze_prototypes()
    print("\n" + "=" * 90 + "\n")