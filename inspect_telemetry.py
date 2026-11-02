#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Human-readable live summary for tdevNext telemetry schema v3."""
import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path, tail=None):
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows[-tail:] if tail and tail > 0 else rows


def header(title):
    print('\n' + '=' * 105)
    print(title)
    print('=' * 105)


def fmt(x, digits=4):
    if x is None:
        return 'N/A'
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    try:
        x = float(x)
        if not math.isfinite(x):
            return 'INVALID'
        if abs(x) < 1e-3 and x != 0:
            return f'{x:.3e}'
        return f'{x:.{digits}f}'
    except Exception:
        return str(x)


def analyze_runtime(tdir, tail):
    rows = load_jsonl(tdir / 'train_iter_telemetry.jsonl', tail)
    if not rows:
        return
    header('1. TRAIN / AMP / LR STATE')
    first, last = rows[0], rows[-1]
    print(f"Window: {len(rows)} records | epoch {first.get('epoch')} iter {first.get('iter')} -> "
          f"epoch {last.get('epoch')} iter {last.get('iter')}")
    print(f"LR used: {fmt(last.get('lr_used'), 8)} | next LR: {fmt(last.get('lr_next'), 8)}")
    print(f"AMP scale: {fmt(last.get('amp_scale_before'), 1)} -> {fmt(last.get('amp_scale_after'), 1)} | "
          f"total skipped steps: {last.get('amp_skipped_steps_total', 0)}")
    print(f"Pre-clip grad norm: {fmt(last.get('pre_clip_total_grad_norm'))} | "
          f"clip coeff: {fmt(last.get('clip_coefficient'))} | clipped={last.get('was_clipped')}")
    loss_keys = ('total_loss','det_loss','aux_total','iou_loss','conf_loss','cls_loss',
                 'mem_loss','proto_loss','intra_loss','rec_loss','afford_loss')
    print('\nLatest losses:')
    for k in loss_keys:
        if k in last:
            print(f"  {k:<18} {fmt(last[k], 6)}")
    contribs = ('mem_contrib','proto_contrib','intra_contrib','rec_contrib','afford_contrib')
    if any(k in last for k in contribs):
        print('\nActual weighted auxiliary contributions to total_loss:')
        for k in contribs:
            if k in last:
                print(f"  {k:<18} {fmt(last[k], 7)}")


def analyze_gradients(tdir, tail):
    rows = load_jsonl(tdir / 'gradient_flow_telemetry.jsonl', tail)
    if not rows:
        return
    header('2. PRE/POST-CLIP GRADIENT FLOW')
    valid_coeff = [r.get('clip_coefficient') for r in rows if r.get('clip_coefficient') is not None]
    clipped = [c for c in valid_coeff if c < 0.999999]
    if valid_coeff:
        print(f"Clipped records: {len(clipped)}/{len(valid_coeff)} ({100*len(clipped)/len(valid_coeff):.1f}%) | "
              f"mean clip coeff={sum(valid_coeff)/len(valid_coeff):.4f} | min={min(valid_coeff):.4f}")
    last = rows[-1]
    print(f"Latest total ||g||: pre={fmt(last.get('pre_clip_total_grad_norm'))} "
          f"post={fmt(last.get('post_clip_total_grad_norm'))}")
    print(f"{'Group':<24} {'pre ||g||':>12} {'post ||g||':>12} {'energy %':>10} {'g/w':>12} {'NZ/grad':>10}")
    print('-' * 86)
    groups = last.get('groups', {})
    ranked = sorted(groups.items(), key=lambda kv: (kv[1].get('grad_norm') or -1), reverse=True)
    for name, g in ranked:
        energy = g.get('gradient_energy_fraction')
        energy = 100 * energy if energy is not None else None
        nz = f"{g.get('nonzero_grad_tensors',0)}/{g.get('with_grad',0)}"
        print(f"{name:<24} {fmt(g.get('grad_norm')):>12} {fmt(g.get('post_clip_grad_norm')):>12} "
              f"{fmt(energy,2):>10} {fmt(g.get('gradient_to_weight_ratio'),6):>12} {nz:>10}")


def analyze_updates(tdir, tail):
    rows = load_jsonl(tdir / 'parameter_update_telemetry.jsonl', tail)
    if not rows:
        return
    header('3. ACTUAL PARAMETER MOVEMENT')
    last = rows[-1]
    print(f"Latest mode={last.get('mode')} optimizer_step={last.get('optimizer_step')} "
          f"lr_used={fmt(last.get('lr_used'),8)} clip_coeff={fmt(last.get('clip_coefficient'))}")
    if last.get('groups'):
        print('\nExact full-group update (heavy interval):')
        print(f"{'Group':<24} {'rel delta':>13} {'delta L2':>13} {'RMS/elem':>13} {'changed':>11}")
        for name, g in sorted(last['groups'].items(), key=lambda kv: kv[1].get('actual_relative_delta',0), reverse=True):
            changed = f"{g.get('changed_tensors',0)}/{g.get('parameter_tensors',0)}"
            print(f"{name:<24} {fmt(g.get('actual_relative_delta'),7):>13} {fmt(g.get('actual_delta_l2'),7):>13} "
                  f"{fmt(g.get('update_rms_per_element'),7):>13} {changed:>11}")
    print('\nSentinel/control updates:')
    for name, g in last.get('sentinels', {}).items():
        print(f"  {name:<58} relΔ={fmt(g.get('actual_relative_delta'),8)} "
              f"ΔL2={fmt(g.get('actual_delta_l2'),8)} value={fmt(g.get('value_mean'),8)}")


def analyze_assignments(tdir, tail):
    rows = load_jsonl(tdir / 'simota_telemetry.jsonl', tail)
    if not rows:
        return
    header('4. SCALE-AWARE SIMOTA / FPN ASSIGNMENT HEALTH')
    total_gt = sum(int(r.get('num_gt_total',0)) for r in rows)
    total_fg = sum(int(r.get('num_fg_total',0)) for r in rows)
    invalid_iou = sum(int((r.get('matched_iou') or {}).get('invalid_or_out_of_range',0)) for r in rows)
    print(f"Window totals: GT={total_gt}, FG={total_fg}, FG/GT={total_fg/max(1,total_gt):.3f}, "
          f"invalid matched IoU={invalid_iou}")

    size_names = ('micro','medium','macro')
    print('\nBy current augmented object scale (sqrt(box area), thresholds 32/96 px):')
    print(f"{'Scale':<10} {'GT':>8} {'FG':>9} {'FG/GT':>9} {'coverage':>11} {'zero-GT':>9} {'mean IoU':>10}")
    for name in size_names:
        gt = fg = covered = zero = 0
        iou_weighted = 0.0
        iou_count = 0
        for r in rows:
            d = (r.get('by_size') or {}).get(name, {})
            g = int(d.get('gt',0)); p = int(d.get('positive_anchors',0)); c = int(d.get('covered_gt',0)); z = int(d.get('zero_positive_gt',0))
            gt += g; fg += p; covered += c; zero += z
            miou = d.get('mean_matched_iou')
            if miou is not None and p:
                iou_weighted += float(miou) * p; iou_count += p
        print(f"{name:<10} {gt:>8} {fg:>9} {fg/max(1,gt):>9.3f} {covered/max(1,gt):>10.2%} "
              f"{zero:>9} {fmt(iou_weighted/max(1,iou_count) if iou_count else None):>10}")

    strides = sorted({k for r in rows for k in (r.get('by_fpn') or {}).keys()})
    if strides:
        print('\nFPN positive routing:')
        for stride in strides:
            positives = sum(int((r.get('by_fpn') or {}).get(stride,{}).get('positive_anchors',0)) for r in rows)
            print(f"  {stride:<12} positives={positives:<8} fraction={positives/max(1,total_fg):.2%}")

    requested = sum(int((r.get('dynamic_k') or {}).get('requested_positives_pre_conflict',0)) for r in rows)
    conflicts = sum(int((r.get('dynamic_k') or {}).get('conflict_anchors',0)) for r in rows)
    fallback = sum(int((r.get('geometry') or {}).get('fallback_gt',0)) for r in rows)
    dyn_vals = [(r.get('dynamic_k') or {}).get('mean') for r in rows]
    dyn_vals = [float(x) for x in dyn_vals if x is not None]
    print('\nDynamic-K / geometry:')
    print(f"  mean dynamic-k={fmt(sum(dyn_vals)/len(dyn_vals) if dyn_vals else None)} | "
          f"requested positives={requested} | final={total_fg} | retention={total_fg/max(1,requested):.2%}")
    print(f"  conflict anchors={conflicts} | nearest-anchor fallback GT={fallback} ({fallback/max(1,total_gt):.3%})")


def analyze_architecture(tdir, tail):
    rows = load_jsonl(tdir / 'architecture_state_telemetry.jsonl', tail)
    if not rows:
        return
    header('5. GATES / RESIDUAL CONTROLS / TTT META-LR')
    first, last = rows[0], rows[-1]
    controls = last.get('controls', {})
    for name, d in controls.items():
        start = (first.get('controls', {}).get(name) or {}).get('mean')
        print(f"  {name:<62} value={fmt(d.get('mean'),8)} start={fmt(start,8)} absmax={fmt(d.get('abs_max'),8)}")
    ttt = last.get('ttt_meta_lr') or {}
    if ttt:
        print(f"\n  TTT LR ceiling: {fmt(ttt.get('lr_ceiling'),8)}")
        for name, d in (ttt.get('parameters') or {}).items():
            print(f"    {name:<24} actual_inner_lr={fmt(d.get('actual_inner_lr'),8)} sigmoid={fmt(d.get('sigmoid_fraction'),6)}")


def analyze_prototypes(tdir):
    rows = load_jsonl(tdir / 'prototype_geometry.jsonl')
    if not rows:
        return
    header('6. PROTOTYPE GEOMETRY')
    first, last = rows[0], rows[-1]
    print(f"Epoch records: {len(rows)} | {first.get('epoch')} -> {last.get('epoch')}")
    for lvl, d in last.get('levels', {}).items():
        norms = [x for row in d.get('raw_l2_norms',[]) for x in row]
        intra = d.get('intra_class_subcentroid_similarity',[])
        print(f"  {lvl:<20} mean ||P||={fmt(sum(norms)/len(norms) if norms else None)} "
              f"mean intra-cos={fmt(sum(intra)/len(intra) if intra else None)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='YOLOX_outputs/tde_yolox_s_full_vnext2/telemetry')
    ap.add_argument('--tail', type=int, default=100, help='recent telemetry records to summarize; 0=all')
    args = ap.parse_args()
    tdir = Path(args.dir)
    if not tdir.exists():
        raise SystemExit(f'Telemetry folder not found: {tdir}')
    tail = None if args.tail == 0 else args.tail
    analyze_runtime(tdir, tail)
    analyze_gradients(tdir, tail)
    analyze_updates(tdir, tail)
    analyze_assignments(tdir, tail)
    analyze_architecture(tdir, tail)
    analyze_prototypes(tdir)


if __name__ == '__main__':
    main()
