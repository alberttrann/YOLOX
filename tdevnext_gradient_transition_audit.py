#!/usr/bin/env python3
"""Forensically segment historical YOLOX gradient telemetry and identify transitions.

Read-only: never changes checkpoint, model or source.  No third-party packages.
Run from YOLOX repo root: python tdevnext_gradient_transition_audit.py
"""
import argparse
import collections
import json
import math
from pathlib import Path


def load(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f'Invalid JSON at {path}:{line_no}: {e}')
                continue
            out.append((line_no, row))
    return out


def kind(x):
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return 'invalid'
    if math.isnan(v): return 'nan'
    if math.isinf(v): return '+inf' if v > 0 else '-inf'
    return 'finite'


def summarize(segment, idx, loss_lookup):
    lines = [row for _, row in segment]
    groups = [row.get('grad_norms', {}) for row in lines]
    n = len(lines)
    flags = [any(kind(v) != 'finite' for v in group.values()) for group in groups]
    allflags = [bool(group) and all(kind(v) != 'finite' for v in group.values()) for group in groups]
    weights_bad = sum(any(kind(v) != 'finite' for v in r.get('weight_norms',{}).values()) for r in lines)
    count_kinds = collections.Counter(kind(v) for group in groups for v in group.values())
    print(f'\n=== RUN SEGMENT {idx} ===')
    print(f'Line range: {segment[0][0]}–{segment[-1][0]} ({n} records)')
    print(f'Epoch/iteration span: {lines[0].get("epoch")}/{lines[0].get("iter")} -> {lines[-1].get("epoch")}/{lines[-1].get("iter")}')
    print(f'Any gradient group nonfinite: {sum(flags)}/{n} ({sum(flags)/n:.2%})')
    print(f'All gradient groups nonfinite: {sum(allflags)}/{n} ({sum(allflags)/n:.2%})')
    print(f'Any weight group norm nonfinite: {weights_bad}/{n}')
    print('Gradient norm value types:', dict(count_kinds))
    by_epoch = collections.defaultdict(lambda: [0, 0, 0])
    for row, b, a in zip(lines, flags, allflags):
        z = by_epoch[row.get('epoch')]; z[0]+=1; z[1]+=b; z[2]+=a
    print('Epoch: n / any-nonfinite / all-nonfinite')
    for e, (s,b,a) in sorted(by_epoch.items()):
        print(f'  {e:3}: {s:4} / {b:4} ({b/s:.1%}) / {a:4} ({a/s:.1%})')
    first = next((i for i,b in enumerate(flags) if b), None)
    persistent = next((i for i,b in enumerate(flags) if all(flags[i:])), None)
    print('First bad line:', segment[first][0] if first is not None else 'none')
    print('First bad line in permanently bad suffix:', segment[persistent][0] if persistent is not None else 'none')
    if first is not None:
        def snapshot(j):
            line_no, row = segment[j]
            c = collections.Counter(kind(v) for v in row.get('grad_norms',{}).values())
            loss_record = loss_lookup.get((row.get('epoch'),row.get('iter')), {})
            return (f'line={line_no} epoch={row.get("epoch")} iter={row.get("iter")} '
                    f'grad_types={dict(c)} total_norm={row.get("total_grad_norm")} '
                    f'weights_bad={sum(kind(v)!="finite" for v in row.get("weight_norms",{}).values())} '
                    f'loss={loss_record.get("total_loss")} lr={loss_record.get("lr")} '
                    f'timestamp={row.get("timestamp")}')
        lo = max(0, first - 4); hi = min(n, first + 5)
        print('\nTransition around first bad:')
        for j in range(lo, hi):print('  ',snapshot(j))
        if persistent is not None and persistent != first:
            print('\nTransition into permanently bad suffix:')
            lo=max(0,persistent-4); hi=min(n,persistent+4)
            for j in range(lo,hi):print('  ',snapshot(j))
    print('\nSample final records:')
    for j in range(max(0,n-3), n):
        r=lines[j]
        print(f'  line={segment[j][0]} epoch={r.get("epoch")} iter={r.get("iter")} '
              f'grad_norm_types={dict(collections.Counter(kind(v) for v in r.get("grad_norms",{}).values()))} '
              f'weight_norm_types={dict(collections.Counter(kind(v) for v in r.get("weight_norms",{}).values()))}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grad', default='YOLOX_outputs/tde_yolox_s/telemetry/gradient_flow_telemetry.jsonl')
    p.add_argument('--loss', default='YOLOX_outputs/tde_yolox_s/telemetry/train_iter_telemetry.jsonl')
    a=p.parse_args()
    rs=load(a.grad)
    if not rs:raise SystemExit('No gradient records')
    loss_lookup={}
    if Path(a.loss).exists():
        for _,r in load(a.loss): loss_lookup[(r.get('epoch'),r.get('iter'))]=r
    segments=[]; current=[]; prev=None
    for item in rs:
        r=item[1]
        if prev is not None and (r.get('epoch',0)<prev.get('epoch',0) or r.get('iter',0)<=prev.get('iter',0)):
            segments.append(current); current=[]
        current.append(item); prev=r
    if current: segments.append(current)
    print(f'Total records: {len(rs)}, independent monotonic segments: {len(segments)}')
    for idx,segment in enumerate(segments,1): summarize(segment,idx,loss_lookup)
    print('\nCaution: non-finite recorded norms are not sufficient to infer optimizer-step success.')
    print('A finite forward loss does not exclude an invalid backward pass.')


if __name__ == '__main__': main()
