#!/usr/bin/env python3
"""Audit historical tde-vNext gradient telemetry WITHOUT rerunning training.

Reads JSONL files written by TelemetryRecorder, reports NaN/Inf rates per epoch,
fully-nonfinite groups, iteration discontinuities, and run-append ambiguity.

Examples:
 python tdevnext_history_audit.py
 python tdevnext_history_audit.py --grad YOLOX_outputs/tde_yolox_s/telemetry/gradient_flow_telemetry.jsonl
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median


def read_jsonl(path):
    records=[]
    for line_no,line in enumerate(Path(path).open('r',encoding='utf-8'),1):
        if not line.strip(): continue
        try: records.append((line_no,json.loads(line)))
        except (ValueError,TypeError) as e: print(f'WARN: Invalid JSON at line {line_no}: {e}')
    return records


def is_bad(x):
    try:return not math.isfinite(float(x))
    except (ValueError,TypeError,OverflowError):return True


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--grad',default='YOLOX_outputs/tde_yolox_s/telemetry/gradient_flow_telemetry.jsonl')
    a.add_argument('--loss',default='YOLOX_outputs/tde_yolox_s/telemetry/train_iter_telemetry.jsonl')
    args=a.parse_args()
    records=read_jsonl(args.grad)
    if not records: raise SystemExit('No records available')
    print(f'Gradient records: {len(records)} in {args.grad}')
    per_epoch=defaultdict(lambda:{'count':0,'bad_any':0,'bad_all':0,'finite_max':[],'first_iter':None,'last_iter':None})
    per_group=Counter()
    suspicious=[]
    prev=None
    totals=Counter()
    for line_no,r in records:
        epoch=r.get('epoch','?'); it=r.get('iter')
        if prev and isinstance(it,(int,float)) and isinstance(prev[1],(int,float)):
            if it <= prev[1]: suspicious.append((line_no,'non-increasing global iteration',prev[1],it,prev[0],epoch))
        if prev and isinstance(epoch,(int,float)) and isinstance(prev[0],(int,float)) and epoch < prev[0]:
            suspicious.append((line_no,'epoch decreased',prev[1],it,prev[0],epoch))
        prev=(epoch,it)
        groups=r.get('grad_norms') or {}
        flags=[is_bad(v) for v in groups.values()]
        anybad=bool(flags) and any(flags)
        allbad=bool(flags) and all(flags)
        ee=per_epoch[epoch]
        ee['count']+=1
        ee['bad_any']+=anybad
        ee['bad_all']+=allbad
        if ee['first_iter'] is None: ee['first_iter']=it
        ee['last_iter']=it
        for name,v in groups.items():
            if is_bad(v): per_group[name]+=1
            else: ee['finite_max'].append(float(v))
        totals['bad_any']+=anybad
        totals['bad_all']+=allbad
        totals['total_bad']+=is_bad(r.get('total_grad_norm',0))
    n=len(records)
    print(f'Any group nonfinite: {totals["bad_any"]}/{n} ({totals["bad_any"]/n:.2%})')
    print(f'Every recorded group nonfinite: {totals["bad_all"]}/{n} ({totals["bad_all"]/n:.2%})')
    print(f'Total reported norm nonfinite: {totals["total_bad"]}/{n} ({totals["total_bad"]/n:.2%})')
    print('\nPer-epoch snapshot (sampled every print_interval iterations):')
    print(' epoch | samples | any_bad | all_bad | first_iter | last_iter')
    for epoch,s in sorted(per_epoch.items(),key=lambda kv: str(kv[0]).zfill(10)):
        print(f'{str(epoch):>6} | {s["count"]:>7} | {s["bad_any"]/s["count"]:>6.1%} | {s["bad_all"]/s["count"]:>6.1%} | {str(s["first_iter"]):>10} | {str(s["last_iter"]):>9}')
    print('\nNonfinite group counts:')
    for group,n_bad in per_group.most_common():
        print(f'  {group:25s} {n_bad:>6}/{n} ({n_bad/n:.2%})')
    print(f'\nIteration / restart discontinuities: {len(suspicious)}')
    for x in suspicious[:15]:
        print('  line=%d %s: iter %s -> %s, epoch %s -> %s' % x)
    if len(suspicious)>15:print(f'  ...and {len(suspicious)-15} more')
    if Path(args.loss).exists():
        losses=read_jsonl(args.loss)
        print(f'\nLoss records: {len(losses)} in {args.loss}')
        per_loss=defaultdict(list)
        for _,r in losses:
            if 'total_loss' in r and not is_bad(r['total_loss']):per_loss[r.get('epoch','?')].append(float(r['total_loss']))
        for epoch,v in sorted(per_loss.items(),key=lambda kv: str(kv[0]).zfill(10)):
            print(f'  epoch={epoch} total_loss median={median(v):.4f} (n={len(v)})')
    print('\nIMPORTANT: Old telemetry does NOT record GradScaler scale or optimizer.step().')
    print('A nonfinite sampled norm is not proof of a skipped optimizer step; step rate cannot be recovered.')
    print('An appended JSONL from multiple runs cannot be fully disambiguated without a run ID.')

if __name__=='__main__': main()
