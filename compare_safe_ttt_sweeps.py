#!/usr/bin/env python3
"""Compare vNext.2 trained vs untrained fixed-target TTT step sweep JSONs.

No training, checkpoint access, or dependencies beyond Python stdlib.

Usage (from YOLOX repo root):
  python compare_safe_ttt_sweeps.py \
    --trained audits/safe_ttt_step_sweep_pilot.json \
    --untrained audits/safe_ttt_step_sweep_untrained.json

This is a research diagnostic of SOURCE clean training-image fixed-target loss.
"""
import argparse
import json
import math
import statistics as stats


def render_row(name, report, step, eps):
    records=[(img['index'],img['candidates'][str(step)]) for img in report['images']]
    ds=[row['total_delta'] for _,row in records]
    strict=sum(bool(row.get('strict_surrogate_pass')) for _,row in records)
    nonreg=sum(bool(row.get('scale_nonregressing')) for _,row in records)
    bg=sum(bool(row.get('background_nonregressing')) for _,row in records)
    cos=[img.get('ssl_detector_grad_cosine') for img in report['images']]
    cos=[v for v in cos if isinstance(v,(int,float)) and math.isfinite(v)]
    taylor=[row.get('taylor_predicted_delta') for _,row in records]
    taylor_pairs=[(a,b) for a,b in zip(ds,taylor) if isinstance(b,(int,float)) and math.isfinite(b)]
    sign_ok=sum(a*b>0 for a,b in taylor_pairs if abs(a)>eps and abs(b)>eps)
    sign_total=sum(1 for a,b in taylor_pairs if abs(a)>eps and abs(b)>eps)
    print(f'{name:<9} {step:<7} {stats.mean(ds):+.3e} {stats.median(ds):+.3e} '
          f'{sum(bool(row.get("meaningfully_beneficial",v<-eps)) for (_,row),v in zip(records,ds)):>2}/{len(ds)} {sum(bool(row.get("meaningfully_harmful",v>eps)) for (_,row),v in zip(records,ds)):>2}/{len(ds)} '
          f'{strict:>2} {nonreg:>2} {bg:>2} '
          f'{stats.mean(cos):+.3f}' if cos else
          f'{name:<9} {step:<7} {stats.mean(ds):+.3e} {stats.median(ds):+.3e} '
          f'{sum(bool(row.get("meaningfully_beneficial",v<-eps)) for (_,row),v in zip(records,ds)):>2}/{len(ds)} {sum(bool(row.get("meaningfully_harmful",v>eps)) for (_,row),v in zip(records,ds)):>2}/{len(ds)} '
          f'{strict:>2} {nonreg:>2} {bg:>2} N/A')
    print(f'  finite-step Taylor sign agreement (excluding effects <= {eps:g}): {sign_ok}/{sign_total} eligible')


def run(a,b,eps):
    pa,pb=a['protocol'],b['protocol']
    for key in ('model_hash','checkpoint','weights','seed','step_scales','step_policy','rank','basis_gain','indices'):
        if pa.get(key)!=pb.get(key):
            raise SystemExit(f'INVALID PAIRED COMPARISON: protocol differs on {key}: '
                             f'{pa.get(key)!r} != {pb.get(key)!r}')
    ia={int(x['index']):x for x in a['images']}
    ib={int(x['index']):x for x in b['images']}
    if ia.keys()!=ib.keys():
        raise SystemExit('INVALID PAIRED COMPARISON: indices do not match')
    if len(ia)!=len(a['images']) or len(ib)!=len(b['images']):
        raise SystemExit('INVALID PAIRED COMPARISON: duplicate indices')
    assert pa.get('safe_weights') and not pb.get('safe_weights'), 'Expected trained and untrained files in specified order'
    for i in sorted(ia):
        aa=ia[i].get('native',{})
        bb=ib[i].get('native',{})
        if aa.get('counts') != bb.get('counts'):
            raise SystemExit(f'INVALID PAIRED COMPARISON: native assignment/scale counts differ on image {i}')
        if abs(float(aa.get('total',0))-float(bb.get('total',0))) > 1e-5:
            raise SystemExit(f'INVALID PAIRED COMPARISON: native reference loss differs on image {i}')
        if ia[i].get('total_gt') != ib[i].get('total_gt'):
            raise SystemExit(f'INVALID PAIRED COMPARISON: GT counts differ on image {i}')
    steps=[float(x) for x in pa['step_scales'] if float(x)>0]
    print(f'PAIRED PROTOCOL OK: {len(ia)} images; shared native model hash and seed; '
          f'step strength = {steps}\n')
    print(f'{"sidecar":<9} {"lambda":<7} {"mean_delta":>11} {"median_delta":>11} '
          f'{"benefit":>7} {"harm":>6} {"strict":>6} {"scale":>6} {"bg":>3} {"mean_cos":>9}')
    for s in steps:
        render_row('untrained',b,s,eps)
        render_row('trained',a,s,eps)
        diffs=[ia[i]['candidates'][str(s)]['total_delta']-ib[i]['candidates'][str(s)]['total_delta']
               for i in sorted(ia)]
        print(f'  TRAINED minus UNTRAINED: mean {stats.mean(diffs):+.3e}, '
              f'better {sum(d < -eps for d in diffs)}/{len(diffs)}, '
              f'worse {sum(d > eps for d in diffs)}/{len(diffs)}, '
              f'near-tied {sum(abs(d)<=eps for d in diffs)}/{len(diffs)}')
        print()
    for name,rs in [('trained',a),('untrained',b)]:
        counts={'any_improvement':0,'any_strict':0}
        for x in rs['images']:
            choices=[x['candidates'][str(s)] for s in steps]
            counts['any_improvement']+=any(z.get('meaningfully_beneficial', z['total_delta'] < -eps) for z in choices)
            counts['any_strict']+=any(z.get('strict_surrogate_pass',False) for z in choices)
        print(name,'ORACLE (offline only; not deployable)',counts)
    print('\n--- Per-image comparison at 0.5 step (trained minus untrained) ---')
    if .5 in steps:
        for i in sorted(ia):
            ta=ia[i]['candidates']['0.5']['total_delta']
            ub=ib[i]['candidates']['0.5']['total_delta']
            print(f'{i:>5}: trained={ta:+.3e}, untrained={ub:+.3e}, paired={ta-ub:+.3e}, '
                  f'train_cos={ia[i].get("ssl_detector_grad_cosine")}, '
                  f'untrain_cos={ib[i].get("ssl_detector_grad_cosine")}')
    print('\nLIMITATIONS: These are source-domain fixed-native-target surrogate effects, '
          'not post-NMS AP/AR, not test-time non-regression, and not a calibrated significance test.')
    print('Inspect Taylor prediction/error, actual component deltas and per-scale flags '
          'before treating tiny (~1e-4) loss changes as reliable improvements.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trained',required=True)
    p.add_argument('--untrained',required=True)
    p.add_argument('--eps',type=float,default=1e-5)
    args=p.parse_args()
    if args.eps<0: p.error('eps must be nonnegative')
    with open(args.trained,encoding='utf-8') as f: ta=json.load(f)
    with open(args.untrained,encoding='utf-8') as f: ub=json.load(f)
    run(ta,ub,args.eps)
