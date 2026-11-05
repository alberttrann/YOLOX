#!/usr/bin/env python3
"""Read-only source-label Phase B safety/gradient-mechanism accounting.

Uses the report produced by tools/pilot_safe_ttt_paired_views.py.
Does not access images, GPU, ACDC, or modify model/checkpoint.
"""
import argparse
import json
import math
import statistics
import random

TASKS = ('box', 'classification', 'positive_objectness', 'background_objectness')
SCALES = ('small', 'medium', 'large')


def mean(xs):
    return statistics.mean(xs) if xs else None


def bootstrap_ci(values, B=4000, seed=7741):
    """Descriptive image-cluster bootstrap; does not correct selection or model variance."""
    n=len(values)
    if n < 2:
        return (None, None)
    r=random.Random(seed)
    boots=sorted(sum(values[r.randrange(n)] for _ in range(n))/n for _ in range(B))
    return boots[int(.025*B)], boots[min(B-1,int(.975*B))]


def assess(report):
    before=report['initial_holdout']
    after=report['trained_holdout']
    if not before or len(before)!=len(after):
        raise ValueError('Missing or mispaired Phase B holdout records')
    if len({x['index'] for x in before}) != len(before):
        raise ValueError('Duplicate source holdout image index')
    output={}
    for view in ('clean','shifted'):
        records=[]
        for a,b in zip(before,after):
            if a['index']!=b['index'] or a['clean_sha256'] != b['clean_sha256'] or a['shifted_sha256']!=b['shifted_sha256']:
                raise ValueError('Changed source holdout identity or photometric transform')
            x,y=a[view],b[view]
            if abs(x['native']-y['native'])>2e-5:
                raise ValueError('Native loss mismatch; cannot attribute changes to sidecar')
            p={'id':a['index'], 'initial':x['delta'], 'trained':y['delta'], 'paired':y['delta']-x['delta']}
            p['components']={k:y['components'][k] for k in TASKS}
            p['component_paired']={k:y['components'][k]-x['components'][k] for k in TASKS}
            p['scales']={}
            for k in SCALES:
                if k not in y['scales']:continue
                if k not in x['scales'] or x['scales'][k]['count']!=y['scales'][k]['count']:
                    raise ValueError('Changed source target scale/assignment')
                p['scales'][k]={'count':y['scales'][k]['count'],
                    'trained':y['scales'][k]['adapted_minus_native_per_positive'],
                    'paired':y['scales'][k]['adapted_minus_native_per_positive']-x['scales'][k]['adapted_minus_native_per_positive']}
            nfg=int(y['counts']['foreground']); nbg=int(y['counts']['background'])
            if nfg<=0 or nbg<=0:continue
            factor=nfg/nbg
            bg_total=y['components']['background_objectness']
            bg_actual=float(y['background']['delta_per_background_anchor'])
            if abs(bg_actual-bg_total*factor)>max(2e-6,abs(bg_actual)*0.05):
                raise ValueError('Inconsistent background normalization in Phase B report')
            p['background']={
                'positive_matches':nfg,'background_anchors':nbg,
                'per_negative_factor':factor,
                'total_detection_delta':bg_total,
                'per_negative_delta':bg_actual,
                'strict_guard_threshold_in_total_loss_units':2e-5/factor,
                'extra_hinge_to_detector_gradient_ratio_when_active':0.5*0.1*factor,
                'extra_hinge_loss_term':0.5*0.1*max(bg_actual,0.0),
                'strict_background_guard_violated':bg_actual>2e-5,
            }
            p['feature_delta']=y.get('relative_feature_change_postcast')
            records.append(p)
        paired=[x['paired'] for x in records]
        output[view]={
            'n':len(records),
            'paired_mean':mean(paired),'paired_median':statistics.median(paired),
            'paired_image_bootstrap_95pct':bootstrap_ci(paired),
            'trained_minus_native_mean':mean([r['trained'] for r in records]),
            'paired_component_means':{k:mean([r['component_paired'][k] for r in records]) for k in TASKS},
            'trained_component_means':{k:mean([r['components'][k] for r in records]) for k in TASKS},
            'trained_scale_means_unweighted_images':{k:mean([r['scales'][k]['trained'] for r in records if k in r['scales']]) for k in SCALES},
            'paired_scale_means_unweighted_images':{k:mean([r['scales'][k]['paired'] for r in records if k in r['scales']]) for k in SCALES},
            'scale_image_coverage':{k:sum(k in r['scales'] for r in records) for k in SCALES},
            'background':{
                'median_guard_threshold_in_total_loss_units':statistics.median([r['background']['strict_guard_threshold_in_total_loss_units'] for r in records]),
                'mean_bg_hinge_gradient_multiplier':mean([r['background']['extra_hinge_to_detector_gradient_ratio_when_active'] for r in records]),
                'mean_extra_hinge_loss':mean([r['background']['extra_hinge_loss_term'] for r in records]),
                'mean_total_bg_detection_delta':mean([r['background']['total_detection_delta'] for r in records]),
                'strict_bg_gate_violations':sum(r['background']['strict_background_guard_violated'] for r in records),
                'mean_foreground_anchor_count':mean([r['background']['positive_matches'] for r in records]),
                'mean_background_anchor_count':mean([r['background']['background_anchors'] for r in records]),
            },
            'records':records}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True)
    p.add_argument('--json',default=None,help='Optional destination for report; refuses overwrite')
    args=p.parse_args()
    with open(args.report,'r',encoding='utf-8') as f: report=json.load(f)
    results=assess(report)
    for v,s in results.items():
        print('\nVIEW',v.upper(),'N',s['n'])
        print('PAIRED mean',f'{s["paired_mean"]:+.8g}','median',f'{s["paired_median"]:+.8g}',
              'image-bootstrap 95% descriptive interval', tuple(f'{x:+.8g}' for x in s['paired_image_bootstrap_95pct']))
        print('AFTER vs native',f'{s["trained_minus_native_mean"]:+.8g}')
        print('TRAINED task components',s['trained_component_means'])
        print('TRAINING CHANGES by task',s['paired_component_means'])
        print('TRAINED per-scale',s['trained_scale_means_unweighted_images'])
        print('TRAINING CHANGES per scale',s['paired_scale_means_unweighted_images'])
        print('SCALE image coverage',s['scale_image_coverage'])
        bg=s['background']
        print('BG mean total delta',f'{bg["mean_total_bg_detection_delta"]:+.8g}',
              'mean +hinge term',f'{bg["mean_extra_hinge_loss"]:+.8g}')
        print('BG extra-hinge gradient multiplier relative to native bg component (WHEN ACTIVE)',
              f'{bg["mean_bg_hinge_gradient_multiplier"]:.6g}')
        print('BG guard equivalent aggregate-loss threshold (median)',f'{bg["median_guard_threshold_in_total_loss_units"]:.6g}',
              'guard violations',bg['strict_bg_gate_violations'],'/',s['n'])
        print('MEAN foreground/bg anchors',bg['mean_foreground_anchor_count'],bg['mean_background_anchor_count'])
    print('\nCAUTION: paired intervals reflect only sampled images, not seeds or source distribution coverage.')
    print('Small detection-loss deltas are not post-NMS AP/AR. An apparent background guard pass is not a safety guarantee.')
    if args.json:
        from pathlib import Path
        out=Path(args.json)
        if out.exists():raise FileExistsError(str(out))
        out.write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')

if __name__=='__main__':main()
