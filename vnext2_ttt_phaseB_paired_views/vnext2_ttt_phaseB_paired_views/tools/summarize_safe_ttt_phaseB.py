#!/usr/bin/env python3
"""Source-only Phase B report review; descriptive, never claims OOD/AP gains."""
from __future__ import annotations
import argparse
import json
import math
import statistics


def summarize_report(report):
    if not isinstance(report,dict) or 'trained_holdout' not in report or 'initial_holdout' not in report:
        raise ValueError('Expected Phase B paired-view report with before/after holdout')
    before=report['initial_holdout'];after=report['trained_holdout']
    if len(before)!=len(after) or len(before)==0:
        raise ValueError('Missing or unequal before/after source holdout')
    out={}
    for view in ('clean','shifted'):
        diffs=[]
        comps={k:[] for k in ('box','classification','positive_objectness','background_objectness')}
        scales={k:[] for k in ('small','medium','large')}
        positives={k:0 for k in scales}
        fg=[]
        for a,b in zip(before,after):
            if a['index']!=b['index'] or a['shifted_sha256']!=b['shifted_sha256']:
                raise ValueError('Mispaired images or changed appearance seeds')
            x=a[view];y=b[view]
            if abs(x['native']-y['native'])>2e-5:
                raise ValueError('Native source baseline changed or numerically unstable')
            diffs.append(y['delta']-x['delta'])
            for k in comps:comps[k].append(y['components'][k])
            for k in scales:
                if k in y['scales']:
                    scales[k].append(y['scales'][k]['adapted_minus_native_per_positive'])
                    positives[k]+=y['scales'][k]['count']
            fg.append(int(y['counts']['foreground']))
        out[view]={'count':len(diffs),
             'paired_training_effect_mean':statistics.mean(diffs),
             'paired_training_effect_median':statistics.median(diffs),
             'component_mean_adapt_minus_native':{k:statistics.mean(v) for k,v in comps.items()},
             'scale_present_image_count':{k:len(v) for k,v in scales.items()},
             'scale_matched_positive_count':positives,
             'scale_mean_per_positive_delta':{k:(statistics.mean(v) if v else None)
                                              for k,v in scales.items()},
             'matched_positive_total':sum(fg)}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',required=True)
    args=ap.parse_args()
    with open(args.report,encoding='utf-8') as f:report=json.load(f)
    calculated=summarize_report(report)
    for view in ('clean','shifted'):
        s=report['summary'][view];t=calculated[view]
        print('\nVIEW',view.upper(),'N',t['count'])
        print('initial_mean',f"{s['mean_initial_delta']:+.8g}",
              'trained_mean',f"{s['mean_trained_delta']:+.8g}",
              'paired_mean',f"{t['paired_training_effect_mean']:+.8g}",
              'paired_median',f"{t['paired_training_effect_median']:+.8g}")
        print('learned_better',s['better_than_initial'],'learned_worse',s['worse_than_initial'],
              'ttt_helpful',s['trained_helpful'],'ttt_harmful',s['trained_harmful'],
              'strict_scale_and_bg_pass',s['trained_strict'])
        print('loss_component_means',t['component_mean_adapt_minus_native'])
        print('per_scale_matched_count',t['scale_matched_positive_count'])
        print('per_scale_mean_delta',t['scale_mean_per_positive_delta'])
    print('\nWARNING: these are source TRAIN fixed-target surrogate losses. '
          'No AP/AR or unseen adverse generalization claims; positive scale coverage is required '
          'before interpreting scale-specific means.')
    if len(report['initial_holdout'])<32:
        print('SMALL_SAMPLE: fewer than 32 independent source holdout images; '
              'do not treat this as a generalization acceptance test')

if __name__=='__main__':main()
