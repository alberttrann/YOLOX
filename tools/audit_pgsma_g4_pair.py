"""Read-only, stdlib-only paired comparison for Pilot G4 semantic objectives.

Compares the v2 and margin_probe runs at *identical native state*, using their
training logs, final result, and optional matched-seed G3 geometry reports.
Never loads a model or changes checkpoints.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

COCO = ('AP50_95','AP50','AP75','AR100','AP_small_COCO','AP_medium_COCO','AP_large_COCO')
KEYS = ('candidate_fg_improvement','candidate_bg_improvement',
        'gate_fg_improvement','gate_bg_improvement',
        'mean_abs_candidate_delta','mean_gate_acceptance')
SIZE = ('lt4','4to8','8to16','16to32','ge32')
LEVEL = ('P3','P4','P5')


def read_json(path):
    with Path(path).open('r', encoding='utf8') as f:
        return json.load(f)


def read_metrics(path):
    rows = {}
    with Path(path).open('r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                step = int(r['step'])
                if step in rows:
                    raise ValueError(f'duplicate training step {step}: {path}')
                rows[step] = r
    return rows


def mean(values):
    return statistics.mean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def weighted(values, weights):
    use = [(v,w) for v,w in zip(values,weights) if v is not None and w]
    return sum(v*w for v,w in use)/sum(w for v,w in use) if use else None


def train_path(left, right):
    keys = sorted(set(left) & set(right))
    if not keys: raise ValueError('No common logged training steps')
    out = []
    for k in keys:
        a,b = left[k],right[k]
        aa,bb = a.get('semantic_utility'),b.get('semantic_utility')
        row = {'step':k,'native_loss_v2':a.get('native_loss'),
               'native_loss_margin':b.get('native_loss')}
        if aa is not None and bb is not None:
            for field in KEYS:
                av,bv=aa.get(field),bb.get(field)
                row[field+'_v2']=av;row[field+'_margin']=bv
                row[field+'_margin_minus_v2']=(bv-av if isinstance(av,(int,float))
                                                and isinstance(bv,(int,float)) else None)
        out.append(row)
    summary={}
    for field in KEYS:
        aval=[r[field+'_v2'] for r in out if isinstance(r.get(field+'_v2'),(int,float))]
        bval=[r[field+'_margin'] for r in out if isinstance(r.get(field+'_margin'),(int,float))]
        if aval and bval:
            summary[field]={'n':len(aval),'v2_mean':mean(aval),'margin_mean':mean(bval),
                'v2_median':median(aval),'margin_median':median(bval),
                'margin_minus_v2_mean':mean(bval)-mean(aval),
                'last_v2':aval[-1],'last_margin':bval[-1]}
    return {'common_logged_steps':keys,'timeseries':out,'summary':summary}


def compare_geom(a,b):
    if len(a['batches']) != len(b['batches']):
        raise ValueError('Geometry runs contain different batch counts')
    if a.get('seed')!=b.get('seed') or a.get('mosaic')!=b.get('mosaic'):
        raise ValueError('Geometry reports must use identical seed and mosaic settings')
    for aa,bb in zip(a['batches'],b['batches']):
        if aa['batch']!=bb['batch']:
            raise ValueError('Geometry batch IDs mismatch')
        for view in ('clean','perturbed'):
            for key in ('fg_anchors','semantic_bg_anchors','total_anchors'):
                if aa[view]['counts'][key]!=bb[view]['counts'][key]:
                    raise ValueError(f'Geometry populations mismatch: batch={aa["batch"]} {view} {key}')
    out={}
    for view in ('clean','perturbed'):
        aa=a['summary'][view];bb=b['summary'][view]
        out[view]={'v2':{},'margin':{},'margin_minus_v2':{},'by_size':{},'by_level':{}}
        for key in ('fg_ungated_mean_bce_gain','fg_gated_mean_bce_gain',
                    'fg_mean_gt_class_delta','fg_mean_margin_improvement',
                    'per_batch_gate_auc_mean_1e-3','all_anchor_abs_delta_mean','all_anchor_gate_mean'):
            out[view]['v2'][key]=aa.get(key);out[view]['margin'][key]=bb.get(key)
            va,vb=aa.get(key),bb.get(key)
            out[view]['margin_minus_v2'][key]=vb-va if va is not None and vb is not None else None
        for group, dest in (('by_size','by_size'),('by_level','by_level')):
            for k in SIZE if group=='by_size' else LEVEL:
                x,y=aa[group][k],bb[group][k]
                if x['anchors'] != y['anchors']:
                    raise ValueError(f'Group anchor count mismatch: {view}/{group}/{k}')
                out[view][dest][k]={'anchors':x['anchors']}
                for field in ('ungated_mean_bce_gain','gated_mean_bce_gain','mean_gt_class_delta'):
                    av,bv=x[field],y[field]
                    out[view][dest][k][field+'_v2']=av
                    out[view][dest][k][field+'_margin']=bv
                    out[view][dest][k][field+'_difference']=bv-av if av is not None and bv is not None else None
        for label, data in (('v2',a),('margin',b)):
            batch_views=[r[view] for r in data['batches']]
            result = out[view][label]
            result['embedding_r_minus_z_norm_by_level_mean']=[
                mean([r['embedding_r_minus_z_norm_by_level'][i]['mean'] for r in batch_views])
                for i in range(3)]
            result['logit_gain_by_level_mean']=[
                mean([r['logit_gain_by_level'][i] for r in batch_views])
                for i in range(3)]
            result['negative_bg_ungated_gain_weighted'] = weighted(
                [r['background'].get('ungated_bce_gain',{}).get('mean') for r in batch_views],
                [r['counts']['semantic_bg_anchors'] for r in batch_views])
            result['negative_bg_gated_gain_weighted'] = weighted(
                [r['background'].get('gated_bce_gain',{}).get('mean') for r in batch_views],
                [r['counts']['semantic_bg_anchors'] for r in batch_views])
            result['foreground_utility_counts']={t:sum(
                int(r['foreground']['utility_threshold_sensitivity']['0.001'][t])
                for r in batch_views) for t in ('useful','harmful')}
    return out


def load_bounded_run(folder):
    """Load the *raw* bounded-pilot format, not compare_pgsma_pilots' summary.

    result.json stores run metadata, hashes and update counts; evaluation
    metrics live under clean_val_step_<steps>.json -> metrics -> mode.
    """
    folder = Path(folder)
    raw = read_json(folder / 'result.json')
    if raw.get('status') != 'CHECKS_PASSED':
        raise ValueError(f'{folder}: incomplete/blocked bounded pilot')
    steps = int(raw['steps_requested'])
    final = read_json(folder / f'clean_val_step_{steps:05d}.json')
    if int(final.get('step', -1)) != steps:
        raise ValueError(f'{folder}: final eval step does not match result.json')
    if final.get('variant') != raw.get('variant'):
        raise ValueError(f'{folder}: variant mismatch between result and final evaluation')
    mode = 'pgsma' if raw['variant'] in ('pgsma', 'pgsma_simota') else 'native'
    modes = final.get('metrics', {})
    if mode not in modes or 'native' not in modes:
        raise ValueError(f'{folder}: missing {mode}/native evaluation modes')
    selected = modes[mode].get('coco_metrics')
    native = modes['native'].get('coco_metrics')
    if not isinstance(selected, dict) or not isinstance(native, dict):
        raise ValueError(f'{folder}: missing COCO metrics (possibly no detections)')
    for key in COCO:
        if key not in selected or key not in native:
            raise ValueError(f'{folder}: missing COCO metric {key}')
    return raw, native, selected, final


def check_matched_run_metadata(a, b):
    """Reject misleading causal comparisons, including different val samples."""
    for key in ('native_state_sha256', 'training_batch_stream_sha256',
                'first_training_batch_sha256', 'seed', 'steps_requested',
                'input_size', 'test_size', 'class_count', 'eval_indices',
                'clean_val_json_sha256', 'native_recipe', 'native_lr',
                'mosaic_enabled', 'mosaic_prob'):
        if key not in a or key not in b:
            raise ValueError(f'Paired experiment missing required field {key}')
        if a[key] != b[key]:
            raise ValueError(f'Paired experiment is NOT comparable: {key} differs')
    source_a = (a.get('checkpoint') or {}).get('sha256')
    source_b = (b.get('checkpoint') or {}).get('sha256')
    if not source_a or source_a != source_b:
        raise ValueError('Paired experiment is NOT comparable: warm-start checkpoint differs')
    for key in ('successful_native_updates', 'successful_semantic_updates'):
        if a.get(key) != b.get(key):
            raise ValueError(f'Paired experiment is NOT comparable: {key} differs')
    if a.get('variant') != 'pgsma_simota' or b.get('variant') != 'pgsma_simota':
        raise ValueError('Both runs must use pgsma_simota')
    if a.get('candidate_objective') != 'v2' or b.get('candidate_objective') != 'margin_probe':
        raise ValueError('Expected objective sequence: v2, margin_probe')


def compare(v2_dir, margin_dir, v2_geometry=None, margin_geometry=None):
    a, a_native, a_selected, a_eval = load_bounded_run(v2_dir)
    b, b_native, b_selected, b_eval = load_bounded_run(margin_dir)
    check_matched_run_metadata(a, b)
    if a_eval['metrics']['native'].get('image_ids') != b_eval['metrics']['native'].get('image_ids'):
        raise ValueError('Same subset indices but evaluation image IDs differ')
    for key in COCO:
        av, bv = a_native[key], b_native[key]
        if not math.isclose(av, bv, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Same-state native inference differs for {key}')
    a_training = read_metrics(Path(v2_dir) / 'train_metrics.jsonl')
    b_training = read_metrics(Path(margin_dir) / 'train_metrics.jsonl')
    if set(a_training) != set(b_training):
        raise ValueError('Different recorded training steps; do not silently compare intersection')
    out = {'status': 'PAIRED_SOURCE_AUDIT',
           'scope': 'Matched native state, sampled clean validation + logged SOURCE training batches only',
           'causal_contract': {'native_state_sha256': a['native_state_sha256'],
                               'training_batch_stream_sha256': a['training_batch_stream_sha256'],
                               'native_eval_equal': True,
                               'eval_images': len(a['eval_indices']),
                               'seed': a['seed']},
           'performance': {},
           'training': train_path(a_training, b_training)}
    for key in COCO:
        base = a_native[key]
        av, bv = a_selected[key], b_selected[key]
        out['performance'][key] = {'native': base, 'v2': av, 'margin': bv,
                                   'v2_minus_native': av - base,
                                   'margin_minus_native': bv - base,
                                   'margin_minus_v2': bv - av}
    if bool(v2_geometry) != bool(margin_geometry):
        raise ValueError('Supply both geometry reports or neither')
    if v2_geometry:
        out['geometry'] = compare_geom(read_json(v2_geometry), read_json(margin_geometry))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2-dir',required=True)
    p.add_argument('--margin-dir',required=True)
    p.add_argument('--v2-geometry')
    p.add_argument('--margin-geometry')
    p.add_argument('--json',required=True)
    args=p.parse_args()
    try:
        result=compare(args.v2_dir,args.margin_dir,args.v2_geometry,args.margin_geometry)
    except (ValueError,KeyError,FileNotFoundError) as e:
        p.error(str(e))
    output=Path(args.json);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print('PAIRED SOURCE AUDIT — native state and training stream matched')
    print('Metric                  v2-native     margin-native   margin-v2')
    for key,record in result['performance'].items():
        print(f'{key:<23} {record["v2_minus_native"]:+.8f}   '
              f'{record["margin_minus_native"]:+.8f}   {record["margin_minus_v2"]:+.8f}')
    print('TRAINING TELEMETRY (means across matched, sampled logged steps; not GT-level confidence intervals)')
    for key,record in result['training']['summary'].items():
        print(f'  {key:<33} v2={record["v2_mean"]:+.6f} margin={record["margin_mean"]:+.6f}')
    if 'geometry' in result:
        for view,record in result['geometry'].items():
            print(view.upper(),'geometry:')
            for tag in ('v2','margin'):
                g=record[tag]
                print(f'  {tag}: fg_BCE={g["fg_ungated_mean_bce_gain"]:+.6f} '
                      f'fg_margin={g["fg_mean_margin_improvement"]:+.6f} '
                      f'bg_BCE={g["negative_bg_ungated_gain_weighted"]:+.6f} '
                      f'gateAUC={g["per_batch_gate_auc_mean_1e-3"]}')
            print('  by size (ungated BCE improvement, margin-v2):')
            for s,v in record['by_size'].items():
                print(f'    {s:<7} N={v["anchors"]:<5} {v["ungated_mean_bce_gain_difference"]:+.7f}')
    print('Saved:',output)

if __name__ == '__main__':
    main()
