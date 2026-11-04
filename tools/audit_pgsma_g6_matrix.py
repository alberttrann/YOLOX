"""Offline, read-only 2x2 Pilot G6 experiment comparison.

Reads raw pgsma_bounded_pilot results and optionally matched G3 geometry reports.
Fails closed on mismatched native states, data streams, evaluation subsets or steps.
No model is imported or trained. Does not authorize full 80-epoch training.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

ARM = {
    'margin': 'margin_probe',
    'bounded': 'bounded_probe',
    'margin_bg': 'margin_bg_probe',
    'bounded_bg': 'bounded_bg_probe',
}
COCO = ('AP50_95','AP50','AP75','AR100','AP_small_COCO','AP_medium_COCO','AP_large_COCO')
SIZE = ('lt4','4to8','8to16','16to32','ge32')
LEVEL = ('P3','P4','P5')
UTILITY = ('candidate_fg_improvement','candidate_bg_improvement',
           'gate_fg_improvement','gate_bg_improvement',
           'mean_abs_candidate_delta','mean_gate_acceptance')
PAIRED_KEYS = ('variant','seed','steps_requested','input_size','test_size','class_count',
               'native_lr','semantic_lr','mosaic_enabled','mosaic_prob',
               'eval_indices','clean_val_json_sha256','native_recipe','checkpoint',
               'native_state_sha256','training_batch_stream_sha256',
               'first_training_batch_sha256','successful_native_updates',
               'successful_semantic_updates')
TRACE_KEYS = ('full_input_sha256','model.all','model.backbone','model.head','model.other',
              'native_pre_step.all','native_pre_step.backbone','native_pre_step.head','native_pre_step.other',
              'native_grad_before_clip','native_grad_after_clip',
              'native_post_step.all','native_post_step.backbone','native_post_step.head',
              'native_post_step.other','native_optimizer_post_step','optimizer',
              'native_loss','iou_loss','cls_loss','conf_loss','rng','rng_after_step')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    result={}
    for i, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        rec=json.loads(raw)
        k=int(rec['step'])
        if k in result:
            raise ValueError(f'duplicate step {k} in {path}, line {i}')
        result[k]=rec
    if not result:
        raise ValueError(f'empty {path}')
    return result


def ensure_number(x, context):
    if not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x):
        raise ValueError(f'nonfinite or missing {context}: {x!r}')
    return float(x)


def load(folder, objective):
    folder=Path(folder)
    meta=read_json(folder/'result.json')
    if meta.get('status')!='CHECKS_PASSED':
        raise ValueError(f'{folder}: pilot did not finish successfully')
    if meta.get('variant')!='pgsma_simota':
        raise ValueError(f'{folder}: expected pgsma_simota')
    if meta.get('candidate_objective')!=objective:
        raise ValueError(f'{folder}: expected {objective}, got {meta.get("candidate_objective")}')
    step=int(meta['steps_requested'])
    if step <= 0 or meta.get('successful_native_updates')!=step or meta.get('successful_semantic_updates')!=step:
        raise ValueError(f'{folder}: missing native/semantic updates')
    ev=read_json(folder/f'clean_val_step_{step:05d}.json')
    if int(ev.get('step',-1))!=step or ev.get('variant')!=meta['variant']:
        raise ValueError(f'{folder}: final eval step/variant mismatch')
    m=ev.get('metrics',{})
    if not all(mode in m for mode in ('native','pgsma')):
        raise ValueError(f'{folder}: native/pgsma inference not present')
    if m['native'].get('image_ids')!=m['pgsma'].get('image_ids'):
        raise ValueError(f'{folder}: mismatched native/semantic evaluation image IDs')
    if len(m['native'].get('image_ids',[]))!=len(meta['eval_indices']):
        raise ValueError(f'{folder}: image ID count mismatch')
    if len(set(m['native']['image_ids']))!=len(m['native']['image_ids']):
        raise ValueError(f'{folder}: duplicated eval image ID')
    for mode in ('native','pgsma'):
        scores=m[mode].get('coco_metrics')
        if not isinstance(scores,dict):
            raise ValueError(f'{folder}: missing {mode} COCO metrics')
        for key in COCO:
            ensure_number(scores.get(key),f'{folder}/{mode}/{key}')
        bins=m[mode].get('post_nms_recall_iou50_by_effective_min_side')
        if not isinstance(bins,dict):
            raise ValueError(f'{folder}: missing {mode} effective-size recall')
        for k in SIZE:
            if k not in bins or not isinstance(bins[k].get('total_gt'),int):
                raise ValueError(f'{folder}: missing {mode}/{k} GT counts')
    logs=read_jsonl(folder/'train_metrics.jsonl')
    trace_path=folder/'native_parity_trace.jsonl'
    if not trace_path.is_file():
        raise ValueError(f'{folder}: missing native parity trace; was --trace-parity enabled?')
    traces=read_jsonl(trace_path)
    if 0 not in traces or step not in traces:
        raise ValueError(f'{folder}: trace missing step 0/final step')
    if traces[0].get('strict_determinism') is not True:
        raise ValueError(f'{folder}: strict determinism not enabled in trace')
    if 'all' not in traces[0].get('model',{}):
        raise ValueError(f'{folder}: missing initial native model trace')
    if 'all' not in traces[step].get('native_post_step',{}):
        raise ValueError(f'{folder}: missing final native model trace')
    return {'dir':str(folder),'meta':meta,'eval':ev,'logs':logs,'traces':traces}


def flatten(d):
    out={}
    for k,v in d.items():
        if k in ('step','phase'):
            continue
        if isinstance(v,dict):
            out.update({k+'.'+sub:value for sub,value in v.items()})
        else:
            out[k]=v
    return out


def paired_contract(runs):
    base=runs['margin']
    for arm,run in runs.items():
        a,b=base['meta'],run['meta']
        for key in PAIRED_KEYS:
            if key not in a or key not in b or a[key]!=b[key]:
                raise ValueError(f'NOT PAIRED: {arm} differs from margin in {key}')
        ae,be=base['eval']['metrics'],run['eval']['metrics']
        for mode in ('native','pgsma'):
            if ae[mode].get('image_ids')!=be[mode].get('image_ids'):
                raise ValueError(f'NOT PAIRED: {arm}/{mode} evaluation image IDs differ')
        for key in COCO:
            va=ensure_number(ae['native']['coco_metrics'][key],key)
            vb=ensure_number(be['native']['coco_metrics'][key],key)
            if not math.isclose(va,vb,rel_tol=0,abs_tol=1e-12):
                raise ValueError(f'NOT PAIRED: native eval {key} differs for {arm}')
        for k in SIZE:
            if ae['native']['post_nms_recall_iou50_by_effective_min_side'][k] != be['native']['post_nms_recall_iou50_by_effective_min_side'][k]:
                raise ValueError(f'NOT PAIRED: native effective-size recall differs for {arm}/{k}')
        if set(base['logs'])!=set(run['logs']):
            raise ValueError(f'NOT PAIRED: logged train step sets differ for {arm}')
        if set(base['traces'])!=set(run['traces']):
            raise ValueError(f'NOT PAIRED: parity trace step sets differ for {arm}')
        for step in base['traces']:
            al=flatten(base['traces'][step]);bl=flatten(run['traces'][step])
            for key in TRACE_KEYS:
                if (key in al)!=(key in bl):
                    raise ValueError(f'NOT PAIRED: trace field {key} absent for one arm at step {step}')
                if key in al and al[key]!=bl[key]:
                    raise ValueError(f'NATIVE DIVERGENCE: {arm} trace step {step} field {key}')
            if run['traces'][step].get('semantic_rng_changed'):
                raise ValueError(f'GLOBAL RNG CONSUMPTION: {arm} step {step}')
    return {'status':'PASS_AT_TRACED_STEPS',
            'native_sha256':base['meta']['native_state_sha256'],
            'stream_sha256':base['meta']['training_batch_stream_sha256'],
            'checkpoint_sha256':base['meta']['checkpoint']['sha256'],
            'eval_images':len(base['meta']['eval_indices']),
            'trace_steps':sorted(base['traces']),
            'log_steps':sorted(base['logs'])}


def metric_report(runs):
    out={}
    base=runs['margin']['eval']['metrics']['pgsma']['coco_metrics']
    for arm,run in runs.items():
        ev=run['eval']['metrics']
        out[arm]={}
        for k in COCO:
            native=float(ev['native']['coco_metrics'][k]);pg=float(ev['pgsma']['coco_metrics'][k])
            out[arm][k]={'native':native,'corrected':pg,
                          'correction_minus_native':pg-native,
                          'corrected_minus_margin':pg-float(base[k])}
        size={}
        for k in SIZE:
            n=ev['native']['post_nms_recall_iou50_by_effective_min_side'][k]
            p=ev['pgsma']['post_nms_recall_iou50_by_effective_min_side'][k]
            if n['total_gt']!=p['total_gt']:
                raise ValueError(f'{arm}/{k}: ground truth counts mismatch')
            nr=n.get('recall_iou50');pr=p.get('recall_iou50')
            size[k]={'total_gt':n['total_gt'],'native':nr,'corrected':pr,
                     'delta':float(pr)-float(nr) if nr is not None and pr is not None else None,
                     'native_matched':n.get('matched'),'corrected_matched':p.get('matched')}
        out[arm]['effective_size_recall_iou50']=size
    return out


def training_report(runs):
    out={}
    for arm,run in runs.items():
        stats={}
        for key in UTILITY:
            vals=[]
            for step,row in run['logs'].items():
                util=row.get('semantic_utility') or {}
                if key in util and util[key] is not None:
                    vals.append((step,ensure_number(util[key],f'{arm}/{step}/{key}')))
            stats[key]={'n_logged':len(vals),
                        'mean':statistics.mean(v for _,v in vals) if vals else None,
                        'last_step':vals[-1][0] if vals else None,
                        'last':vals[-1][1] if vals else None}
        out[arm]=stats
    return out


def geom_report(runs, geometry):
    if not geometry:
        return None
    if set(geometry)!=set(ARM):
        raise ValueError('Provide --geometry for all four arms or for none')
    reports={name:read_json(path) for name,path in geometry.items()}
    base=reports['margin']
    for arm,report in reports.items():
        if report.get('status')!='CHECKS_PASSED':
            raise ValueError(f'{arm}: geometry incomplete')
        if report.get('seed')!=base.get('seed') or report.get('mosaic')!=base.get('mosaic'):
            raise ValueError(f'{arm}: geometry seed/mosaic differs')
        if len(report['batches'])!=len(base['batches']):
            raise ValueError(f'{arm}: geometry batch count differs')
        for a,b in zip(base['batches'],report['batches']):
            if a['batch']!=b['batch']:
                raise ValueError(f'{arm}: geometry batch order differs')
            for view in ('clean','perturbed'):
                for k in ('fg_anchors','semantic_bg_anchors','total_anchors'):
                    if a[view]['counts'][k]!=b[view]['counts'][k]:
                        raise ValueError(f'{arm}: geometry populations differ at batch {a["batch"]}/{view}/{k}')
    result={}
    for view in ('clean','perturbed'):
        result[view]={}
        for arm,report in reports.items():
            s=report['summary'][view]
            batches=[r[view] for r in report['batches']]
            def bg_weighted(mode):
                xs=[(r['background'].get(mode+'_bce_gain',{}).get('mean'),r['counts']['semantic_bg_anchors']) for r in batches]
                xs=[(float(x),int(n)) for x,n in xs if x is not None and n>0]
                return sum(x*n for x,n in xs)/sum(n for _,n in xs) if xs else None
            def utility_count(label):
                return sum(int(r['foreground']['utility_threshold_sensitivity']['0.001'].get(label,0)) for r in batches)
            result[view][arm]={
                'foreground_ungated_gain':s['fg_ungated_mean_bce_gain'],
                'foreground_gated_gain':s['fg_gated_mean_bce_gain'],
                'bg_ungated_gain':bg_weighted('ungated'),
                'bg_gated_gain':bg_weighted('gated'),
                'gt_class_delta':s['fg_mean_gt_class_delta'],
                'margin_delta':s['fg_mean_margin_improvement'],
                'gate_auc_mean_per_batch':s['per_batch_gate_auc_mean_1e-3'],
                'abs_delta_mean':s['all_anchor_abs_delta_mean'],
                'gate_mean':s['all_anchor_gate_mean'],
                'useful_fg':utility_count('useful'),
                'harmful_fg':utility_count('harmful'),
                'by_size':s['by_size'],
                'by_level':s['by_level'],
            }
    return result


def compare(arm_paths,geometry=None):
    if set(arm_paths)!=set(ARM):
        raise ValueError('Exactly four arms required')
    runs={arm:load(arm_paths[arm],objective) for arm,objective in ARM.items()}
    parity=paired_contract(runs)
    return {'status':'PAIRED_SOURCE_DIAGNOSTIC',
            'native_parity':parity,
            'performance':metric_report(runs),
            'training':training_report(runs),
            'geometry':geom_report(runs,geometry),
            'qualification':'PILOT_DIAGNOSTICS_ONLY_NOT_FINAL_MODEL_SELECTION',
            'full_training_authorized':False,
            'limitations':['128 sampled clean source validation images, not full validation',
                           'metric deltas are not confidence intervals or cross-seed replication',
                           'logged batch averages are not GT-level statistics',
                           'even validated sampled hard negatives do not establish universal background safety',
                           'native trace parity only at recorded steps']}


def fmt(x):
    return 'N/A' if x is None else f'{x:+.8f}'


def print_report(report):
    print('PILOT G6 2x2 FACTORIAL — PAIRED CLEAN SOURCE DIAGNOSTIC')
    print('Parity:',report['native_parity']['status'], '| eval images:', report['native_parity']['eval_images'])
    print('Arm                 dAP50:95      dAP50         dAP75         dAR100       dSmall        dMedium       dLarge')
    for arm in ARM:
        p=report['performance'][arm]
        print(f'{arm:<19}'+ ' '.join(f'{fmt(p[k]["correction_minus_native"]):>12}' for k in COCO))
    print('TRAINING: logged-step means; positive improvement = lower BCE')
    print('Arm                  FG candidate    BG candidate    FG gated      BG gated       |delta|       gate')
    for arm in ARM:
        t=report['training'][arm]
        print(f'{arm:<20}'+ ' '.join(f'{fmt(t[k]["mean"]):>12}' for k in UTILITY))
    print('EFFECTIVE-SIZE POST-NMS RECALL @ IOU .50 — Δ vs native; NOT COCO AR')
    for arm in ARM:
        sizes=report['performance'][arm]['effective_size_recall_iou50']
        print(f'{arm:<19}'+ ' '.join(f'{k}={fmt(sizes[k]["delta"])} (N={sizes[k]["total_gt"]})' for k in SIZE))
    if report['geometry']:
        for view,byarm in report['geometry'].items():
            print(view.upper(),'GEOMETRY (sampled training features):')
            for arm in ARM:
                s=byarm[arm]
                print(f'  {arm:<15} fg={fmt(s["foreground_ungated_gain"])} bg={fmt(s["bg_ungated_gain"])} '
                      f'P3={fmt(s["by_level"]["P3"]["ungated_mean_bce_gain"])} '
                      f'4to8={fmt(s["by_size"]["4to8"]["ungated_mean_bce_gain"])} '
                      f'gateAUC={s["gate_auc_mean_per_batch"]:.3f}')
    print('Decision: no full 80-epoch authorization; require held-out confirmation, more seeds, adverse and UAV evaluations.')


def parse_arm_paths(values,kind):
    output={}
    for value in values:
        if '=' not in value:
            raise ValueError(f'{kind} expects arm=path, got {value!r}')
        key,path=value.split('=',1)
        if key not in ARM or not path or key in output:
            raise ValueError(f'{kind}: unknown/duplicate arm or empty path: {value!r}')
        output[key]=path
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='Repeat 4 times: margin=path, bounded=path, margin_bg=path, bounded_bg=path')
    p.add_argument('--geometry',action='append',default=[],help='Optional: provide ALL 4 matched G3 geometry paths')
    p.add_argument('--json',required=True)
    args=p.parse_args()
    try:
        report=compare(parse_arm_paths(args.run,'--run'),parse_arm_paths(args.geometry,'--geometry'))
    except (ValueError,KeyError,TypeError,FileNotFoundError,ZeroDivisionError) as e:
        p.error(str(e))
    target=Path(args.json);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print_report(report)
    print('Saved:',target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
