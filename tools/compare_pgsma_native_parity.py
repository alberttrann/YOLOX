"""Forensic first-divergence comparator for --trace-parity pilot outputs.

Does not require torch or load checkpoints. It verifies native source isolation,
not whether PG-SMA improves detection. No authorization for full training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def read_trace(path):
    result={}
    for i,raw in enumerate(Path(path).read_text(encoding='utf8').splitlines(),1):
        if not raw.strip():
            continue
        row=json.loads(raw)
        step=row['step']
        if step in result:
            raise ValueError(f'Duplicate step {step} at line {i} in {path}')
        result[step]=row
    if not result or 0 not in result:
        raise ValueError(f'Empty/incomplete trace: {path}')
    return result


def flatten(row):
    d={}
    for key,value in row.items():
        if key in ('step','phase'):
            continue
        if isinstance(value,dict):
            for sub,val in value.items():
                d[f'{key}.{sub}']=val
        else:
            d[key]=value
    return d


def compare(left,right):
    a=read_trace(left);b=read_trace(right)
    common=sorted(set(a)&set(b))
    if set(a)!=set(b):
        raise ValueError(f'Trace steps differ: left={sorted(a)}, right={sorted(b)}')
    first=None; all_diff=[]
    shared_keys=('full_input_sha256','model.all','model.backbone','model.head','model.other',
                 'native_pre_step.all','native_pre_step.backbone',
                 'native_pre_step.head','native_pre_step.other',
                 'native_grad_before_clip','native_grad_after_clip',
                 'native_post_step.all','native_post_step.backbone',
                 'native_post_step.head','native_post_step.other',
                 'native_optimizer_post_step','optimizer','native_loss','iou_loss',
                 'cls_loss','conf_loss','rng','rng_after_step')
    for step in common:
        la=flatten(a[step]);rb=flatten(b[step]); changed={}
        for k in shared_keys:
            if k in la and k in rb and la[k]!=rb[k]:
                changed[k]={'left':la[k],'right':rb[k]}
        if changed:
            item={'step':step,'changed':changed}
            all_diff.append(item)
            if first is None:first=item
    return {
        'status':'PARITY_MATCH' if not all_diff else 'PARITY_DIVERGED',
        'common_steps':common,
        'first_divergence':first,
        'divergent_steps':[x['step'] for x in all_diff],
        'native_state_final_equal': (
            flatten(a[common[-1]]).get('native_post_step.all')==
            flatten(b[common[-1]]).get('native_post_step.all')
            if common[-1]>0 else
            flatten(a[0]).get('model.all')==flatten(b[0]).get('model.all')),
        'semantic_rng_consumption':{
            'left':[s for s,r in a.items() if r.get('semantic_rng_changed')],
            'right':[s for s,r in b.items() if r.get('semantic_rng_changed')],
        },
        'interpretation': (
            'A native mismatch invalidates cross-run attribution. Compare earliest changed '
            'loss, gradient, state, optimizer, RNG fields to localize the discrepancy. '
            'A hash mismatch alone does not prove the semantic branch overwrote native weights.'
            if first else
            'Native-state parity holds at traced points. Validate full per-step trace '
            'and compare native-disabled versus semantic-enabled inference on the same model.'
        ),
        'full_training_authorized':False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('native_trace')
    p.add_argument('semantic_trace')
    p.add_argument('--json',default=None)
    args=p.parse_args()
    try:
        result=compare(args.native_trace,args.semantic_trace)
    except Exception as error:
        result={'status':'BLOCKED','error':str(error)}
    output=json.dumps(result,indent=2,ensure_ascii=False)
    print(output)
    if args.json:Path(args.json).write_text(output+'\n',encoding='utf8')
    return 0 if result['status']=='PARITY_MATCH' else 2


if __name__=='__main__':
    raise SystemExit(main())
