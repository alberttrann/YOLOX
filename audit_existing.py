#!/usr/bin/env python3
"""Read-only, no-GPU audit of already generated tdevNext.3 A4 results.

No PyTorch, pycocotools, dataset decoding, model loading, or bootstrap is used.
It scans P0 metadata/detection files and completed bootstrap outputs, checks
policy consistency, and builds a report using the metrics ALREADY computed.
"""
from __future__ import annotations
import argparse
import ast
import collections
import csv
import json
import re
from pathlib import Path

MODELS = ('A0','A1','A2','A3','A4','A4R')
DOMAINS = ('clean','fog','rain','snow','night')
EXPECTED_BRANCHES = {
    'A0':('on',), 'A1':('on',), 'A2':('on','fourier-off'),
    'A3':('on','memory-off'),
    'A4':('on','fourier-off','memory-off','both-off'), 'A4R':('on',)
}
METRIC = re.compile(r'Average\s+(Precision|Recall)\s+\((AP|AR)\)\s*@\[\s*IoU\s*=\s*([^|]+)\|\s*area\s*=\s*([^|]+)\|\s*maxDets\s*=\s*(\d+)\s*\]\s*=\s*(-?\d+(?:\.\d+)?)')

def parse_coco_summary(s):
    ans={}
    for kind, _short, threshold, area, max_dets, value in METRIC.findall(str(s)):
        name='AP' if kind=='Precision' else 'AR'
        threshold=threshold.strip();area=area.strip();max_dets=int(max_dets)
        key=None
        if name=='AP':
            if area=='all' and threshold=='0.50:0.95':key='AP'
            elif area=='all' and threshold=='0.50':key='AP50'
            elif area=='all' and threshold=='0.75':key='AP75'
            elif area in ('small','medium','large') and threshold=='0.50:0.95':key='AP_'+area
        elif name=='AR' and threshold=='0.50:0.95':
            if area=='all' and max_dets==100:key='AR100'
            elif area in ('small','medium','large') and max_dets==100:key='AR100_'+area
        if key:ans[key]=float(value)
    return ans

def source_audit(repo:Path):
    path=repo/'exps/default/tde_yolox_s.py'
    out={'path':str(path),'status':'unavailable'}
    if not path.is_file():return out
    src=path.read_text(encoding='utf-8')
    try: tree=ast.parse(src)
    except SyntaxError as exc:return {'path':str(path),'status':'cannot_parse','error':str(exc)}
    defs=[node for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='get_data_loader']
    if not defs:return {'path':str(path),'status':'no_local_override'}
    node=defs[-1]
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else 'unknown'
           for n in ast.walk(node) if isinstance(n,ast.Call)}
    return {'path':str(path),'status':'inspected','mosaic_detection_constructed':('MosaicDetection' in calls),
            'batch_sampler_constructed':('YoloBatchSampler' in calls),
            'get_dataset_called':('get_dataset' in calls),
            'assessment':('POSSIBLE_NO_MOSAIC_WRAPPER' if 'YoloBatchSampler' in calls and 'MosaicDetection' not in calls else 'REVIEW_LOADER_PATH'),
            'note':'Static detection only. Verify actual runtime dataset wrapper if custom factories or monkeypatches exist.'}

def dump_csv(path,rows):
    if not rows:return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def scan(repo:Path,output:Path):
    base=repo/'YOLOX_outputs/a4_diagnostics'
    rows=[];missing=[];bad=[];signatures=collections.defaultdict(list);perpair=[]
    for model in MODELS:
      for domain in DOMAINS:
       for branch in EXPECTED_BRANCHES[model]:
        directory=base/'p0'/model/domain/branch
        m=directory/'metadata.json';det=directory/'detections.json'
        if not m.is_file() or not det.is_file():
            missing.append(f'{model}/{domain}/{branch}');continue
        try:
            x=json.loads(m.read_text(encoding='utf-8'))
            # Do not parse giant detections.json files: the metrics were already
            # scored and are stored in metadata. Avoid hours of redundant I/O.
            if det.stat().st_size < 2:raise ValueError('Detection JSON is empty/truncated')
            if x.get('branch') != branch:raise ValueError('Metadata branch mismatch')
            x_metrics=parse_coco_summary(x.get('summary',''))
            row={'model':model,'domain':domain,'branch':branch,
                 'checkpoint_sha256':x.get('checkpoint_sha256'),
                 'precision_mode':x.get('precision_mode','unspecified'),
                 'data_ann':x.get('data_ann'),
                 'weights':x.get('weights'),
                 'prediction_count':x.get('prediction_count'),**x_metrics}
            if not x_metrics:bad.append(f'{model}/{domain}/{branch}: metrics absent from metadata summary')
            rows.append(row)
            sig=(x.get('data_ann'),json.dumps(x.get('categories'),sort_keys=True),str(x.get('input_size')),x.get('conf'),x.get('nms'),x.get('weights'),x.get('precision_mode'))
            signatures[domain].append((model,branch,sig))
        except (OSError,ValueError,TypeError) as exc:
            bad.append(f'{model}/{domain}/{branch}: {type(exc).__name__}: {exc}')
    outlookup={(r['model'],r['domain'],r['branch']):r for r in rows}
    comps=(('A2','on','A2','fourier-off'),('A3','on','A3','memory-off'),
           ('A4','on','A4','fourier-off'),('A4','on','A4','memory-off'),
           ('A4','on','A4','both-off'),('A2','on','A1','on'),('A4','on','A4R','on'))
    for d in DOMAINS:
      for am,ab,bm,bb in comps:
        left=outlookup.get((am,d,ab));right=outlookup.get((bm,d,bb))
        if not left or not right:continue
        if left['precision_mode']!=right['precision_mode'] or left['data_ann']!=right['data_ann']:
            bad.append(f'UNPAIRED {d}: {am}/{ab} vs {bm}/{bb} precision/data differ');continue
        row={'domain':d,'left':f'{am}/{ab}','right':f'{bm}/{bb}'}
        for key in ('AP','AP50','AP75','AR100','AP_small','AR100_small','AP_medium','AP_large'):
            if key in left and key in right:row['delta_'+key]=left[key]-right[key]
        perpair.append(row)
    for d,entries in signatures.items():
        signatures_set={str(e[2]) for e in entries}
        if len(signatures_set)>1:
            bad.append(f'{d}: inconsistent P0 input/protocol signatures ({len(signatures_set)} variants)')
    boot=[]
    bp=base/'bootstrap'
    if bp.exists():
      for path in sorted(bp.rglob('*.json')):
        try:
            obj=json.loads(path.read_text(encoding='utf-8'))
            for metric,values in obj.get('metrics',{}).items():
                ci=values.get('ci95',[None,None]);ci=(ci+[None,None])[:2]
                boot.append({'path':str(path.relative_to(base)).replace('\\','/'),
                             'metric':metric,'delta_mean':values.get('delta_mean'),
                             'ci95_lo':ci[0],'ci95_hi':ci[1],
                             'bootstrap_positive_fraction':values.get('prob_positive'),
                             'iterations':obj.get('iterations')})
        except Exception as exc:bad.append(f'{path.name}: bootstrap parse error {exc}')
    manifest={}
    for model,name in (('A4','tdevnext3_a4_fourier_grounded'),('A4R','tdevnext3_a4_orthogonal_control')):
        path=repo/'YOLOX_outputs'/name/'a4_provenance.json'
        if path.exists():
            rec=json.loads(path.read_text(encoding='utf-8')); recorded=set(rec.get('source_sha256',{}))
            relevant=['exps/default/tde_yolox_s.py','yolox/models/fourier_invariance.py',
                      'yolox/models/fourier_backbone.py','yolox/models/yolo_pafpn.py',
                      'yolox/core/trainer.py','tools/train_fourier_a4.py']
            manifest[model]={'manifest_found':True,'basis':rec.get('filter_basis'),
                'source_files_recorded':len(recorded),'important_unhashed_files':[q for q in relevant if q not in recorded]}
        else:manifest[model]={'manifest_found':False}
    output.mkdir(parents=True,exist_ok=True)
    dump_csv(output/'existing_p0_metrics.csv',rows)
    dump_csv(output/'existing_p0_deltas.csv',perpair)
    dump_csv(output/'existing_bootstrap_index.csv',boot)
    result={'source_data_loader':source_audit(repo),'p0_branches_found':len(rows),
            'p0_branches_expected':sum(len(EXPECTED_BRANCHES[m]) for m in MODELS)*len(DOMAINS),
            'p0_missing':missing,'diagnostic_warnings':bad,
            'bootstrap_files_found':len({r['path'] for r in boot}),
            'manifest_coverage':manifest}
    (output/'integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# tdevNext.3: read-only local evidence snapshot','',
           f'P0 branches present: **{len(rows)}/{result["p0_branches_expected"]}**. '
           f'Completed bootstrap files: **{result["bootstrap_files_found"]}**.',
           '', '## Training loader', '',f'`{result["source_data_loader"]}`','',
           '## Already computed point-estimate contrasts','',
           '|Domain|Left|Right|ΔAP (pp)|ΔAR100 (pp)|Δsmall AP (pp)|Δlarge AP (pp)|',
           '|---|---|---|---:|---:|---:|---:|']
    def fmt(row,k): return ('{:+.3f}'.format(row[k]*100) if k in row else '—')
    for row in perpair:
        lines.append(f'|{row["domain"]}|{row["left"]}|{row["right"]}|{fmt(row,"delta_AP")}|{fmt(row,"delta_AR100")}|{fmt(row,"delta_AP_small")}|{fmt(row,"delta_AP_large")}|')
    lines.extend(['','## Missing evidence','',f'Uncaptured P0 branches: {len(missing)}.','',
                  '## Warnings','']+[f'- {w}' for w in bad[:100]])
    lines.extend(['','All numbers above were **read from existing files**; no inference, scoring or bootstrap was run.'])
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--output',default='audits/a4_readonly_evidence')
    a=p.parse_args();res=scan(Path(a.repo).resolve(),Path(a.output).resolve())
    print('READ_ONLY_AUDIT_COMPLETE',json.dumps({'p0':f'{res["p0_branches_found"]}/{res["p0_branches_expected"]}',
          'bootstrap_files':res['bootstrap_files_found'],'warnings':len(res['diagnostic_warnings']),
          'loader':res['source_data_loader'].get('assessment')}))

if __name__=='__main__':main()
