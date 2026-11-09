"""Analyze A0/A1/A2/A3 clean-val logs, OOD results and provenance.

Only compares observed metrics. Missing values are left null; no invented AP.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from fourier_suite_metrics import VARIANTS,parse_training_evaluations,delta

PRIMARY=['AP','AP50','AP75','AR100','AP_small','AP_medium','AP_large',
         'AR100_small','AR100_medium','AR100_large']
OTHER=['forward_ms','nms_ms','inference_ms']


def read_json(file):
    return json.loads(file.read_text(encoding='utf-8')) if file.is_file() else None


def write_csv(path, rows, columns):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default='.')
    p.add_argument('--output',default='YOLOX_outputs/fourier_suite_reports')
    p.add_argument('--epochs',nargs='+',type=int,default=[1,2])
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args(argv)
    repo=Path(a.repo).resolve()
    report=(repo/a.output).resolve();report.mkdir(parents=True,exist_ok=True)
    allmetrics={};rows=[];class_rows=[];provenance={}
    for label, (_,dirname,_) in VARIANTS.items():
        loc=repo/'YOLOX_outputs'/dirname
        manifests=read_json(loc/'fourier_architecture_manifest.json')
        protocol=read_json(loc/'suite_protocol.json')
        provenance[label]={'architecture':manifests,'suite_protocol':protocol}
        if manifests and manifests.get('variant')!=VARIANTS[label][0]:
            raise RuntimeError(f'{label}: manifest variant mismatch: {manifests.get("variant")}')
        logfile=loc/'train_log.txt'
        allmetrics[label]=(parse_training_evaluations(logfile.read_text(encoding='utf-8',errors='replace'))
                           if logfile.is_file() else {})
        for epoch in a.epochs:
            m=allmetrics[label].get(epoch,{})
            rows.append({'variant':label,'epoch':epoch,'checkpoint':str(loc/f'epoch_{epoch}_ckpt.pth'),
                         'checkpoint_exists':(loc/f'epoch_{epoch}_ckpt.pth').is_file(),
                         **{k:m.get(k) for k in PRIMARY+OTHER}})
            for kind in ('AP','AR'):
                for name,val in m.get('per_class_'+kind,{}).items():
                    class_rows.append({'variant':label,'epoch':epoch,'class':name,'metric':kind,'value':val})
    missing=[f'{l} epoch {e}' for l in VARIANTS for e in a.epochs if e not in allmetrics[l]]
    if missing:print('MISSING_EVALUATIONS',', '.join(missing))
    if a.require_complete and missing:
        raise RuntimeError('Not all requested evaluations are present; refusing complete-suite claims')
    comparisons=[('A1','A0'),('A2','A1'),('A3','A2'),('A3','A0')]
    deltas=[]
    for epoch in a.epochs:
        for left,right in comparisons:
            ml=allmetrics[left].get(epoch,{})
            mr=allmetrics[right].get(epoch,{})
            entry={'comparison':f'{left}-{right}','epoch':epoch,
                   **{k:delta(ml.get(k),mr.get(k)) for k in PRIMARY+OTHER}}
            noninfer=[k for k in ('AP_small','AP_medium','AP_large',
                                   'AR100_small','AR100_medium','AR100_large')
                      if entry[k] is not None and entry[k]<0]
            entry['scale_regressions']=','.join(noninfer) if ml and mr else None
            deltas.append(entry)
    write_csv(report/'clean_metrics.csv',rows,['variant','epoch','checkpoint','checkpoint_exists']+PRIMARY+OTHER)
    write_csv(report/'clean_deltas.csv',deltas,['comparison','epoch']+PRIMARY+OTHER+['scale_regressions'])
    write_csv(report/'clean_per_class.csv',class_rows,['variant','epoch','class','metric','value'])
    (report/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    summary={'epochs':a.epochs,'missing':missing,
             'absolute':{k:{str(ep):v for ep,v in vals.items() if ep in a.epochs}
                         for k,vals in allmetrics.items()},'comparisons':deltas}
    (report/'clean_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nCLEAN VALIDATION - fraction units')
    print('variant epoch '+''.join(f'{k:>13}' for k in ('AP','AR100','AP_small','AP_medium','AP_large')))
    for row in rows:
        print(f'{row["variant"]:>7} {row["epoch"]:>5} '+ ''.join(
            f'{row[k]:>13.3f}' if row[k] is not None else f'{"missing":>13}'
            for k in ('AP','AR100','AP_small','AP_medium','AP_large')))
    print('\nDIFFERENCES - positive means improvement over right-hand variant')
    for d in deltas:
        print(f'{d["comparison"]:>6} epoch={d["epoch"]} AP={d["AP"]} AR100={d["AR100"]}'
              f' scale_regressions={d["scale_regressions"]}')
    print('REPORTS',report)
    return 0


if __name__=='__main__':
    sys.exit(main())
