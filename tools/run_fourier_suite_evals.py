"""Strict, condition-blind OOD A0/A1/A2/A3 evaluator and result recorder.

Requires the user's installed 01_eval_only patch: --model-exp, --data-exp,
--weights ema, --ttt-mode off.  Will not run a legacy data experiment as model.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from fourier_suite_metrics import VARIANTS,parse_coco_summary,delta

CORE_KEYS=['AP','AP50','AP75','AR100','AP_small','AP_medium','AP_large',
           'AR100_small','AR100_medium','AR100_large','forward_ms','inference_ms']


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default='.')
    p.add_argument('--epoch',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--data-exps',nargs='+',default=[
        'exps/default/test_acdc_fog.py','exps/default/test_acdc_rain.py',
        'exps/default/test_acdc_snow.py','exps/default/test_acdc_night.py'])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--skip-existing',action='store_true')
    a=p.parse_args(argv)
    repo=Path(a.repo).resolve()
    eval_script=repo/'tools/eval.py'
    if not eval_script.is_file():raise FileNotFoundError(eval_script)
    src=eval_script.read_text(encoding='utf-8')
    for required in ('--model-exp','--data-exp','--ttt-mode','--weights'):
        if required not in src:
            raise RuntimeError('Installed eval.py lacks strict Fourier-compatible flag '+required)
    dest=repo/'YOLOX_outputs/fourier_suite_reports/ood'
    if not a.dry_run:dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label in a.variants:
        variant,exp_name,exp_filename=VARIANTS[label]
        exp_path='exps/default/'+exp_filename
        exp_root=repo/'YOLOX_outputs'/exp_name
        ckpt=exp_root/f'epoch_{a.epoch}_ckpt.pth'
        if not ckpt.is_file():
            raise FileNotFoundError(f'Missing {label} epoch-{a.epoch} history checkpoint: {ckpt}. '
                                    'Do not substitute a different epoch.')
        manifest=exp_root/'fourier_architecture_manifest.json'
        if not manifest.is_file() or json.loads(manifest.read_text(encoding='utf-8')).get('variant')!=variant:
            raise RuntimeError(f'{label} checkpoint provenance missing or variant mismatch')
        for data_exp in a.data_exps:
            data_name=Path(data_exp).stem
            if not (repo/data_exp).is_file():raise FileNotFoundError(repo/data_exp)
            tag=f'{label}__{data_name}__epoch{a.epoch}'
            txt_file=dest/f'{tag}.log'
            meta_file=dest/f'{tag}.json'
            cmd=[sys.executable,str(eval_script),'--model-exp',exp_path,'--data-exp',data_exp,
                 '-c',str(ckpt),'--weights','ema','--ttt-mode','off','-b',str(a.batch_size),
                 '--conf','0.01','--nms','0.60']
            if a.dry_run:
                print('DRY_RUN',label,data_name,' '.join(cmd));continue
            if a.skip_existing and meta_file.is_file():
                previous=json.loads(meta_file.read_text(encoding='utf-8'))
                if previous.get('command')!=cmd or previous.get('returncode')!=0:
                    raise RuntimeError(f'Previous result has different protocol or failed: {meta_file}')
                row=previous['metrics']
                print('SKIP_EXISTING',tag)
            else:
                print('EVALUATING',tag,flush=True)
                process=subprocess.run(cmd,cwd=repo,capture_output=True,text=True,encoding='utf-8',errors='replace')
                combined=process.stdout+'\n'+process.stderr
                txt_file.write_text(combined,encoding='utf-8')
                metrics=parse_coco_summary(combined)
                if process.returncode or metrics.get('AP') is None:
                    raise RuntimeError(f'Evaluation failed / no COCO AP for {tag}. '
                                       f'Exit={process.returncode}. See {txt_file}')
                row={k:v for k,v in metrics.items()}
                meta_file.write_text(json.dumps({
                    'variant':label,'data_exp':data_exp,'epoch':a.epoch,
                    'weights':'ema','ttt_mode':'off','conf':0.01,'nms':0.60,
                    'batch_size':a.batch_size,'checkpoint':str(ckpt),
                    'command':cmd,'returncode':process.returncode,'metrics':row
                },indent=2),encoding='utf-8')
                print('DONE',tag,'AP',metrics['AP'],'AR100',metrics.get('AR100'))
            rows.append({'variant':label,'data_exp':data_exp,'epoch':a.epoch,
                         **{key:row.get(key) for key in CORE_KEYS}})
    if a.dry_run:return 0
    out_csv=dest/'ood_metrics.csv'
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','data_exp','epoch']+CORE_KEYS)
        writer.writeheader();writer.writerows(rows)
    indexed={(r['variant'],r['data_exp']):r for r in rows}
    deltas=[]
    for data_exp in a.data_exps:
        for l,r in [('A1','A0'),('A2','A1'),('A3','A2'),('A3','A0')]:
            x=indexed.get((l,data_exp));y=indexed.get((r,data_exp))
            if not x or not y:continue
            deltas.append({'dataset':data_exp,'comparison':f'{l}-{r}',
                           **{key:delta(x.get(key),y.get(key)) for key in CORE_KEYS}})
    with (dest/'ood_deltas.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','comparison']+CORE_KEYS)
        writer.writeheader();writer.writerows(deltas)
    print('SAVED',out_csv,'and',dest/'ood_deltas.csv')
    return 0


if __name__=='__main__':sys.exit(main())
