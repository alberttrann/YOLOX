"""P2 fail-closed multi-dataset campaign with identical evaluation policy.

Manifest contains explicit model checkpoints and registered data-experiments.
No condition-specific tuning, no guessed class remaps. Calls counterfactual
runner for each pair to retain exact detection lists and metadata.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--weights',choices=['ema','raw'],default='ema')
    a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    required={'models','domains','batch_size','conf','nms'}
    if not required.issubset(manifest):raise ValueError('Missing manifest keys: '+str(required-set(manifest)))
    if not manifest['models'] or not manifest['domains']:raise ValueError('Empty evaluation campaign')
    runner=Path(__file__).with_name('a4_eval_counterfactual.py')
    for model in manifest['models']:
        for domain in manifest['domains']:
            if not all(k in model for k in ('name','model_exp','checkpoint')):
                raise RuntimeError('Model spec missing name, model_exp, or checkpoint')
            if not all(k in domain for k in ('name','data_exp')):
                raise RuntimeError('Domain spec missing name or data_exp')
            if not Path(model['checkpoint']).exists():
                raise FileNotFoundError('Missing checkpoint: '+model['checkpoint'])
            if not Path(domain['data_exp']).exists():
                raise FileNotFoundError('Missing data exp: '+domain['data_exp'])
            out=Path(a.output_root)/model['name']/domain['name']
            cmd=[sys.executable,str(runner),'--model-exp',model['model_exp'],
                 '--data-exp',domain['data_exp'],'--checkpoint',model['checkpoint'],
                 '--weights',a.weights,'--output-dir',str(out),
                 '--batch-size',str(manifest['batch_size']),
                 '--conf',str(manifest['conf']),'--nms',str(manifest['nms']),
                 '--branches','on']
            if domain.get('split','val')=='test':cmd.append('--test')
            elif domain.get('split','val')!='val':raise ValueError('Unknown domain split')
            print('CAMPAIGN',model['name'],domain['name'],' '.join(cmd),flush=True)
            if a.execute:
                subprocess.run(cmd,check=True)


if __name__=='__main__':main()
