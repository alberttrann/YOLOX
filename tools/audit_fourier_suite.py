"""Read-only suite provenance, variant and metric preflight.

Run without CUDA to check model experiment definitions, dataset geometry,
A2 checkpoint protection, historical epoch files, and adapter flags.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from fourier_suite_metrics import VARIANTS


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args(argv)
    repo=Path(args.repo).resolve()
    sys.path.insert(0,str(repo))
    from yolox.exp import get_exp
    values={}
    for key,(variant,exp_name,exp_file) in VARIANTS.items():
        exp=get_exp(str(repo/'exps/default'/exp_file),None)
        if exp.fourier_variant!=variant or exp.exp_name!=exp_name:
            raise RuntimeError(f'{key} variant/output mismatch')
        preflight=exp._preflight()
        manifests=repo/'YOLOX_outputs'/exp_name/'fourier_architecture_manifest.json'
        recorded=json.loads(manifests.read_text(encoding='utf-8')) if manifests.is_file() else None
        state=repo/'YOLOX_outputs'/exp_name
        values[key]={
            'variant':variant,'experiment':exp_name,'preflight':preflight,
            'seed':exp.seed,'batch_independent_config_batch':8,
            'max_epoch':exp.max_epoch,'basic_lr_per_img':exp.basic_lr_per_img,
            'warmup_epochs':exp.warmup_epochs,'no_aug_epochs':exp.no_aug_epochs,
            'mosaic_prob':exp.mosaic_prob,'mixup_prob':exp.mixup_prob,
            'input_size':exp.input_size,'test_size':exp.test_size,
            'checkpoints':{str(i):(state/f'epoch_{i}_ckpt.pth').is_file() for i in (1,2)},
            'provenance_present':bool(recorded),
            'provenance_variant_matches':(recorded.get('variant')==variant if recorded else None),
        }
        print(f'{key} {variant}: exp={exp_name}, input={exp.input_size}, classes={exp.num_classes}, '
              f'epoch1={values[key]["checkpoints"]["1"]}, epoch2={values[key]["checkpoints"]["2"]}')
    ref=values['A2']
    checked=['preflight','seed','max_epoch','basic_lr_per_img','warmup_epochs','no_aug_epochs',
             'mosaic_prob','mixup_prob','input_size','test_size']
    for key,vals in values.items():
        for field in checked:
            if vals[field]!=ref[field]:raise RuntimeError(f'{key}: {field} differs from A2')
        if vals['provenance_variant_matches'] is False:
            raise RuntimeError(f'{key} manifest variant mismatch')
    if args.require_complete:
        incomplete=[(k,e) for k,v in values.items() for e,exists in v['checkpoints'].items() if not exists]
        if incomplete:raise RuntimeError('Missing history checkpoints: '+str(incomplete))
    print('PASS experiment configuration parity. Warning: CUDA RNG/dataloader order and dataset hashes '
          'must still be inspected for exact per-sample stochastic parity.')


if __name__=='__main__':main()
