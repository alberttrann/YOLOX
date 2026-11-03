#!/usr/bin/env python3
"""Export EMA *weights only* for controlled fresh vNext.2 ablation runs.

Usage from YOLOX root:
python tools/export_vnext2_epoch2_ema.py \
  --input YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth \
  --output YOLOX_outputs/vnext2_epoch2_ema_weights_only.pth \
  --model-exp exps/default/tde_yolox_s_ttt_context_ablation.py

Does not edit source checkpoint, resume epoch, optimizer, or EMA state.
"""
import argparse
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model-exp',required=True)
    args=p.parse_args()
    source=Path(args.input).resolve()
    target=Path(args.output).resolve()
    if source==target:
        raise ValueError('Never overwrite the original checkpoint')
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    if not isinstance(ckpt,dict) or not isinstance(ckpt.get('ema_model'),dict):
        raise RuntimeError('Checkpoint must contain EMA weights under ema_model')
    weights={k.removeprefix('module.'):v.detach().cpu().clone()
             for k,v in ckpt['ema_model'].items()}
    from yolox.exp import get_exp
    exp=get_exp(args.model_exp,None)
    model=exp.get_model().cpu()
    model.load_state_dict(weights,strict=True)
    print('STRICT EMA KEY MATCH:',len(weights),'keys; model parameters:',sum(p.numel() for p in model.parameters()))
    del model
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':weights,
                'origin':str(source),
                'source_ema_checkpoint_epoch':ckpt.get('start_epoch'),
                'note':'fresh-training INITIAL WEIGHTS ONLY; not a resume checkpoint'},target)
    print('SAVED',target)
    print('TRAIN WITH -c THIS_FILE WITHOUT --resume; reset optimizer, EMA and schedule')

if __name__=='__main__':
    main()
