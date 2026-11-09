"""Read-only A4/A4-R verification of dataset wrapper, LR policy and residual gates.

No training, forward pass, evaluation, optimizer updates or file mutations.
Usage:
 python a41_no_training_probe.py --repo D:/YOLOX-3rd \
   --checkpoint YOLOX_outputs/tdevnext3_a4_fourier_grounded/epoch_2_ckpt.pth

IMPORTANT: only load checkpoints you created/trust; torch.load can unpickle data.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize_dataset(dataset):
    chain = []
    seen = set()
    obj = dataset
    while obj is not None and id(obj) not in seen and len(chain) < 12:
        seen.add(id(obj))
        fields = {}
        for key in ('mosaic', 'mosaic_prob', 'mixup_prob', 'max_labels', 'img_size', 'input_dim'):
            val = getattr(obj, key, None)
            if isinstance(val, (int, float, bool, str, tuple, list)):
                fields[key] = val
        preproc = getattr(obj, 'preproc', None)
        if preproc is not None:
            fields['preproc'] = type(preproc).__name__
            if hasattr(preproc, 'max_labels'):
                fields['preproc_max_labels'] = getattr(preproc, 'max_labels')
        chain.append({'type': f'{type(obj).__module__}.{type(obj).__name__}', **fields})
        obj = getattr(obj, 'dataset', None)
    return chain


def checkpoint_gates(path, key):
    import torch
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError('Checkpoint is not a dictionary')
    state = ckpt.get(key)
    if not isinstance(state, dict):
        raise RuntimeError(f'Checkpoint lacks {key} state dict: keys={list(ckpt)}')
    gates = []
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        short = name.removeprefix('module.')
        if short.endswith('native_skip_gain') or short.endswith('mem_gate'):
            val = float(value.detach().float().reshape(-1)[0])
            gates.append({'name': short, 'raw': val, 'tanh': float(torch.tanh(torch.tensor(val))),
                          'effective_spi_coefficient': float(.25 * torch.tanh(torch.tensor(val))) if short.endswith('native_skip_gain') else None})
    return gates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='.')
    p.add_argument('--exp', default='exps/default/tde_yolox_s_fourier_a4.py')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--weights', choices=['ema_model','model'], default='ema_model')
    a = p.parse_args()
    root = Path(a.repo).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from yolox.exp import get_exp
    exp_path = root/a.exp
    exp = get_exp(str(exp_path), None)
    loader = exp.get_data_loader(batch_size=1, is_distributed=False,
                                 no_aug=False, cache_img=False)
    chain = summarize_dataset(loader.dataset)
    ds_types = [item['type'] for item in chain]
    msrc = inspect.getsource(type(exp).get_data_loader)
    rsrc = inspect.getsource(type(exp).random_resize)
    report = {
        'experiment': a.exp,
        'experiment_source_sha256': sha(exp_path),
        'loader_class': type(loader).__name__,
        'dataset_chain': chain,
        'sampler_class': type(getattr(loader, 'batch_sampler', None)).__name__,
        'sampler_mosaic_flag': getattr(getattr(loader, 'batch_sampler', None), 'mosaic', None),
        'explicit_mosaic_wrapper_in_runtime_chain': any('MosaicDetection' in name for name in ds_types),
        'loader_source_calls_mosaic_detection': 'MosaicDetection(' in msrc,
        'random_resize_source_conditions_on_epoch_or_no_aug': ('no_aug_epochs' in rsrc or 'no_aug' in rsrc),
        'exp_mosaic_prob': getattr(exp, 'mosaic_prob', None),
        'no_aug_epochs': getattr(exp, 'no_aug_epochs', None),
        'max_epoch': getattr(exp, 'max_epoch', None),
    }
    if a.checkpoint:
        ckpt = (root/a.checkpoint).resolve()
        report['checkpoint'] = str(ckpt)
        report['checkpoint_sha256'] = sha(ckpt)
        report['residual_gates'] = checkpoint_gates(ckpt, a.weights)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print('NOTE: source + wrapper inspection is read-only. To prove stochastic Mosaic rate, a controlled sample-level probe is still required.')

if __name__ == '__main__':
    main()
