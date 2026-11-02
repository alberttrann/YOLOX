#!/usr/bin/env python3
"""Single-batch tde-vNext backward forensics.

Usage (from YOLOX repository root):
  python tdevnext_gradient_forensics.py --checkpoint YOLOX_outputs/tde_yolox_s/last_epoch_ckpt.pth --batch-size 2 --init-scale 1

Runs the SAME model weights and cached image batch in four configurations:
  AMP / FP32  x  Tribrid global context active / forcibly bypassed.
No optimizer step is performed and no checkpoint is modified.

Note: `bypass` replaces the Tribrid global_mixer computation with an actual
zero tensor. Setting the external gate to zero is NOT the same test: the
original branch is still evaluated and its backward may be non-finite.
"""

import argparse
import contextlib
import json
import math
import random
import types
from pathlib import Path

import torch


def _group(name):
    if 'global_mixer' in name: return 'Tribrid global_mixer'
    if 'C3_' in name: return 'Tribrid dense & gates'
    if 'affordance_head' in name: return 'Affordance'
    if 'ttt_lrs' in name: return 'TTT meta LR'
    if 'projector' in name and 'dark2' in name: return 'TTT projector'
    if 'dark2' in name: return 'Backbone dark2'
    if 'dark3' in name: return 'Backbone dark3'
    if 'dark4' in name: return 'Backbone dark4'
    if 'dark5' in name: return 'Backbone dark5'
    if 'spi_highway' in name: return 'SPI'
    if 'head.reg_' in name or 'head.gamma_reg' in name: return 'Head bbox'
    if 'head.obj_' in name: return 'Head objectness'
    if 'head.cls_' in name: return 'Head classification'
    if 'memory' in name or 'latent' in name or 'mem_gate' in name: return 'Memory'
    return 'Other'


def gradient_report(model):
    groups = {}
    bad_params = []
    sum_sq64 = 0.0
    sum_sq32 = 0.0
    n_grad = 0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None: continue
        n_grad += 1
        g = grad.detach()
        finite = torch.isfinite(g)
        nonfinite = (~finite).sum().item()
        stats = groups.setdefault(_group(name), {'params': 0, 'bad_params': 0, 'bad_elements': 0})
        stats['params'] += 1
        stats['bad_elements'] += int(nonfinite)
        if nonfinite:
            stats['bad_params'] += 1
            if len(bad_params) < 30:
                bad_params.append({
                    'name': name,
                    'nan': int(torch.isnan(g).sum().item()),
                    'pos_inf': int(torch.isposinf(g).sum().item()),
                    'neg_inf': int(torch.isneginf(g).sum().item()),
                    'elements': g.numel(),
                })
        # Sum in FP64 to distinguish norm overflow from truly nonfinite grads.
        # Nonfinite elements are excluded from this norm deliberately.
        gd = torch.where(finite, g.double(), torch.zeros_like(g, dtype=torch.float64))
        sum_sq64 += gd.square().sum().item()
        # Mimic the original recorder's FP32 norm reduction, including bad elements.
        sum_sq32 += g.float().square().sum().item()

    return {
        'gradient_tensors': n_grad,
        'nonfinite_gradient_parameters': sum(v['bad_params'] for v in groups.values()),
        'nonfinite_gradient_elements': sum(v['bad_elements'] for v in groups.values()),
        'finite_element_norm_fp64': math.sqrt(sum_sq64),
        'original_style_grad_norm_fp32': math.sqrt(sum_sq32) if sum_sq32 >= 0 else float('nan'),
        'groups': groups,
        'first_bad_parameters': bad_params,
    }


@contextlib.contextmanager
def bypass_tribrid_global(model, enabled):
    if not enabled:
        yield 0
        return
    saved = []
    for module in model.modules():
        if module.__class__.__name__ == 'C2f_Tribrid':
            mixer = module.global_mixer
            saved.append((mixer, mixer.forward))
            def _zero_forward(self, x):
                return torch.zeros_like(x)
            mixer.forward = types.MethodType(_zero_forward, mixer)
    try:
        yield len(saved)
    finally:
        for mixer, original in saved:
            mixer.forward = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp', default='exps/default/tde_yolox_s.py')
    p.add_argument('--checkpoint', required=True, help='Use last_epoch_ckpt.pth for latest state, not best_ckpt.pth.')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--init-scale', type=float, default=1.0,
                   help='AMP GradScaler init scale; use 1 to separate intrinsic numerical issues from scale overflow, then retry 65536.')
    p.add_argument('--seed', type=int, default=703)
    p.add_argument('--anomaly', action='store_true', help='Enable expensive autograd anomaly traceback for first failing mode.')
    p.add_argument('--output', default='tdevnext_gradient_forensics.json')
    args = p.parse_args()
    if args.batch_size < 2:
        p.error('Use --batch-size >= 2 so the TTT meta sub-batch is nonempty.')
    if not torch.cuda.is_available():
        p.error('This diagnostic requires CUDA because the supplied project code uses CUDA-dependent code paths.')

    from yolox.exp import get_exp
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    exp = get_exp(args.exp, None)
    exp.data_num_workers = 0  # Windows-safe diagnostic without persistent subprocess workers
    model = exp.get_model().cuda().train()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print('Checkpoint:', args.checkpoint)
    print('Missing keys:', len(missing), 'Unexpected keys:', len(unexpected))
    if missing or unexpected:
        print('First missing:', missing[:12]); print('First unexpected:', unexpected[:12])
        raise RuntimeError('Checkpoint mismatch; use a checkpoint created by the same architecture.')
    epoch = args.epoch or int(ckpt.get('start_epoch', 1))
    if hasattr(model, 'set_meta_training_state'):
        model.set_meta_training_state(epoch, getattr(exp, 'max_epoch', 80))

    loader = exp.get_data_loader(batch_size=args.batch_size, is_distributed=False,
                                 no_aug=True, cache_img=False)
    batch = next(iter(loader))
    inps, targets = batch[:2]
    inps = inps.cuda(non_blocking=False).float()
    targets = targets.cuda(non_blocking=False).float()
    inps, targets = exp.preprocess(inps, targets, exp.input_size)
    print('Batch:', tuple(inps.shape), 'Epoch:', epoch, 'AMP init scale:', args.init_scale)

    # Snapshot buffers because repeated train forwards modify BatchNorm stats.
    orig_buffers = {name: v.detach().clone() for name, v in model.named_buffers()}
    modes = [(True, False), (True, True), (False, False), (False, True)]
    results = []
    for amp, bypass in modes:
        name = ('AMP' if amp else 'FP32') + (' + BYPASS_CONTEXT' if bypass else ' + FULL_CONTEXT')
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for key, value in model.named_buffers():
                if key in orig_buffers: value.copy_(orig_buffers[key])
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=args.init_scale)
        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        print('\n---', name, '---', flush=True)
        with bypass_tribrid_global(model, bypass) as n_bypassed:
            try:
                with torch.autograd.set_detect_anomaly(args.anomaly):
                    with torch.cuda.amp.autocast(enabled=amp):
                        out = model(inps, targets)
                    loss = out['total_loss']
                    if torch.isfinite(loss).item():
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        report = gradient_report(model)
                    else:
                        report = {'error': 'non-finite forward loss'}
                record = {
                    'mode': name, 'branches_bypassed': n_bypassed,
                    'loss': float(loss.detach().cpu()),
                    'components': {k: float(v.detach().cpu()) for k, v in out.items()
                                   if k in ('iou_loss', 'conf_loss', 'cls_loss', 'mem_loss', 'rec_loss', 'afford_loss')},
                    **report
                }
                print('Loss:', record['loss'])
                print('Nonfinite gradient parameters:', report.get('nonfinite_gradient_parameters'))
                print('Finite-element FP64 norm:', report.get('finite_element_norm_fp64'))
                print('Original-style FP32 norm:', report.get('original_style_grad_norm_fp32'))
                print('Bad groups:', {k:v['bad_params'] for k,v in report.get('groups',{}).items() if v['bad_params']})
                print('First bad parameters:', [x['name'] for x in report.get('first_bad_parameters',[])[:8]])
            except Exception as exc:
                record = {'mode':name, 'branches_bypassed':n_bypassed,
                          'error':f'{type(exc).__name__}: {exc}'}
                print('ERROR:',record['error'],flush=True)
            results.append(record)
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps({'checkpoint':args.checkpoint,'epoch':epoch,'results':results}, indent=2,
                                           allow_nan=True), encoding='utf-8')
    print('\nSaved:',args.output)
    print('No optimizer steps were executed. Compare FULL_CONTEXT and BYPASS_CONTEXT first.')


if __name__ == '__main__':
    main()
