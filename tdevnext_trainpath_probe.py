#!/usr/bin/env python3
"""Training-path AMP/gradient forensics for the supplied tde-vNext YOLOX repository.

Reproduces the key parts of Trainer.train_one_iter: data casting to FP16/FP32,
Exp.preprocess, autocast, GradScaler scale/unscale, gradient inspection and
step-skip detection. Uses lr=0 SGD ONLY to observe whether GradScaler calls
optimizer.step(). No weights are intentionally updated, and nothing is saved
back to the checkpoint. BatchNorm running buffers change on forward passes
but are reset before each variant for a paired comparison.

Examples:
 python tdevnext_trainpath_probe.py --checkpoint YOLOX_outputs/tde_yolox_s/last_epoch_ckpt.pth --batch-size 8 --max-batches 4 --input-dtype fp16 --scales 1,256,65536
 python tdevnext_trainpath_probe.py --checkpoint YOLOX_outputs/tde_yolox_s/last_epoch_ckpt.pth --batch-size 8 --max-batches 4 --input-dtype fp32 --scales 1,65536
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.0, momentum=0.0, weight_decay=0.0)
        self.calls = 0

    def step(self, closure=None):
        self.calls += 1
        return super().step(closure)


def inspect_grads(model, show=12):
    bad_params = []
    total_nan = total_pos_inf = total_neg_inf = 0
    sum_sq64 = 0.0
    max_abs = 0.0
    grad_param_count = 0
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        grad_param_count += 1
        g = g.detach()
        bad = ~torch.isfinite(g)
        nbad = int(bad.sum().item())
        if nbad:
            nan = int(torch.isnan(g).sum().item())
            pos = int(torch.isposinf(g).sum().item())
            neg = int(torch.isneginf(g).sum().item())
            total_nan += nan
            total_pos_inf += pos
            total_neg_inf += neg
            if len(bad_params) < show:
                bad_params.append({'name': name, 'nan': nan, 'posinf': pos, 'neginf': neg})
        gf = g.detach().double()
        good = gf[torch.isfinite(gf)]
        if good.numel():
            sum_sq64 += good.square().sum().item()
            max_abs = max(max_abs, good.abs().max().item())
    return {
        'grad_tensors':grad_param_count, 'bad_param_examples':bad_params,
        'bad_elements':total_nan + total_pos_inf + total_neg_inf,
        'nan_elements':total_nan, 'posinf_elements':total_pos_inf, 'neginf_elements':total_neg_inf,
        'finite_element_l2_fp64':math.sqrt(sum_sq64), 'largest_finite_grad_abs':max_abs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', default='exps/default/tde_yolox_s.py')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-batches', type=int, default=4)
    p.add_argument('--input-dtype', choices=['fp16','fp32'], default='fp16',
                   help='fp16 reproduces Trainer.inps.to(data_type) when --fp16 is enabled')
    p.add_argument('--scales', default='1,256,65536', help='Independent initial loss-scale experiments')
    p.add_argument('--seed', type=int, default=703)
    p.add_argument('--epoch', type=int, default=None)
    p.add_argument('--size', default=None, help='Optional training resize as H,W, e.g. 640,1152')
    p.add_argument('--output', default='tdevnext_trainpath_probe.json')
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('This probe requires CUDA')
    if args.batch_size < 2 or args.max_batches < 1:
        p.error('--batch-size must be >=2 and --max-batches >=1')
    scales = [float(v.strip()) for v in args.scales.split(',')]
    if not scales or any(v <= 0 or not math.isfinite(v) for v in scales):
        p.error('--scales must contain positive finite numbers')

    from yolox.exp import get_exp
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    exp = get_exp(args.exp, None)
    exp.data_num_workers = 0
    model = exp.get_model().cuda().train()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=True)
    epoch = args.epoch or int(checkpoint.get('start_epoch', 1))
    if hasattr(model, 'set_meta_training_state'):
        model.set_meta_training_state(epoch, getattr(exp, 'max_epoch', 80))

    # Fresh batches from the original training dataset. no_aug=False is intentional.
    loader = exp.get_data_loader(args.batch_size, is_distributed=False, no_aug=False, cache_img=False)
    input_size = tuple(map(int, args.size.split(','))) if args.size else tuple(exp.input_size)
    input_dtype = torch.float16 if args.input_dtype == 'fp16' else torch.float32
    before = {name:v.detach().clone() for name,v in model.named_buffers()}
    results=[]
    print('Checkpoint:',args.checkpoint,'epoch:',epoch,'input_size:',input_size)
    print('Data dtype:',args.input_dtype,'AMP enabled: True; loss scales:',scales)
    print('NOTE: grad scaler has a fresh state for every test. No weight updates (lr=0).', flush=True)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        inps, targets = batch[:2]
        inps = inps.cuda().to(input_dtype)
        targets = targets.cuda().to(input_dtype)
        targets.requires_grad = False
        inps, targets = exp.preprocess(inps, targets, input_size)
        gt_count = int((targets[..., 3] > 0).sum().item()) if targets.ndim == 3 and targets.shape[-1] > 3 else None
        for scale in scales:
            with torch.no_grad():
                for n,b in model.named_buffers():
                    if n in before: b.copy_(before[n])
            model.zero_grad(set_to_none=True)
            seed = args.seed + batch_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            optimizer = CountingSGD(model.parameters())
            scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=scale)
            rec={'batch_index':batch_index, 'shape':list(inps.shape), 'ground_truth_count':gt_count, 'scale':scale}
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    out = model(inps, targets)
                loss = out['total_loss']
                rec['loss'] = float(loss.detach().cpu())
                rec['loss_finite'] = bool(torch.isfinite(loss).item())
                if rec['loss_finite']:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    rec.update(inspect_grads(model))
                    # With lr=0.0, this records GradScaler's skip decision while preserving all model weights.
                    scaler.step(optimizer)
                    scaler.update()
                    rec['optimizer_step_called'] = bool(optimizer.calls)
                    rec['scale_after'] = float(scaler.get_scale())
            except (RuntimeError, ValueError, AssertionError) as e:
                rec['error'] = type(e).__name__ + ': ' + str(e)
            results.append(rec)
            print('batch=%d gt=%s scale=%g loss=%s bad=%s step=%s scale_after=%s' % (
                batch_index,gt_count,scale,rec.get('loss'),rec.get('bad_elements'),
                rec.get('optimizer_step_called'),rec.get('scale_after')),flush=True)
            if rec.get('bad_param_examples'):
                print('  first bad params:', rec['bad_param_examples'][:5],flush=True)
        del inps,targets
    Path(args.output).write_text(json.dumps({'checkpoint':args.checkpoint,'config':vars(args),'results':results},
                                       indent=2,allow_nan=False), encoding='utf-8')
    print('\nSaved:',args.output)
    counts=Counter((r.get('scale'),r.get('optimizer_step_called')) for r in results)
    print('step called counts by (scale,step_called):',dict(counts))
    print('No weights were changed by the probe (optimizer learning rate = 0).')


if __name__=='__main__':
    main()
