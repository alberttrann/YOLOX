#!/usr/bin/env python3
"""Independent diagnostics for the supplied tde-vNext YOLOX code.
Run from the root of the actual YOLOX repository, for example:
  python tdevnext_audit_probe.py --exp exps/default/tde_yolox_s.py \
      --init-checkpoint yolox_s.pth \
      --checkpoint YOLOX_outputs/tde_yolox_s/best_ckpt.pth \
      --telemetry YOLOX_outputs/tde_yolox_s/telemetry/gradient_flow_telemetry.jsonl
It never modifies checkpoint files or source modules.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import torch


def norm_of_grad(p):
    return None if p.grad is None else float(p.grad.detach().float().norm())


def probe_tribrid():
    from yolox.models.tribrid_neck import C2f_Tribrid
    torch.manual_seed(2026)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    x = torch.randn(2, 32, 8, 8)
    names = ('gate', 'cv1.conv.weight', 'cv2.conv.weight',
             'global_mixer.3.proj.weight', 'global_mixer.3.local_v_conv.weight',
             'global_mixer.3.qkv.weight')
    for gate_value in (0.0, 1.0):
        m = C2f_Tribrid(32, 32, n=1, saliency_floor=0.15)
        with torch.no_grad():
            m.gate.fill_(gate_value)
        m.train()
        m(x).square().mean().backward()
        d = dict(m.named_parameters())
        print(f'\nTRIBRID gate_init={gate_value}:')
        for name in names:
            print(f'  {name:49s} grad_norm={norm_of_grad(d[name])}')
        print('  NOTE: gated branch QKV initially has zero gradient because its final'
              ' projections are zero; it should receive gradients after those'
              ' projections update. At gate=0 even those projections are dead.')


def unpack(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    return d, d.get('model', d) if isinstance(d, dict) else d


def scan_init(exp_path, checkpoint):
    from yolox.exp import get_exp
    model = get_exp(exp_path, None).get_model()
    model_sd = model.state_dict()
    ckpt, pretrained = unpack(checkpoint)
    pretrained = {k.removeprefix('module.'): v for k, v in pretrained.items()}
    source = dict(pretrained)
    for k, v in pretrained.items():
        if 'dark2.' in k and 'backbone_stage' not in k and '.bn.' not in k:
            source[k.replace('dark2.', 'dark2.backbone_stage.')] = v

    param_keys = set(dict(model.named_parameters()))
    by_region = defaultdict(lambda: [0, 0, 0, 0])  # matching-count,total-count,matching-numel,total-numel
    for k, v in model_sd.items():
        if k not in param_keys:
            continue
        parts = k.split('.')
        if k.startswith('backbone.backbone.'):
            region = '.'.join(parts[:3])
        elif k.startswith('backbone.'):
            region = '.'.join(parts[:2])
        elif k.startswith('head.'):
            region = 'head.' + parts[1]
        else:
            region = parts[0]
        stat = by_region[region]
        stat[1] += 1
        stat[3] += v.numel()
        if k in source and source[k].shape == v.shape:
            stat[0] += 1
            stat[2] += v.numel()
    print('\nPRETRAINED CHECKPOINT KEY/SHAPE COVERAGE (matching != successful semantic transfer):')
    for region, (kmatch, ktotal, nmatch, ntotal) in sorted(by_region.items()):
        print(f'  {region:42s} keys={kmatch:4}/{ktotal:<4} elements={nmatch:>10}/{ntotal:<10} ({nmatch / max(ntotal,1):6.1%})')
    print('  IMPORTANT: This reports shape/key compatibility only; it cannot prove'
          ' downstream feature equivalence or semantic correctness.')


def scan_saved_checkpoint(checkpoint):
    ckpt, raw = unpack(checkpoint)
    ema = ckpt.get('ema_model') if isinstance(ckpt, dict) else None
    print(f'\nCHECKPOINT: {checkpoint}')
    if isinstance(ckpt, dict):
        print('  start_epoch:', ckpt.get('start_epoch'), 'current_ap:', ckpt.get('curr_ap'),
              'best_ap:', ckpt.get('best_ap'))
    if ema is None:
        print('  No separate EMA model stored.'); return
    groups = defaultdict(lambda: [0., 0., 0])
    for k, r in raw.items():
        e = ema.get(k)
        if e is None or not torch.is_tensor(r) or not torch.is_floating_point(r) or r.shape != e.shape:
            continue
        group = '.'.join(k.split('.')[:2])
        vals = groups[group]
        vals[0] += (r.float()-e.float()).square().sum().item()
        vals[1] += e.float().square().sum().item()
        vals[2] += r.numel()
    print('  Relative L2 delta raw vs EMA by parameter group:')
    for k, (a, b, n) in sorted(groups.items()):
        print(f'    {k:36s} {math.sqrt(a) / max(1e-12, math.sqrt(b)):.6g} ({n} elements)')
    print('  Validation uses EMA; standard tools/eval.py in supplied code uses raw weights.')


def scan_telemetry(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    per_group = defaultdict(lambda: [0, 0])
    nonfinite_records = 0
    for record in records:
        had = False
        for group, value in record.get('grad_norms', {}).items():
            per_group[group][1] += 1
            if not math.isfinite(float(value)):
                per_group[group][0] += 1
                had = True
        nonfinite_records += had
    print(f'\nTELEMETRY: {len(records)} recorded steps; {nonfinite_records} had >=1 non-finite recorded group norm')
    for group, (bad, all_) in sorted(per_group.items()):
        print(f'  {group:30s} {bad:5d}/{all_} ({bad/max(1, all_):.2%})')
    print('  A non-finite norm can arise from a non-finite gradient or numeric overflow'
          ' during norm reduction. Add finite checks on individual gradients before clipping.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default='exps/default/tde_yolox_s.py')
    p.add_argument('--init-checkpoint')
    p.add_argument('--checkpoint')
    p.add_argument('--telemetry')
    p.add_argument('--skip-tribrid', action='store_true')
    args = p.parse_args()
    print('PyTorch version:', torch.__version__)
    if not args.skip_tribrid:
        probe_tribrid()
    if args.init_checkpoint:
        scan_init(args.exp, args.init_checkpoint)
    if args.checkpoint:
        scan_saved_checkpoint(args.checkpoint)
    if args.telemetry:
        scan_telemetry(args.telemetry)


if __name__ == '__main__':
    main()
