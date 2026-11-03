#!/usr/bin/env python3
"""Read-only vNext.2 checkpoint audit on two real evaluation samples.

Uses the trained model constructor and the existing COCO/ValTransform evaluation
pipeline. It does NOT alter model files or checkpoints and never optimizes.

Examples:
 python tools/diagnose_vnext2.py --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth \
  --model-exp exps/default/tde_yolox_s.py --data-exp exps/default/test_acdc_fog.py
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from yolox.exp import get_exp


def relative_l2(a, b):
    a = a.detach().double().reshape(-1)
    b = b.detach().double().reshape(-1)
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1.e-12)).item()


def deltas(a, b):
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}')
    d = (a.detach().float() - b.detach().float()).abs()
    return {'max_abs': float(d.max().item()), 'mean_abs': float(d.mean().item()),
            'relative_l2': relative_l2(a, b)}


def annotation_report(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    images = doc.get('images', [])
    anns = doc.get('annotations', [])
    categories = doc.get('categories', [])
    per_image = Counter(a['image_id'] for a in anns if 'image_id' in a and not a.get('ignore', False))
    per_category = Counter(a['category_id'] for a in anns if 'category_id' in a and not a.get('ignore', False))
    hist = sorted([per_image.get(im['id'], 0) for im in images])
    def percentile(frac):
        return hist[min(len(hist)-1, int(math.ceil(len(hist)*frac))-1)] if hist else 0
    return {
        'annotation_file': str(path), 'images': len(images), 'annotations': len(anns),
        'crowd_annotations': sum(int(a.get('iscrowd', 0)) for a in anns),
        'images_without_annotations': sum(n == 0 for n in hist),
        'objects_per_image': {'max': max(hist, default=0), 'p50': percentile(.50),
                              'p90': percentile(.90), 'p99': percentile(.99)},
        'class_support': [{'name': c.get('name', str(c['id'])), 'category_id': c['id'],
                           'num_annotations': per_category.get(c['id'], 0)} for c in categories],
    }


def get_image(dataset, index, device):
    item = dataset[index]
    tensor = torch.as_tensor(item[0], dtype=torch.float32, device=device)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise RuntimeError(f'Expected val-transformed CHW image, got {tuple(tensor.shape)}')
    return tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-exp', default='exps/default/tde_yolox_s.py')
    parser.add_argument('--data-exp', required=True)
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--image-index', type=int, default=0)
    parser.add_argument('--other-index', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.fp16 and device.type != 'cuda':
        raise RuntimeError('FP16 evaluation requested but CUDA is unavailable')

    exp = get_exp(args.model_exp, None)
    data_exp = get_exp(args.data_exp, None)
    for field in ('data_dir', 'val_ann', 'test_ann', 'test_size'):
        setattr(exp, field, getattr(data_exp, field))
    model = exp.get_model()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.weights == 'ema':
        if not isinstance(ck, dict) or 'ema_model' not in ck:
            raise RuntimeError('Checkpoint has no EMA state')
        state = ck['ema_model']
    else:
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state.items()}, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)  # Emulate true frozen EMA inference, even for raw weights.
    ck_epoch = int(ck.get('start_epoch', 1)) if isinstance(ck, dict) else 1
    if hasattr(model, 'set_meta_training_state'):
        model.set_meta_training_state(max(1, ck_epoch), int(exp.max_epoch))
    if not hasattr(model, 'eval_ttt_enabled'):
        raise RuntimeError('Model lacks eval_ttt_enabled protocol switch; apply vNext.2 eval-only fix')

    # Preserve the user's actual evaluation preprocessing, not random synthetic noise.
    dataset = exp.get_eval_dataset(legacy=False)
    if len(dataset) <= max(args.image_index, args.other_index):
        raise IndexError(f'Dataset has {len(dataset)} samples; need indices {args.image_index}, {args.other_index}')
    x = get_image(dataset, args.image_index, device)
    y = get_image(dataset, args.other_index, device)
    if x.shape != y.shape:
        raise RuntimeError(f'ValTransform images disagree in shape: {tuple(x.shape)} vs {tuple(y.shape)}')

    dark = model.backbone.backbone
    captures = {}
    def hook_native(_module, _inputs, output):
        captures['native'] = output.detach().float().cpu()
    def hook_dark3(_module, inputs):
        captures['adapted'] = inputs[0].detach().float().cpu()
    h0 = dark.dark2.backbone_stage.register_forward_hook(hook_native)
    h1 = dark.dark3.register_forward_pre_hook(hook_dark3)
    snapshot = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    def forward(batch, enable_ttt):
        captures.clear()
        model.eval_ttt_enabled = enable_ttt
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                result = model(batch)
        if not isinstance(result, torch.Tensor):
            raise RuntimeError(f'Expected decoded tensor for eval output, got {type(result)}')
        out = result.detach().float().cpu()
        if not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite model prediction')
        for key in ('native', 'adapted'):
            if key not in captures or not torch.isfinite(captures[key]).all():
                raise RuntimeError(f'Missing or invalid feature capture: {key}')
        return out, captures['native'].clone(), captures['adapted'].clone()

    try:
        dg, base_native, base_adapted = forward(x, False)
        tta, adapted_native, adapted_features = forward(x, True)
        dg_pair, _, _ = forward(torch.cat((x, y), dim=0), False)
        tta_pair, _, _ = forward(torch.cat((x, y), dim=0), True)
        tta_rev, _, _ = forward(torch.cat((y, x), dim=0), True)
    finally:
        h0.remove(); h1.remove()

    changed = [name for name, value in model.state_dict().items()
               if not torch.equal(value.detach().cpu(), snapshot[name])]
    report = {
        'checkpoint': str(args.checkpoint), 'weights': args.weights,
        'checkpoint_start_epoch': ck_epoch,
        'images': [args.image_index, args.other_index],
        'input': list(x.shape), 'amp_fp16': args.fp16,
        'dg_native_to_dark3': deltas(base_adapted, base_native),
        'tta_native_to_dark3': deltas(adapted_features, adapted_native),
        'ttt_feature_effect': deltas(adapted_features, base_adapted),
        'ttt_prediction_effect': deltas(tta, dg),
        'dg_batch_composition_difference': deltas(dg_pair[:1], dg),
        'tta_batch_composition_difference': deltas(tta_pair[:1], tta),
        'tta_order_difference': deltas(tta_rev[1:2], tta),
        'state_mutations': changed,
        'dataset': annotation_report(exp.val_ann),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if changed:
        raise RuntimeError(f'In-place model-state mutation detected: {changed[:8]}')
    if report['tta_batch_composition_difference']['relative_l2'] > max(
            1.e-4, 5 * report['dg_batch_composition_difference']['relative_l2']):
        print('DIAGNOSTIC WARNING: TTT appears batch-composition-dependent; compare paired conditions')
    else:
        print('DIAGNOSTIC: no large TTT-specific batch dependence on these two samples')


if __name__ == '__main__':
    main()
