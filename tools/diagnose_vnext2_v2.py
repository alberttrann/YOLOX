#!/usr/bin/env python3
"""Read-only vNext.2 TTT diagnosis: native/static/adapted, inner step, matched-batch controls.

Requires vNext.2 + 01_eval_only.patch. This module installs no persistent hooks,
changes no files in the repo except the requested JSON report, and never optimizes.

Example (PowerShell):
 python tools/diagnose_vnext2_v2.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py `
  --data-exp exps/default/test_acdc_fog.py --weights ema `
  --image-index 0 --other-index 1 `
  --output YOLOX_outputs/vnext2_ttt_fog_v2.json
"""
import argparse
import json
import math
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

# Keep import lazy so pure math tests do not require the full YOLOX installation.


def tensor_delta(a, b):
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {tuple(a.shape)} / {tuple(b.shape)}')
    af, bf = a.detach().double().reshape(-1), b.detach().double().reshape(-1)
    if af.numel() == 0:
        return {'max_abs': 0.0, 'mean_abs': 0.0, 'rms_abs': 0.0, 'relative_l2': 0.0}
    d = af - bf
    return {
        'max_abs': float(d.abs().max()),
        'mean_abs': float(d.abs().mean()),
        'rms_abs': float(d.square().mean().sqrt()),
        'relative_l2': float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(bf).clamp_min(1e-12)),
    }



def matched_batch_controls(f_aa, f_ab, f_ba, f_bb):
    """Same-image matched-slot controls for A and B.

    aa=[A,A], ab=[A,B], ba=[B,A], bb=[B,B].
    Each comparison must reference the SAME image in BOTH tensors.
    """
    return {
        'A_first_slot_companion_effect': tensor_delta(f_ab['predictions'][:1], f_aa['predictions'][:1]),
        'A_permutation_effect': tensor_delta(f_ba['predictions'][1:2], f_ab['predictions'][:1]),
        'B_second_slot_companion_effect': tensor_delta(f_ab['predictions'][1:2], f_bb['predictions'][1:2]),
        'B_permutation_effect': tensor_delta(f_ab['predictions'][1:2], f_ba['predictions'][:1]),
        'A_first_slot_dark2_effect': tensor_delta(f_ab['features'][:1], f_aa['features'][:1]),
        'A_permutation_dark2_effect': tensor_delta(f_ba['features'][1:2], f_ab['features'][:1]),
        'B_second_slot_dark2_effect': tensor_delta(f_ab['features'][1:2], f_bb['features'][1:2]),
        'B_permutation_dark2_effect': tensor_delta(f_ab['features'][1:2], f_ba['features'][:1]),
    }


def tensor_summary(t):
    x = t.detach().float()
    return {'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean()),
            'rms': float(x.square().mean().sqrt()), 'l2': float(torch.linalg.vector_norm(x))}


def annotation_report(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    images, anns = doc.get('images', []), doc.get('annotations', [])
    counts = Counter(a.get('image_id') for a in anns if not a.get('ignore', False))
    classes = Counter(a.get('category_id') for a in anns if not a.get('ignore', False))
    hist = sorted(counts.get(i['id'], 0) for i in images)
    def pct(q):
        return hist[min(len(hist)-1, max(0, math.ceil(q*len(hist))-1))] if hist else 0
    return {
        'annotation_file': str(path), 'images': len(images), 'annotations': len(anns),
        'crowd_annotations': sum(int(a.get('iscrowd', 0)) for a in anns),
        'objects_per_image': {'max': max(hist, default=0), 'p50': pct(.5), 'p90': pct(.9), 'p99': pct(.99)},
        'images_without_annotations': sum(n == 0 for n in hist),
        'class_support': [{'name': c.get('name', str(c['id'])), 'category_id': c['id'],
                           'num_annotations': classes.get(c['id'], 0)} for c in doc.get('categories', [])],
    }


def make_static_adapter(dark2):
    """Proxy *only* during diagnostic forward. Does not change parameters or source."""
    if not dark2.residual_adapter:
        raise RuntimeError('Native/static/TTT comparison requires residual_adapter=True')
    def f(native_features, run_ttt=True):
        if not run_ttt:
            return native_features
        v = native_features.float()
        normalized = F.group_norm(v, 8, eps=1e-5)
        result = v + dark2.residual_gain * (
            dark2.residual_scale.detach().float().tanh() * normalized
            + dark2.residual_bias.detach().float().tanh())
        return result.to(native_features.dtype)
    return f


def inner_step_probe(dark2, native_features):
    """Independent *non-mutating* replica of vNext.2's FP32 residual inner step.

    Always run on a single image to isolate batch coupling. Replica fidelity is
    checked against actual model TTT features in main; do not infer a correct
    replica if the installed TTT implementation has changed.
    """
    if native_features.shape[0] != 1:
        raise ValueError('inner_step_probe expects one independent image')
    with torch.enable_grad(), torch.autocast(device_type=native_features.device.type, enabled=False):
        source = native_features.detach().float()
        with torch.no_grad():
            target = dark2._get_noise_clipped_target(source)
            mask = dark2._get_relative_contrast_mask(target, dark2.mask_ratio).float()
            normalized = F.group_norm(source, 8, eps=1e-5)
        # Test-time fast weights *must* be leaves; EMA persistent parameters are frozen.
        scale = dark2.residual_scale.detach().float().clone().requires_grad_(True)
        bias = dark2.residual_bias.detach().float().clone().requires_grad_(True)
        token = dark2.projector.mask_token.detach().float()
        def corrected(s, b):
            return source + dark2.residual_gain * (s.tanh() * normalized + b.tanh())
        def objective(s, b):
            # Evaluation noise=0; this is NOT the outer training objective.
            corruption = corrected(s, b) * (1.0-mask) + token*mask
            prediction = dark2.projector(corruption)
            return F.mse_loss(prediction.float(), target)
        before = objective(scale, bias)
        gs, gb = torch.autograd.grad(before, (scale, bias), create_graph=False)
        if not bool(torch.isfinite(gs).all() and torch.isfinite(gb).all()):
            raise FloatingPointError('Non-finite TTT inner gradient')
        grad_norm = torch.sqrt(gs.detach().square().sum() + gb.detach().square().sum())
        clip_factor = float((dark2.residual_grad_clip / grad_norm.clamp_min(1e-12)).clamp(max=1.0))
        fraction = min(1., max(0., dark2.current_epoch/max(1, dark2.max_epochs)))
        ceiling = dark2.min_meta_lr + (dark2.base_max_lr-dark2.min_meta_lr)*(
            .5*(1. + math.cos(math.pi*fraction)))
        lr_s = dark2.min_meta_lr + (ceiling-dark2.min_meta_lr)*torch.sigmoid(dark2.ttt_lrs['residual_scale'])
        lr_b = dark2.min_meta_lr + (ceiling-dark2.min_meta_lr)*torch.sigmoid(dark2.ttt_lrs['residual_bias'])
        ns = scale - lr_s.detach()*gs.detach()*clip_factor
        nb = bias - lr_b.detach()*gb.detach()*clip_factor
        after = objective(ns, nb)
        f_static = corrected(scale, bias)
        f_adapt = corrected(ns, nb)
        return {
            'inner_loss_before': float(before.detach()),
            'inner_loss_after': float(after.detach()),
            'absolute_inner_loss_reduction': float((before-after).detach()),
            'relative_inner_loss_reduction': float(((before-after) / before.detach().clamp_min(1e-12)).detach()),
            'gradient_scale_l2': float(torch.linalg.vector_norm(gs.detach())),
            'gradient_bias_l2': float(torch.linalg.vector_norm(gb.detach())),
            'gradient_total_l2': float(grad_norm.detach()),
            'gradient_clip_factor': clip_factor,
            'effective_lr_scale': float(lr_s.detach()),
            'effective_lr_bias': float(lr_b.detach()),
            'fast_scale_step_l2': float(torch.linalg.vector_norm((ns-scale).detach())),
            'fast_bias_step_l2': float(torch.linalg.vector_norm((nb-bias).detach())),
            'persistent_scale': tensor_summary(scale),
            'persistent_bias': tensor_summary(bias),
            'mask_coverage_fraction': float(mask.detach().mean()),
            'static_to_adapted_feature': tensor_delta(f_adapt, f_static),
            'source_to_static_feature': tensor_delta(f_static, source),
            'source_to_adapted_feature': tensor_delta(f_adapt, source),
            # Returned for comparing to *real* patched TTT pathway (not JSON).
            '_reconstructed_ttt_feature': f_adapt.detach().to(native_features.dtype),
        }


def gt_boxes_for_index(dataset, index, input_shape):
    """Official YOLOX ValTransform's top-left letterbox geometry, for diagnostic ONLY.

    GT is read from original COCO boxes and projected by min(targetH/H,targetW/W).
    If your fork uses centered padding/crops, replace this with its recorded affine
    transform; the script calls this out so local metrics cannot silently mislead.
    """
    if not hasattr(dataset, 'coco') or not hasattr(dataset, 'ids'):
        return {'supported': False, 'reason': 'dataset lacks COCO ids metadata', 'boxes': []}
    image_id = dataset.ids[index]
    meta = dataset.coco.loadImgs([image_id])[0]
    h, w = int(meta['height']), int(meta['width'])
    if h <= 0 or w <= 0:
        return {'supported': False, 'reason': 'bad COCO metadata size', 'boxes': []}
    ih, iw = input_shape[-2:]
    scale = min(ih/h, iw/w)
    boxes = []
    for ann in dataset.coco.imgToAnns.get(image_id, []):
        if ann.get('ignore', False) or ann.get('iscrowd', 0):
            continue
        x, y, bw, bh = [float(v) for v in ann['bbox']]
        if bw <= 0 or bh <= 0:
            continue
        boxes.append({'category_id': int(ann['category_id']), 'bbox_input_xyxy': [
            x*scale, y*scale, (x+bw)*scale, (y+bh)*scale],
            'area_fraction_input': bw*bh*scale*scale / (ih*iw), 'id': ann.get('id')})
    return {'supported': True, 'image_id': int(image_id), 'source_hw': [h,w],
            'scale': scale, 'geometry_assumption': 'YOLOX top-left resize/letterbox; no crop',
            'boxes': boxes}


def spatial_effects(base, altered, gt_info, max_boxes=50):
    """Per-GT and background corrections on dark2 grid; does not claim causality."""
    if base.ndim != 4 or base.shape[0] != 1 or base.shape != altered.shape:
        raise ValueError('Expected matching [1,C,H,W] feature tensors')
    if not gt_info.get('supported'):
        return {'supported': False, 'reason': gt_info.get('reason')}
    bh, bw = gt_info['source_hw']
    _ = (bh, bw)
    fh, fw = base.shape[2:]
    # Input shape from actual fixed ValTransform, not image metadata.
    image_h, image_w = gt_info['canvas_hw']
    union = torch.zeros((fh, fw), dtype=torch.bool)
    per_gt = []
    for obj in gt_info['boxes'][:max_boxes]:
        x1, y1, x2, y2 = obj['bbox_input_xyxy']
        left = max(0, min(fw-1, int(math.floor(x1*fw/image_w))))
        right = max(left+1, min(fw, int(math.ceil(x2*fw/image_w))))
        top = max(0, min(fh-1, int(math.floor(y1*fh/image_h))))
        bottom = max(top+1, min(fh, int(math.ceil(y2*fh/image_h))))
        mask = torch.zeros((fh,fw), dtype=torch.bool)
        mask[top:bottom, left:right] = True
        union |= mask
        delta = tensor_delta(altered[..., mask], base[..., mask])
        per_gt.append({'id': obj['id'], 'category_id': obj['category_id'],
                       'area_fraction_input': obj['area_fraction_input'],
                       'feature_cells': int(mask.sum()), 'effect': delta})
    results = {'supported': True, 'gt_count': len(gt_info['boxes']), 'reported_gt_count': len(per_gt),
               'gt_regions': per_gt, 'foreground_union_fraction': float(union.float().mean())}
    if bool(union.any()):
        results['foreground_union_effect'] = tensor_delta(altered[...,union], base[...,union])
    if bool((~union).any()):
        results['background_effect'] = tensor_delta(altered[...,~union], base[...,~union])
    return results


def summarize_postprocess(result):
    if result is None or result.numel() == 0:
        return {'count': 0, 'classes': {}, 'mean_joint_conf': None, 'detections': []}
    d = result.detach().float().cpu()
    conf = d[:,4]*d[:,5]
    class_counts = Counter(str(int(cls)) for cls in d[:,6])
    return {'count': len(d), 'classes': dict(class_counts),
            'mean_joint_conf': float(conf.mean()),
            'detections': d.tolist()}


def match_detections(a, b, min_iou=0.5):
    """Pair nearest same-class detections across modes; not ground-truth AP."""
    aa, bb = a['detections'], b['detections']
    matches, used = [], set()
    for i, det in enumerate(aa):
        x1,y1,x2,y2 = det[:4]
        candidate = None
        for j, od in enumerate(bb):
            if j in used or round(det[6]) != round(od[6]):
                continue
            X1,Y1,X2,Y2 = od[:4]
            inter = max(0.,min(x2,X2)-max(x1,X1))*max(0.,min(y2,Y2)-max(y1,Y1))
            area1 = max(0.,x2-x1)*max(0.,y2-y1)
            area2 = max(0.,X2-X1)*max(0.,Y2-Y1)
            iou = inter/max(1e-12,area1+area2-inter)
            if iou >= min_iou and (candidate is None or iou > candidate[0]):
                candidate = (iou,j,od)
        if candidate is not None:
            iou,j,od = candidate
            used.add(j)
            matches.append({'iou': iou, 'score_change': od[4]*od[5]-det[4]*det[5],
                            'box_max_coord_change': max(abs(v-w) for v,w in zip(det[:4],od[:4]))})
    return {'a_count': len(aa), 'b_count': len(bb), 'paired': len(matches),
            'a_unpaired': len(aa)-len(matches), 'b_unpaired': len(bb)-len(matches),
            'mean_pair_iou': (sum(x['iou'] for x in matches)/len(matches)) if matches else None,
            'mean_score_change': (sum(x['score_change'] for x in matches)/len(matches)) if matches else None,
            'max_box_coord_change': max((x['box_max_coord_change'] for x in matches), default=None)}


def get_image(dataset, index, device):
    sample = dataset[index]
    tensor = torch.as_tensor(sample[0], dtype=torch.float32, device=device)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f'Expected CHW, got {tuple(tensor.shape)}')
    return tensor.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-exp', default='exps/default/tde_yolox_s.py')
    parser.add_argument('--data-exp', required=True)
    parser.add_argument('--weights', choices=('ema','raw'), default='ema')
    parser.add_argument('--image-index', type=int, default=0)
    parser.add_argument('--other-index', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='Separately assess FP16; use FP32 for matching existing ACDC runs')
    parser.add_argument('--conf', type=float, default=0.01)
    parser.add_argument('--nms', type=float, default=None)
    parser.add_argument('--max-gt-details', type=int, default=50)
    parser.add_argument('--output')
    parser.add_argument('--verbose-json', action='store_true', help='Print full per-object JSON to console')
    args = parser.parse_args()
    if args.image_index == args.other_index:
        parser.error('Select different images A and B for the batch-composition test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.fp16 and device.type != 'cuda':
        parser.error('--fp16 requires CUDA')

    from yolox.exp import get_exp
    from yolox.utils import postprocess
    exp = get_exp(args.model_exp, None)
    data_exp = get_exp(args.data_exp, None)
    for key in ('data_dir','val_ann','test_ann','test_size'):
        setattr(exp, key, getattr(data_exp, key))
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ck.get('ema_model') if args.weights == 'ema' and isinstance(ck, dict) else None
    if args.weights == 'ema' and state is None:
        raise RuntimeError('Checkpoint has no EMA state')
    if args.weights == 'raw':
        state = ck.get('model', ck) if isinstance(ck, dict) else ck
    model = exp.get_model()
    model.load_state_dict({k[7:] if k.startswith('module.') else k:v for k,v in state.items()}, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    ck_epoch = int(ck.get('start_epoch', 1)) if isinstance(ck,dict) else 1
    if hasattr(model,'set_meta_training_state'):
        model.set_meta_training_state(max(1,ck_epoch), int(exp.max_epoch))
    if not hasattr(model,'eval_ttt_enabled'):
        raise RuntimeError('Requires evaluation protocol from 01_eval_only.patch')
    dark2 = model.backbone.backbone.dark2
    if not getattr(dark2,'residual_adapter',False):
        raise RuntimeError('Expected pretrained-compatible vNext.2 residual TTT')
    dataset = exp.get_eval_dataset(legacy=False)
    if len(dataset) <= max(args.image_index,args.other_index) or min(args.image_index,args.other_index)<0:
        raise IndexError(f'Image indices not within dataset length {len(dataset)}')
    x = get_image(dataset, args.image_index, device)
    y = get_image(dataset, args.other_index, device)
    if x.shape != y.shape:
        raise RuntimeError(f'ValTransform shapes differ {tuple(x.shape)} {tuple(y.shape)}')
    dark = model.backbone.backbone
    captured = {}
    def native_hook(_module,_args,out):
        captured['native'] = out.detach().cpu()
    def adapted_hook(_module,args):
        captured['dark3_input'] = args[0].detach().cpu()
    handles = (dark2.backbone_stage.register_forward_hook(native_hook),
               dark.dark3.register_forward_pre_hook(adapted_hook))
    state_before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    nms = float(exp.nmsthre if args.nms is None else args.nms)

    def run(batch, mode, do_post=False):
        captured.clear()
        model.eval_ttt_enabled = (mode != 'native')
        @contextmanager
        def override_static():
            if mode == 'static':
                with patch.object(dark2, 'forward_from_features', new=make_static_adapter(dark2)):
                    yield
            else:
                yield
        with override_static(), torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32,
                                enabled=args.fp16):
                predicted = model(batch)
        if not isinstance(predicted,torch.Tensor):
            raise RuntimeError(f'Model eval returned {type(predicted)} instead of decoded Tensor')
        if not bool(torch.isfinite(predicted).all()):
            raise FloatingPointError(f'Nonfinite decoded predictions in mode {mode}')
        if not all(k in captured for k in ('native','dark3_input')):
            raise RuntimeError('Missing dark2 native / dark3 pre-hook capture')
        if not all(bool(torch.isfinite(captured[k]).all()) for k in ('native','dark3_input')):
            raise FloatingPointError(f'Nonfinite feature capture in mode {mode}')
        result = {'predictions':predicted.detach().float().cpu(),
                  'native':captured['native'], 'features':captured['dark3_input']}
        if do_post:
            with torch.no_grad():
                detections = postprocess(predicted.detach().clone(), exp.num_classes, args.conf, nms)
            result['postprocess']=[summarize_postprocess(z) for z in detections]
        return result

    try:
        native = run(x,'native',True)
        static = run(x,'static',True)
        adapted = run(x,'adapted',True)
        # Same-shape, same-slot composition and permutation controls.
        aa = torch.cat([x,x],0)
        ab = torch.cat([x,y],0)
        ba = torch.cat([y,x],0)
        bb = torch.cat([y,y],0)
        controls={}
        for mode in ('native','static','adapted'):
            f_aa, f_ab, f_ba, f_bb=(run(t,mode) for t in (aa,ab,ba,bb))
            controls[mode] = matched_batch_controls(f_aa, f_ab, f_ba, f_bb)
        # Full analytic replica of the fast-weight step, for this ONE image.
        ttt_inner=inner_step_probe(dark2,native['native'].to(device))
        reconstructed=ttt_inner.pop('_reconstructed_ttt_feature').detach().float().cpu()
        replica_check=tensor_delta(reconstructed,adapted['features'])
        gt=gt_boxes_for_index(dataset,args.image_index,list(x.shape))
        gt['canvas_hw']=list(x.shape[-2:])
        foreground={
            'native_to_static':spatial_effects(native['features'],static['features'],gt,args.max_gt_details),
            'static_to_adapted':spatial_effects(static['features'],adapted['features'],gt,args.max_gt_details),
            'native_to_adapted':spatial_effects(native['features'],adapted['features'],gt,args.max_gt_details),
        }
    finally:
        for h in handles: h.remove()
        model.eval_ttt_enabled = False

    state_changed=[k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),state_before[k])]
    report={
        'protocol': {'checkpoint':str(args.checkpoint),'weights':args.weights,
                     'checkpoint_start_epoch':ck_epoch, 'model_exp':args.model_exp,
                     'data_exp':args.data_exp,'image_indices':[args.image_index,args.other_index],
                     'input_shape':list(x.shape),'fp16':args.fp16,
                     'class_count':int(exp.num_classes),'nms_threshold':nms,'conf_threshold':args.conf,
                     'interpretation':'Read-only diagnostic; never modifies optimizer, training source, or checkpoint'},
        'three_way_feature_effect': {
            'native_to_static':tensor_delta(static['features'],native['features']),
            'native_to_adapted':tensor_delta(adapted['features'],native['features']),
            'static_to_adapted_INNER_STEP_ONLY':tensor_delta(adapted['features'],static['features'])},
        'three_way_decoded_prediction_effect': {
            'native_to_static':tensor_delta(static['predictions'],native['predictions']),
            'native_to_adapted':tensor_delta(adapted['predictions'],native['predictions']),
            'static_to_adapted_INNER_STEP_ONLY':tensor_delta(adapted['predictions'],static['predictions'])},
        'inner_step_mechanics':ttt_inner,
        'inner_step_replica_vs_actual_ttt_feature':replica_check,
        'batch_controls_same_shape':controls,
        'object_conditioned_dark2_effect':foreground,
        'post_nms_differences':{
            'native_to_static':match_detections(native['postprocess'][0],static['postprocess'][0]),
            'native_to_adapted':match_detections(native['postprocess'][0],adapted['postprocess'][0]),
            'static_to_adapted':match_detections(static['postprocess'][0],adapted['postprocess'][0])},
        'post_nms_counts':{k:v['postprocess'][0]['count'] for k,v in (
            ('native',native),('static',static),('adapted',adapted))},
        'image_gt':{k:v for k,v in gt.items() if k!='boxes'},
        'dataset':annotation_report(exp.val_ann),
        'registered_state_mutations':state_changed,
        'warnings':[]}
    if replica_check['relative_l2'] > 1e-5:
        report['warnings'].append('Independent TTT step replica differs from actual TTT forward; inspect installed implementation before inferring inner-step fidelity.')
    if state_changed:
        report['warnings'].append('Unexpected persistent model weight/buffer mutation detected!')
    # Compare excess *at dark2*, not decoded outputs which have downstream CUDA jitter.
    dg_comp=controls['native']['A_first_slot_dark2_effect']['relative_l2']
    tta_comp=controls['adapted']['A_first_slot_dark2_effect']['relative_l2']
    if tta_comp > max(1e-6, 5*dg_comp):
        report['warnings'].append('TTT batch composition changes dark2 features beyond matched native control. Shared fast weights may be coupled across images.')
    if args.verbose_json:
        print(json.dumps(report,indent=2,allow_nan=False))
    else:
        compact={
            'checkpoint':report['protocol']['checkpoint'],
            'data_exp':report['protocol']['data_exp'],
            'image_indices':report['protocol']['image_indices'],
            'feature_relative_l2':{k:round(v['relative_l2'],9) for k,v in report['three_way_feature_effect'].items()},
            'prediction_relative_l2':{k:round(v['relative_l2'],9) for k,v in report['three_way_decoded_prediction_effect'].items()},
            'inner_step':{k:round(v,9) if isinstance(v,float) else v
                for k,v in report['inner_step_mechanics'].items()
                if k in ('inner_loss_before','inner_loss_after','relative_inner_loss_reduction',
                         'gradient_total_l2','gradient_clip_factor','effective_lr_scale',
                         'effective_lr_bias','fast_scale_step_l2','fast_bias_step_l2')},
            'replica_fidelity_relative_l2':report['inner_step_replica_vs_actual_ttt_feature']['relative_l2'],
            'batch_comp_dark2_relative_l2':{mode:val['A_first_slot_dark2_effect']['relative_l2']
                for mode,val in report['batch_controls_same_shape'].items()},
            'post_nms_counts':report['post_nms_counts'],
            'post_nms_static_to_ttt':report['post_nms_differences']['static_to_adapted'],
            'state_mutations':report['registered_state_mutations'],
            'warnings':report['warnings'],
        }
        print(json.dumps(compact,indent=2,allow_nan=False))
    if args.output:
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(f'SAVED_DIAGNOSTIC={path}')
    if state_changed:
        raise RuntimeError('The read-only evaluation unexpectedly mutated persistent state')
    if report['warnings']:
        for warning in report['warnings']:
            warnings.warn(warning)


if __name__=='__main__':
    main()
