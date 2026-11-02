
"""tdevNext vNext.2: real-BDD, checkpoint-loaded AMP vs FP32 forensic audit.

Run from YOLOX root after copying to tools/audit_vnext2_real_amp.py:
    python tools/audit_vnext2_real_amp.py --checkpoint weights/yolox_s.pth --batch-size 2 --steps 8
    python tools/audit_vnext2_real_amp.py --checkpoint weights/yolox_s.pth --batch-size 2 --steps 8 --fp32
    python tools/audit_vnext2_real_amp.py --checkpoint weights/yolox_s.pth --batch-size 2 --steps 1 --trace-modules

No checkpoint is saved. Non-finite forward loss fails with per-component diagnostics.
AMP gradient overflow can be skipped via GradScaler only after a finite forward.
"""
import argparse
import math
import traceback
from contextlib import nullcontext

import torch
from yolox.exp import get_exp

LOSS_KEYS = ("total_loss", "det_loss", "aux_total", "iou_loss", "conf_loss",
             "cls_loss", "mem_loss", "proto_loss", "intra_loss", "rec_loss",
             "afford_loss", "num_fg", "num_fg_total", "num_gt_total",
             "ttt_prob", "aux_factor")
SNAPSHOT_KEYS = (
    'head.cls_preds.0.weight', 'head.obj_preds.0.weight',
    'backbone.backbone.dark2.residual_scale',
    'backbone.C3_p3.gate', 'backbone.C3_p3.global_mixer.3.qkv.weight',
    'backbone.spi_highway.router.3.weight',
    'head.memory_banks.0.prototypes',
)


def scalar_float(value):
    """Handle PyTorch scalar tensors, Python ints and floats safely."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar, got {tuple(value.shape)}")
        return float(value.detach().float().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Unsupported output scalar type: {type(value).__name__}")


def loss_report(outputs):
    return {k: scalar_float(outputs[k]) for k in LOSS_KEYS if k in outputs}


def load_compatible_pretrained(model, checkpoint):
    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        # Legacy checkpoint content may require weights_only=False; only load your own trusted checkpoints.
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
    original = {k.removeprefix('module.'): v for k, v in original.items()}
    candidates = dict(original)
    for key, value in original.items():
        if 'dark2.' in key and 'dark2.backbone_stage.' not in key:
            candidates[key.replace('dark2.', 'dark2.backbone_stage.')] = value
    target = model.state_dict()
    matched = {k: v for k, v in candidates.items()
               if k in target and torch.is_tensor(v) and target[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    matched_elements = sum(target[k].numel() for k in matched)
    total_elements = sum(v.numel() for v in target.values())
    print(f'CHECKPOINT_LOADED matched_keys={len(matched)} state_elements={matched_elements:,}/{total_elements:,} '
          f'({matched_elements / max(1, total_elements):.2%})', flush=True)
    if not any(k.startswith('backbone.backbone.dark2.backbone_stage.') for k in matched):
        raise RuntimeError('Native dark2 not loaded; refuse to audit uninitialized backbone.')


def prepare_real_loader(exp, batch_size):
    exp.data_num_workers = 0
    loader = exp.get_data_loader(batch_size=batch_size, is_distributed=False,
                                 no_aug=False, cache_img=False)
    return loader, iter(loader)


def sample_batch(iterator, device, exp):
    batch = next(iterator)
    images, targets = batch[:2]
    images = images.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    images, targets = exp.preprocess(images, targets, exp.input_size)
    return images, targets


def iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_tensors(item)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from iter_tensors(item)


def finite_gradient_report(named_params):
    bad, nonzero = [], 0
    for name, p in named_params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all().item():
            bad.append(name)
        if torch.count_nonzero(p.grad).item():
            nonzero += 1
    return bad, nonzero


def parameter_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in SNAPSHOT_KEYS}


def parameter_deltas(model, previous):
    current = dict(model.named_parameters())
    return {k: float(torch.linalg.vector_norm((current[k].detach() - v).float()).item())
            for k, v in previous.items()}


class ModuleForwardTracer:
    """Record first observed nonfinite module outputs; functional ops need finer probes."""
    def __init__(self, model, max_events=30):
        self.events = []
        self.last_finite = []
        self.handles = []
        self.max_events = max_events
        self.observed = 0
        for name, module in model.named_modules():
            if not name or any(True for _ in module.children()):
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, args, output):
            for tensor in iter_tensors(output):
                if not tensor.is_floating_point() or tensor.numel() == 0:
                    continue
                self.observed += 1
                is_finite = bool(torch.isfinite(tensor).all().item())
                if not is_finite:
                    if len(self.events) < self.max_events:
                        invalid = int((~torch.isfinite(tensor)).sum().item())
                        self.events.append((name, tuple(tensor.shape), str(tensor.dtype), invalid))
                else:
                    self.last_finite.append(name)
                    if len(self.last_finite) > 8:
                        self.last_finite.pop(0)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summarize(self, label):
        print(f'{label} MODULE_TRACE observed_tensors={self.observed} nonfinite_events={len(self.events)}', flush=True)
        if self.events:
            print(f'{label} FIRST_NONFINITE_MODULE_OUTPUTS {self.events[:self.max_events]}', flush=True)
        print(f'{label} LAST_FINITE_MODULE_OUTPUTS {self.last_finite}', flush=True)
        print('NOTE: a forward hook identifies the first *observed module output*, not the exact functional operation.', flush=True)


def run_forward(model, images, targets, amp, amp_dtype, trace=False):
    tracer = ModuleForwardTracer(model) if trace else None
    try:
        with torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype):
            outputs = model(images, targets)
        return outputs, tracer, None
    except Exception as exc:
        return None, tracer, exc


def report_forward(outputs, label):
    if outputs is None:
        return False
    vals = loss_report(outputs)
    print(f'{label} LOSSES ' + ' '.join(
        f'{k}={v:.7g}' for k, v in vals.items()
    ), flush=True)
    bad = [k for k, v in vals.items() if not math.isfinite(v)]
    if bad:
        print(f'{label} NONFINITE_LOSSES {bad}', flush=True)
    return len(bad) == 0


def compare_fp32_after_amp_failure(model, images, targets, state, cpu_rng, gpu_rng, trace):
    """Use same batch, same pre-forward BN buffers and RNG without modifying trained weights."""
    with torch.no_grad():
        model.load_state_dict(state, strict=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    print('COMPARISON same batch/checkpoint/pre-forward buffers/RNG, FP32 autocast disabled', flush=True)
    outputs, tracer, exc = run_forward(model, images, targets, False, torch.float16, trace=trace)
    if exc is not None:
        print(f'FP32_COMPARISON_EXCEPTION {type(exc).__name__}: {exc}', flush=True)
    else:
        report_forward(outputs, 'FP32_COMPARISON')
    if tracer is not None:
        tracer.summarize('FP32_COMPARISON')
        tracer.close()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this real-GPU audit')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    exp = get_exp(args.exp, None)
    model = exp.get_model()
    load_compatible_pretrained(model, args.checkpoint)
    model.to(device).train()
    model.set_meta_training_state(1, exp.max_epoch)

    optimizer = exp.get_optimizer(args.batch_size)
    for group in optimizer.param_groups:
        group['lr'] = args.lr
    loader, loader_iter = prepare_real_loader(exp, args.batch_size)
    amp = not args.fp32
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp and not args.bf16,
                                       init_scale=args.init_scale, growth_interval=2000)
    success, skipped = 0, 0
    print(f'AUDIT_SETTINGS real_images=True shape={exp.input_size} '
          f'dtype={"fp32" if not amp else str(amp_dtype)} batch={args.batch_size} steps={args.steps} '
          f'lr={args.lr} initial_scale={scaler.get_scale()} trace={args.trace_modules}', flush=True)
    print('INITIAL_GPU_MEMORY_ALLOCATED_MB', round(torch.cuda.memory_allocated() / 2**20, 1), flush=True)

    for index in range(args.steps):
        try:
            images, targets = sample_batch(loader_iter, device, exp)
        except StopIteration:
            loader_iter = iter(loader)
            images, targets = sample_batch(loader_iter, device, exp)

        optimizer.zero_grad(set_to_none=True)
        # Diagnostic restores precise preforward buffers for same-batch FP32 comparison.
        state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                 if args.compare_on_failure and amp else None)
        cpu_rng = torch.get_rng_state() if state is not None else None
        gpu_rng = torch.cuda.get_rng_state() if state is not None else None

        outputs, tracer, exc = run_forward(model, images, targets, amp, amp_dtype, trace=args.trace_modules)
        if exc is not None:
            print(f'STEP {index + 1} FORWARD_EXCEPTION {type(exc).__name__}: {exc}', flush=True)
            if tracer is not None:
                tracer.summarize('AMP' if amp else 'FP32')
                tracer.close()
            if state is not None:
                compare_fp32_after_amp_failure(model, images, targets, state,
                                               cpu_rng, gpu_rng, trace=args.trace_modules)
            raise exc
        good = report_forward(outputs, f'STEP {index + 1} {"AMP" if amp else "FP32"}')
        if tracer is not None:
            tracer.summarize('AMP' if amp else 'FP32')
            tracer.close()
        if not good:
            if state is not None:
                compare_fp32_after_amp_failure(model, images, targets, state,
                                               cpu_rng, gpu_rng, trace=args.trace_modules)
            raise FloatingPointError(f'Nonfinite FORWARD component(s) at step {index+1}')
        del state

        loss = outputs['total_loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        invalid, nonzero = finite_gradient_report(model.named_parameters())
        if invalid and (not amp or args.bf16):
            raise FloatingPointError(f'Unscaled backward nonfinite without active GradScaler at step {index + 1}: {invalid[:25]}')
        grad_norm = (float('nan') if invalid else
                     float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0,
                                                            error_if_nonfinite=True).item()))
        before = parameter_snapshot(model)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        did_step = (not amp or args.bf16 or new_scale >= old_scale)
        if invalid and did_step:
            raise RuntimeError('Non-finite gradient but optimizer update was not skipped')
        if did_step:
            success += 1
            with torch.no_grad():
                if hasattr(model, 'project_prototypes_to_unit_sphere'):
                    model.project_prototypes_to_unit_sphere()
        else:
            skipped += 1
        delta = parameter_deltas(model, before)
        if not did_step and any(v > 0 for v in delta.values()):
            raise AssertionError('Parameters updated despite scaler skip')
        print(f'STEP {index+1}/{args.steps} stepped={did_step} '
              f'scale={old_scale:g}->{new_scale:g} bad_grads={len(invalid)} '
              f'nonzero_grad_tensors={nonzero} grad_norm={grad_norm:.7g} '
              f'memory_mb={torch.cuda.memory_allocated()/2**20:.1f}', flush=True)
        if invalid:
            print('INVALID_GRADIENT_PARAMETERS', invalid[:30], flush=True)
        if index in (0, args.steps - 1):
            print('WEIGHT_DELTAS', delta, flush=True)

    print(f'FINAL successful_optimizer_steps={success} amp_skipped_steps={skipped} '
          f'peak_gpu_allocated_mb={torch.cuda.max_memory_allocated()/2**20:.1f}', flush=True)
    if success == 0:
        raise RuntimeError('Zero successful optimizer steps')
    if skipped >= max(3, math.ceil(args.steps / 2)):
        raise RuntimeError('Persistent AMP skips: inspect backward gradients')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='exps/default/tde_yolox_s.py')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--init-scale', type=float, default=1024.0)
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--trace-modules', action='store_true',
                    help='GPU-synchronized checks of all leaf-module outputs (slow, for debugging)')
    ap.add_argument('--compare-on-failure', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--seed', type=int, default=7119)
    args = ap.parse_args()
    if args.fp32 and args.bf16:
        ap.error('--fp32 and --bf16 are mutually exclusive')
    run(args)


if __name__ == '__main__':
    main()
