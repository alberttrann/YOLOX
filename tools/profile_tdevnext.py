"""Measured CUDA latency: same checkpoint, native shape, DG vs TTA.

No precision or TTT policy selected by condition identity.
Run:
 python tools/profile_tdevnext.py --model-exp exps/default/tde_yolox_s.py \
    --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth --height 576 --width 1024
Measure deployment hardware separately; laptop GPU results are not edge FPS.
"""
import argparse
import statistics
import torch
from yolox.exp import get_exp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-exp', default='exps/default/tde_yolox_s.py')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--weights', choices=['raw','ema'], default='ema')
    p.add_argument('--height', type=int, default=576)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--runs', type=int, default=50)
    p.add_argument('--fp16', action='store_true')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required for accurate event timing')
    exp = get_exp(a.model_exp, None)
    model = exp.get_model().cuda().eval()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if a.weights == 'ema':
        if ck.get('ema_model') is None:
            raise RuntimeError('Checkpoint does not include EMA weights')
        state = ck['ema_model']
    else:
        state = ck['model']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()}, strict=True)
    model.set_meta_training_state(max(1, int(ck.get('start_epoch',1))), exp.max_epoch)
    x=torch.randn(a.batch_size,3,a.height,a.width, device='cuda', dtype=torch.float32)
    # Deliberately no deploy folding: compare identical checkpoint semantics.
    for mode in ('dg', 'tta'):
        model.eval_ttt_enabled = mode == 'tta'
        samples=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(a.warmup+a.runs):
            begin=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            # no_grad, not inference_mode: TTT creates a local autograd island.
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=a.fp16):
                    begin.record()
                    _=model(x)
                    end.record()
            end.synchronize()
            if step>=a.warmup:
                samples.append(begin.elapsed_time(end))
        print(f'MODE={mode} batch={a.batch_size} input={a.height}x{a.width} fp16={a.fp16} '
              f'forward_mean_ms={statistics.mean(samples):.3f} '
              f'forward_median_ms={statistics.median(samples):.3f} '
              f'forward_p95_ms={sorted(samples)[int(0.95*(len(samples)-1))]:.3f} '
              f'peak_allocated_mb={torch.cuda.max_memory_allocated()/2**20:.1f} '
              'EXCLUDES=preprocess,nms,transfers')

if __name__ == '__main__':
    main()
