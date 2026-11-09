"""P2 high-resolution CUDA forward-only timing and memory matrix.

Explicitly excludes preprocessing, host transfers, decode/NMS and rendering;
end-to-end deployment must also be benchmarked on the target device.
"""
import argparse
import json
import statistics
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from yolox.exp import get_exp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--weights',choices=['ema','model'],default='ema')
    p.add_argument('--sizes',nargs='+',default=['576x1024','768x768','1088x1920','1536x2560','2176x3840'])
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=40)
    p.add_argument('--fp16',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for GPU profile')
    exp=get_exp(a.model_exp,None)
    if getattr(exp,'a4_filter_basis',None) is not None:
        from a4_verify_checkpoint import verify
        verify(a.model_exp,a.checkpoint)
    m=exp.get_model().cuda().eval()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    key='ema_model' if a.weights=='ema' else 'model'
    if key not in ck or not isinstance(ck[key],dict):raise RuntimeError('Missing '+key+' state')
    m.load_state_dict({k.removeprefix('module.'):v for k,v in ck[key].items()},strict=True)
    output=[]
    for size in a.sizes:
        h,w=[int(t) for t in size.split('x')]
        if h%32 or w%32:raise ValueError('Sizes must be stride-32 aligned')
        x=None
        try:
            x=torch.randn(a.batch_size,3,h,w,device='cuda')
            torch.cuda.reset_peak_memory_stats()
            timings=[]
            with torch.inference_mode():
                for _ in range(a.warmup):
                    with torch.autocast('cuda',enabled=a.fp16):m(x)
                torch.cuda.synchronize()
                for _ in range(a.repeats):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.autocast('cuda',enabled=a.fp16):m(x)
                    end.record();end.synchronize();timings.append(start.elapsed_time(end))
            sorted_times=sorted(timings)
            report={'size':size,'batch':a.batch_size,'fp16':a.fp16,
                    'mean_ms':statistics.mean(timings),
                    'p50_ms':statistics.median(timings),
                    'p95_ms':sorted_times[int(.95*(len(sorted_times)-1))],
                    'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'status':'PASS','scope':'model_forward_only'}
        except torch.cuda.OutOfMemoryError as exc:
            report={'size':size,'status':'OOM','error':str(exc)}
        output.append(report);print(report,flush=True)
        if x is not None: del x
        torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps({'model_exp':a.model_exp,'results':output},indent=2),encoding='utf-8')


if __name__=='__main__':main()
