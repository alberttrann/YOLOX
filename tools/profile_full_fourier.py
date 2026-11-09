"""Measured full-model FP16 inference latency on current CUDA device.

Reports forward latency only, not NMS/transfer/preprocessing or edge FPS.
Benchmark deployment engine separately on the target edge accelerator.
"""
import argparse
import statistics
import torch
from yolox.exp import get_exp

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--weights',choices=['raw','ema'],default='ema')
    p.add_argument('--height',type=int,default=576)
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--runs',type=int,default=50)
    p.add_argument('--warmup',type=int,default=12)
    p.add_argument('--fp16',action='store_true')
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for timing')
    exp=get_exp(a.model_exp,None)
    m=exp.get_model().cuda().eval()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=ck['ema_model' if a.weights=='ema' else 'model']
    if state is None:raise ValueError('Requested weight state missing')
    m.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    x=torch.randn(a.batch,3,a.height,a.width,device='cuda')
    samples=[];torch.cuda.reset_peak_memory_stats()
    for i in range(a.warmup+a.runs):
        a_evt=torch.cuda.Event(enable_timing=True);b_evt=torch.cuda.Event(enable_timing=True)
        with torch.no_grad(),torch.amp.autocast('cuda',enabled=a.fp16):
            a_evt.record();_ = m(x);b_evt.record()
        b_evt.synchronize()
        if i>=a.warmup:samples.append(a_evt.elapsed_time(b_evt))
    vals=sorted(samples)
    print('EXP',exp.exp_name,'BATCH',a.batch,'SIZE',(a.height,a.width),'AMP',a.fp16,
          'MEDIAN_MS',round(statistics.median(vals),3),'MEAN_MS',round(statistics.mean(vals),3),
          'P95_MS',round(vals[int(.95*(len(vals)-1))],3),
          'PEAK_MB',round(torch.cuda.max_memory_allocated()/2**20,1),
          'NOTE=forward-only, excludes_preprocess_NMS_transfers')

if __name__=='__main__':main()
