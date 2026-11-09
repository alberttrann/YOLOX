"""Read-only parameter-utilization audit for epoch 1/2 EMA Fourier checkpoints.

A nonzero weight change or memory gate is NOT proof of detection benefit;
these diagnostics help identify dead branches before interpreting AP/AR.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from fourier_suite_metrics import VARIANTS


def stats_from_state(state):
    result={}
    for kind, pattern in (
        ('adapter_project',lambda k:k.startswith('head.spectral_adapters.') and '.project.weight' in k),
        ('adapter_gate',lambda k:k.startswith('head.spectral_adapters.') and '.gate.weight' in k),
        ('memory_proto',lambda k:k.startswith('head.memory_banks.') and '.prototypes' in k),
    ):
        values=[v.float().reshape(-1) for k,v in state.items() if pattern(k)]
        if not values:
            result[kind+'_norm']=None;result[kind+'_elements']=0
        else:
            combined=torch.cat(values)
            result[kind+'_norm']=float(torch.linalg.vector_norm(combined))
            result[kind+'_elements']=int(combined.numel())
    gate=state.get('head.mem_gate')
    result['engram_gate_raw']=float(gate.float().item()) if gate is not None else None
    result['engram_gate_tanh']=(math.tanh(result['engram_gate_raw']) if gate is not None else None)
    return result


def changes(before,after):
    vals={}
    for kind,part in (
        ('adapter_project','head.spectral_adapters.'),
        ('memory','head.memory_banks.'),
    ):
        keys=[k for k in before if k.startswith(part) and k in after and
              before[k].is_floating_point() and before[k].shape==after[k].shape]
        sq=0.0;num=0
        for k in keys:
            x=after[k].float()-before[k].float()
            sq+=float(x.square().sum());num+=x.numel()
        vals[kind+'_delta_l2']=sq**0.5 if keys else None
        vals[kind+'_delta_elements']=num
    if 'head.mem_gate' in before and 'head.mem_gate' in after:
        vals['engram_gate_delta']=float((after['head.mem_gate']-before['head.mem_gate']).item())
    else:
        vals['engram_gate_delta']=None
    return vals


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default='.')
    p.add_argument('--epoch-a',type=int,default=1)
    p.add_argument('--epoch-b',type=int,default=2)
    p.add_argument('--weights',choices=['ema','raw'],default='ema')
    args=p.parse_args(argv)
    root=Path(args.repo).resolve()
    output=root/'YOLOX_outputs/fourier_suite_reports'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,(variant,exp_name,_) in VARIANTS.items():
        folder=root/'YOLOX_outputs'/exp_name
        state_list=[]
        for epoch in (args.epoch_a,args.epoch_b):
            ckpt=folder/f'epoch_{epoch}_ckpt.pth'
            if not ckpt.is_file():
                print('MISSING',label,epoch,ckpt)
                state_list.append(None)
                continue
            # Trusted local checkpoints only, as torch.load deserializes Python.
            bundle=torch.load(str(ckpt),map_location='cpu',weights_only=False)
            state=bundle['ema_model' if args.weights=='ema' else 'model']
            if state is None:raise RuntimeError(f'{ckpt} missing {args.weights} state')
            state_list.append(state)
            row={'variant':label,'epoch':epoch,**stats_from_state(state)}
            rows.append(row)
            print('WEIGHTS',label,epoch,'adapter_project_norm',row['adapter_project_norm'],
                  'memory_gate_tanh',row['engram_gate_tanh'],'memory_proto_norm',row['memory_proto_norm'])
        if all(s is not None for s in state_list):
            row={'variant':label,'epoch':f'{args.epoch_a}->{args.epoch_b}',
                 **changes(*state_list)}
            rows.append(row)
            print('DELTA',label,'adapter_project_l2',row['adapter_project_delta_l2'],
                  'memory_l2',row['memory_delta_l2'],'gate_delta',row['engram_gate_delta'])
    (output/'weight_utilization.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('OUTPUT',output/'weight_utilization.json')


if __name__=='__main__':main()
