"""One-batch audit of detection losses, gradient conflicts and residual energy.

NO optimizer steps. By default uses a synthetic image/GT; --real-batch reads
one real data-loader batch. This is an audit sample, not an AP/AR conclusion.
"""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from yolox.exp import get_exp


def group_name(name):
    if 'backbone.backbone.dark2' in name:return 'native_dark2'
    if 'backbone.backbone.dark3' in name:return 'native_dark3'
    if 'backbone.backbone.dark4' in name:return 'native_dark4'
    if 'backbone.backbone.dark5' in name:return 'native_dark5'
    if 'backbone.spi_highway' in name:return 'SPI_P3_detail'
    if 'backbone.C3_p3' in name:return 'Tribrid_P3'
    if 'backbone.C3_p4' in name:return 'Tribrid_P4'
    if 'backbone.C3_n3' in name:return 'Tribrid_N4'
    if 'backbone.C3_n4' in name:return 'Tribrid_N5'
    if 'head.spectral_adapters' in name:return 'Fourier_P3_P5'
    if 'head.memory_banks' in name:return 'Engram_prototypes'
    if 'head.latent_projectors' in name or 'head.spatial_smoothers' in name:
        return 'Engram_query'
    if 'head.reg_preds' in name:return 'bbox_regression'
    if 'head.cls_preds' in name:return 'class_prediction'
    if 'head.obj_preds' in name:return 'objectness'
    return 'other'


def energy_ratio(probe):
    if probe is None:return None
    x, d = probe
    num=torch.linalg.vector_norm(d.float()).item()
    den=max(torch.linalg.vector_norm(x.float()).item(),1e-12)
    return num/den


def gradient_survey(model, output, sample_name):
    family_params={}
    params=[p for p in model.parameters() if p.requires_grad]
    id2name={id(p):n for n,p in model.named_parameters()}
    for index,p in enumerate(params):
        fam=group_name(id2name[id(p)])
        family_params.setdefault(fam,[]).append(index)
    terms={'regression':output['iou_loss'],
           'objectness':output['conf_loss'],
           'classification':output['cls_loss'],
           'prototype_aux':output['mem_loss']}
    if float(output['l1_loss'].detach()):terms['L1']=output['l1_loss']
    summaries={}; directional={}
    for key,loss in terms.items():
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        family={}
        for name,idxs in family_params.items():
            square=0.; all_finite=True
            for i in idxs:
                g=grads[i]
                if g is not None:
                    gf=g.detach().float()
                    square+=float((gf*gf).sum())
                    if not bool(torch.isfinite(gf).all()):all_finite=False
            family[name]={'grad_l2':square**.5,'finite':all_finite,
                          'trainable_tensors':len(idxs)}
        summaries[key]=family
        directional[key]=[g.detach().float().cpu() if g is not None else None for g in grads]
    cosines={}
    pairs=[('classification','regression'),('classification','objectness'),
           ('prototype_aux','classification'),('prototype_aux','regression')]
    for left,right in pairs:
        per_group={}
        for name,idxs in family_params.items():
            dot=0.;norm_l=0.;norm_r=0.
            for i in idxs:
                a=directional[left][i];b=directional[right][i]
                if a is None or b is None:continue
                dot+=float((a*b).sum());norm_l+=float((a*a).sum());norm_r+=float((b*b).sum())
            per_group[name]=dot/(math.sqrt(norm_l*norm_r)+1e-20) if norm_l*norm_r>0 else None
        cosines[left+'_vs_'+right]=per_group
    neck=model.backbone;head=model.head
    residuals={'SPI':energy_ratio(neck.spi_highway.last_probe)}
    for name in ('C3_p3','C3_p4','C3_n3','C3_n4'):
        residuals[name]=energy_ratio(getattr(neck,name).last_probe)
    for i,adapter in enumerate(head.spectral_adapters):
        residuals['spectral_P'+str(3+i)]=energy_ratio(adapter.last_probe)
    for i,probe in enumerate(head.memory_probes):
        residuals['Engram_P'+str(3+i)]=energy_ratio(probe)
    return {'sample':sample_name,'losses':{k:float(v.detach()) for k,v in terms.items()},
            'gradient_l2':summaries,'gradient_cosines':cosines,
            'residual_l2_ratios':residuals,
            'assignment':head.last_assignment_stats,
            'assignment_by_relative_size':head.last_coverage_by_size,
            'grounding':head.last_proto_diagnostic,
            'caveats':['single-batch gradients do not establish general complementarity',
                       'relative size bins are NOT COCO small/medium/large',
                       'no optimization, no AP/AR claim']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s_fourier_a41.py')
    p.add_argument('--checkpoint',help='Optional A4.1 checkpoint; raw model state only')
    p.add_argument('--weights',choices=['model','ema'],default='ema')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--real-batch',action='store_true')
    p.add_argument('--output',default='audits/a41_gradients.json')
    a=p.parse_args()
    e=get_exp(a.model_exp,None)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    m=e.get_model()
    if a.checkpoint:
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        key='ema_model' if a.weights=='ema' else 'model'
        state=saved.get(key) if isinstance(saved,dict) else None
        if not isinstance(state,dict):
            raise RuntimeError('Requested checkpoint weight kind unavailable: '+key)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        m.load_state_dict(state,strict=True)
    m=m.to(a.device).train()
    for name in ('C3_p3','C3_p4','C3_n3','C3_n4'):
        getattr(m.backbone,name).telemetry=True
    m.backbone.spi_highway.telemetry=True
    for adapter in m.head.spectral_adapters:adapter.telemetry=True
    m.head.capture_memory=True
    if a.real_batch:
        old_workers=e.data_num_workers;e.data_num_workers=0
        try:
            loader=e.get_data_loader(batch_size=2,is_distributed=False,no_aug=False,cache_img=False)
            x,y=next(iter(loader))[:2]
        finally:e.data_num_workers=old_workers
        x=x.to(a.device,dtype=torch.float32);y=y.to(a.device,dtype=torch.float32)
        sample='real_training_batch'
    else:
        h,w=128,192
        x=torch.randn(2,3,h,w,device=a.device)
        y=torch.zeros(2,16,5,device=a.device)
        y[0,0]=torch.tensor([0.,64.,48.,8.,12.],device=a.device)
        y[0,1]=torch.tensor([1.,90.,64.,70.,52.],device=a.device)
        y[1,0]=torch.tensor([2.,100.,80.,18.,20.],device=a.device)
        sample='synthetic_micro_medium_large'
    outputs=m(x,y)
    if not bool(torch.isfinite(outputs['total_loss'])):
        raise RuntimeError('Nonfinite forward result')
    report=gradient_survey(m,outputs,sample)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('A41_GRADIENT_SURVEY_PASS',a.output)
    print('RESIDUAL_L2_RATIOS',report['residual_l2_ratios'])
    print('ASSIGNMENT',report['assignment'])


if __name__=='__main__':main()
