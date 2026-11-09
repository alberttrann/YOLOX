"""P1: actual object/background residual energy at P3/P4/P5.

Reads annotation boxes from original COCO GT. Assumes standard YOLOX
ValTransform top-left letterbox; confirm for a locally customized transform.
This is a diagnostic of residual utilization, not an object-detection metric.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from yolox.exp import get_exp


def summarize(x,r,bbox=None):
    if bbox is not None:
        x0,y0,x1,y1=bbox
        x=x[...,y0:y1,x0:x1];r=r[...,y0:y1,x0:x1]
    if x.numel()==0:return None
    xe=float(torch.sqrt(x.float().square().mean()+1e-12))
    re=float(torch.sqrt(r.float().square().mean()+1e-12))
    return re/(xe+1e-12)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--weights',choices=['ema','model'],default='ema')
    p.add_argument('--images',type=int,default=24)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    exp=get_exp(a.model_exp,None)
    if getattr(exp,'a4_filter_basis',None) is not None:
        from a4_verify_checkpoint import verify
        verify(a.model_exp,a.checkpoint)
    model=exp.get_model()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict) or a.weights not in ck or not isinstance(ck[a.weights],dict):
        # Historical format calls the EMA key 'ema_model'.
        src=ck.get('ema_model') if a.weights=='ema' and isinstance(ck,dict) else None
        if not isinstance(src,dict):raise RuntimeError('Checkpoint missing strict selected weights')
    else:src=ck[a.weights]
    model.load_state_dict({k.removeprefix('module.'):v for k,v in src.items()},strict=True)
    model=model.to(a.device).eval()
    captures={}
    handles=[]
    for level,module in enumerate(model.head.spectral_adapters):
        module.telemetry=True
        def capture(mod,inputs,output,level=level):
            captures[level]=(inputs[0].detach(),(output-inputs[0]).detach())
        handles.append(module.register_forward_hook(capture))
    if hasattr(model.head,'capture_memory'):model.head.capture_memory=True
    if hasattr(model.backbone.spi_highway,'telemetry'):
        model.backbone.spi_highway.telemetry=True
    ds=exp.get_eval_dataset();coco=ds.coco
    rows=[]
    for i in range(min(a.images,len(ds))):
        img,target,info,img_id=ds[i]
        if hasattr(img_id,'item'):img_id=int(img_id.item())
        elif isinstance(img_id,(list,tuple)):img_id=int(img_id[0])
        else:img_id=int(img_id)
        xx=torch.as_tensor(img,device=a.device).unsqueeze(0).float()
        with torch.inference_mode():model(xx)
        ann=coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        oh,ow=info[:2]
        ratio=min(xx.shape[-2]/float(oh),xx.shape[-1]/float(ow))
        objects=[]
        for z in ann:
            if z.get('iscrowd',0):continue
            x,y,w,h=z['bbox']
            area=float(z.get('area',w*h))
            bucket=('small' if area<32**2 else 'medium' if area<96**2 else 'large')
            objects.append((bucket,(x*ratio,y*ratio,(x+w)*ratio,(y+h)*ratio)))
        for level,adapter in enumerate(model.head.spectral_adapters):
            if level not in captures:continue
            original,correction=captures[level]
            _,_,fh,fw=original.shape
            sy=fh/xx.shape[-2];sx=fw/xx.shape[-1]
            for bucket,box in objects:
                l=max(0,min(fw-1,int(box[0]*sx)));t=max(0,min(fh-1,int(box[1]*sy)))
                rr=min(fw,max(l+1,int(box[2]*sx+0.999)))
                bb=min(fh,max(t+1,int(box[3]*sy+0.999)))
                value=summarize(original,correction,(l,t,rr,bb))
                rows.append({'image_id':img_id,'level':level,'bucket':bucket,
                             'relative_residual':value})
            rows.append({'image_id':img_id,'level':level,'bucket':'whole_image',
                         'relative_residual':summarize(original,correction)})
        if hasattr(model.head,'memory_probes'):
            for level,(original,correction) in enumerate(model.head.memory_probes):
                rows.append({'image_id':img_id,'level':level,'bucket':'memory_whole_image',
                             'relative_residual':summarize(original,correction)})
        spi=getattr(model.backbone.spi_highway,'last_probe',None)
        if spi is not None:
            rows.append({'image_id':img_id,'level':3,'bucket':'spi_whole_image',
                         'relative_residual':summarize(*spi)})
        if (i+1)%8==0:print('PROBE',i+1,'/',a.images,flush=True)
    grouped=collections.defaultdict(list)
    for r in rows:
        if r['relative_residual'] is not None:
            grouped[f'{r["level"]}_{r["bucket"]}'].append(r['relative_residual'])
    out={'assumption':'top-left YOLOX letterbox; inspect local ValTransform',
         'measurements':rows,
         'grouped':{k:{'count':len(v),'mean':sum(v)/len(v),'max':max(v)} for k,v in grouped.items()}}
    for h in handles:h.remove()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['grouped'],indent=2))


if __name__=='__main__':main()
