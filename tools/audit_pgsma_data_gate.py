"""Multi-batch real loader/validity/SCPA readiness audit; no optimizer updates.

Run from the YOLOX repository after installing this data patch:
    python -m tools.audit_pgsma_data_gate --batches 6 --batch-size 2 --json audits/data_gate.json

Mosaic's synthetic output canvas is considered valid training content;
this check does not claim pixel-provenance reconstruction or no GT filtering.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch


def audit(batches=6,batch_size=2):
    from exps.default.tde_yolox_s_pgsma import Exp
    from yolox.models.pgsma_validity import valid_anchors_from_image_mask
    exp=Exp()
    exp.data_num_workers=0
    report={'status':'CHECKS_PASSED','batches':int(batches),
            'batch_size':int(batch_size),'runs':{},'blockers':[],
            'limits':['Mosaic synthetic border is treated as valid train background',
                      'YOLOX TrainTransform may remove subpixel boxes; this is not an annotation-survival audit',
                      'No validation of ignore regions not present in COCO JSON',
                      'No real CUDA training or TensorRT measurement']}
    for name,no_aug,force in [('no_aug',True,False),('forced_mosaic',False,True),('mixed_mosaic',False,False)]:
        loader=exp.get_data_loader(batch_size,is_distributed=False,no_aug=no_aug,cache_img=False)
        ds=loader.dataset
        if type(ds).__name__!='MosaicDetection':
            report['blockers'].append(name+': MosaicDetection wrapper missing')
        if force:
            ds.mosaic_prob=1.0
        fractions=[];gt_counts=[];anchor_counts=[];bad_centers=0
        try:
            iterator=iter(loader)
            for _ in range(batches):
                images,labels,*_=next(iterator)
                if images.ndim!=4 or images.shape[1]!=4:
                    report['blockers'].append(name+': expected 4-channel RGB+mask images')
                    break
                mask=images[:,3:4].float()
                if not bool(torch.isfinite(mask).all()):
                    report['blockers'].append(name+': nonfinite mask values')
                if not bool(((mask==0.)|(mask==1.)).all()):
                    report['blockers'].append(name+': mask must be binary before preprocessing')
                if not bool(torch.isfinite(labels).all()):
                    report['blockers'].append(name+': nonfinite labels')
                b,_,h,w=images.shape
                feats=[torch.empty(b,1,h//s,w//s) for s in (8,16,32)]
                flags=valid_anchors_from_image_mask(mask,feats)
                anchor_counts.extend(flags.sum(-1).tolist())
                fractions.extend(mask.mean(dim=(1,2,3)).tolist())
                valid=(labels[...,3]>0)&(labels[...,4]>0)
                gt_counts.extend(valid.sum(-1).tolist())
                if bool((valid.sum(-1)==labels.shape[1]).any()):
                    report['blockers'].append(name+': annotation capacity reached; investigate truncation')
                # GT centers ought to be in the valid training canvas.
                for sample in range(b):
                    centers=labels[sample,valid[sample],1:3].float()
                    if centers.numel()==0:continue
                    x=centers[:,0].floor().long().clamp(0,w-1)
                    y=centers[:,1].floor().long().clamp(0,h-1)
                    bad_centers+=int((mask[sample,0,y,x]<.5).sum())
        except Exception as err:
            report['blockers'].append(name+': '+type(err).__name__+': '+str(err))
        if bad_centers:
            report['blockers'].append(name+': '+str(bad_centers)+' valid GT centers lie in invalid content')
        if force and fractions and min(fractions)<1.:
            report['blockers'].append('forced_mosaic: expected full synthetic composite canvas valid')
        report['runs'][name]={'dataset_type':type(ds).__name__,
            'examples':len(fractions),'valid_pixel_fraction_range':
              [round(min(fractions),5),round(max(fractions),5)] if fractions else None,
            'gt_count_range':[min(gt_counts),max(gt_counts)] if gt_counts else None,
            'valid_anchor_count_range':[min(anchor_counts),max(anchor_counts)] if anchor_counts else None,
            'invalid_gt_centers':bad_centers}
    if report['blockers']:
        report['status']='BLOCKED'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--json',default=None)
    a=p.parse_args()
    try:
        if not (1<=a.batches<=500 and 1<=a.batch_size<=256):
            raise ValueError('Bad batch or sample count')
        result=audit(a.batches,a.batch_size)
    except Exception as err:
        result={'status':'BLOCKED','error':type(err).__name__+': '+str(err)}
    text=json.dumps(result,indent=2,allow_nan=False)
    print(text)
    if a.json:
        path=Path(a.json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text+'\n',encoding='utf8')
    sys.exit(0 if result['status']=='CHECKS_PASSED' else 2)

if __name__=='__main__':main()
