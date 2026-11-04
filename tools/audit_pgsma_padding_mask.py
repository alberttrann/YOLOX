"""Synthetic aspect-ratio stress audit of real YOLOX valid pixel/anchor masks.

No BDD images needed. Tests letterbox rectangles unlike 16:9 BDD, verifies
anchor-center selection and checks no positive centers land in invalid pixels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


def audit():
    from yolox.data.pgsma_train_transform import PGSMAChannelMaskTransform
    from yolox.models.pgsma_validity import valid_anchors_from_image_mask
    h,w=576,1024
    cases=[('reference_16_9',720,1280),('square',640,640),
           ('portrait',1280,720),('four_three',480,640),('wide',480,1440)]
    records=[];blockers=[]
    transform=PGSMAChannelMaskTransform(max_labels=16,flip_prob=0.,hsv_prob=0.)
    for name,src_h,src_w in cases:
        image=np.zeros((src_h,src_w,3),dtype=np.uint8)
        # label set empty: test geometry without relying on a synthetic object.
        img4, labels=transform(image,np.zeros((0,5),dtype=np.float32),(h,w))
        assert img4.shape==(4,h,w),img4.shape
        mask=torch.from_numpy(img4[None,3:4]).float()
        scale=min(h/src_h,w/src_w)
        vh=min(h,int(src_h*scale));vw=min(w,int(src_w*scale))
        expected=np.zeros((h,w),dtype=np.float32)
        expected[:vh,:vw]=1.
        mismatch=int(np.count_nonzero(img4[3]!=expected))
        features=[torch.empty(1,1,h//s,w//s) for s in (8,16,32)]
        flags=valid_anchors_from_image_mask(mask,features)[0]
        by_level=[];cursor=0
        for stride,feature in zip((8,16,32),features):
            ny,nx=h//stride,w//stride
            block=flags[cursor:cursor+ny*nx].reshape(ny,nx)
            center_ys=(torch.arange(ny)*stride+stride//2)[:,None]
            center_xs=(torch.arange(nx)*stride+stride//2)[None,:]
            exp_flags=(center_ys<vh)&(center_xs<vw)
            if not torch.equal(block,exp_flags):blockers.append(name+f' stride {stride} differs from anchor-center rule')
            by_level.append({'stride':stride,'valid_anchors':int(block.sum()),'total_anchors':int(block.numel())})
            cursor+=ny*nx
        if mismatch:blockers.append(name+f': invalid pixel mask, mismatch={mismatch}')
        if name=='reference_16_9' and vw!=w:blockers.append('reference aspect unexpectedly padded')
        if name!='reference_16_9' and vh==h and vw==w:blockers.append(name+': aspect test generated no padding')
        records.append({'case':name,'source_hw':[src_h,src_w],
                        'valid_rectangle_hw':[vh,vw],
                        'valid_pixel_fraction':round(float(mask.mean()),6),
                        'pixel_mask_mismatches':mismatch,'levels':by_level})
    return {'status':'CHECKS_PASSED' if not blockers else 'BLOCKED',
            'cases':records,'blockers':blockers,
            'scope':'Valid-content geometry only; not ignored-object or Mosaic provenance masks'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',default='audits/padding_mask_stress.json')
    a=p.parse_args()
    try:res=audit()
    except Exception as e:res={'status':'BLOCKED','error':type(e).__name__+': '+str(e)}
    out=json.dumps(res,indent=2,allow_nan=False)
    print(out)
    path=Path(a.json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(out+'\n',encoding='utf8')
    sys.exit(0 if res['status']=='CHECKS_PASSED' else 2)

if __name__=='__main__':main()
