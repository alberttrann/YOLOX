"""P1: per-class and per-size raw prediction shifts for paired counterfactuals.

Greedy nearest same-class prediction analysis. It does NOT replace COCO AP,
whose greedy ranking and ignore/crowd semantics are more complex.
"""
import argparse
import collections
import json
from pathlib import Path


def iou(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    l=max(ax,bx);t=max(ay,by);r=min(ax+aw,bx+bw);bot=min(ay+ah,by+bh)
    inter=max(0,r-l)*max(0,bot-t)
    return inter/max(1e-9,aw*ah+bw*bh-inter)


def details(gt,ds):
    by_image=collections.defaultdict(list)
    for p in ds:by_image[(int(p['image_id']),int(p['category_id']))].append(p)
    samples=[]
    for a in gt['annotations']:
        if a.get('iscrowd',0):continue
        key=(int(a['image_id']),int(a['category_id']))
        candidate=by_image[key]
        best=max(candidate,key=lambda p:iou(a['bbox'],p['bbox']),default=None)
        score=best.get('score',0) if best else 0
        match_iou=iou(a['bbox'],best['bbox']) if best else 0
        area=float(a.get('area',a['bbox'][2]*a['bbox'][3]))
        category=('small' if area<1024 else 'medium' if area<9216 else 'large')
        samples.append({'image_id':key[0],'class_id':key[1], 'gt_id':int(a['id']),
                        'size':category,'best_iou':match_iou,'best_score':float(score),
                        'matched_50':match_iou>=.5})
    return {t['gt_id']:t for t in samples}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--left',required=True)
    p.add_argument('--right',required=True)
    p.add_argument('--allow-different-checkpoints',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    l,r=Path(a.left),Path(a.right)
    lm=json.loads((l/'metadata.json').read_text());rm=json.loads((r/'metadata.json').read_text())
    for key in ('data_ann','conf','nms','weights'):
        if lm[key]!=rm[key]:raise RuntimeError('Unpaired input: '+key)
    # Legacy FP16 (whole model.half) and CUDA AMP are NOT interchangeable.
    for key in ('fp16','precision_mode'):
        if lm.get(key) != rm.get(key):
            raise RuntimeError('Unpaired precision: '+key+'; rerun both with one mode')
    if not a.allow_different_checkpoints and lm['checkpoint_sha256']!=rm['checkpoint_sha256']:
        raise RuntimeError('Cross-checkpoint comparison requires --allow-different-checkpoints')
    gt=json.loads(Path(lm['data_ann']).read_text())
    dl=details(gt,json.loads((l/'detections.json').read_text()))
    dr=details(gt,json.loads((r/'detections.json').read_text()))
    rows=[];by_group=collections.defaultdict(list)
    for aid,x in dl.items():
        y=dr[aid]
        row={'gt_id':aid,'image_id':x['image_id'],'class_id':x['class_id'],
             'size':x['size'],'delta_score':x['best_score']-y['best_score'],
             'delta_best_iou':x['best_iou']-y['best_iou'],
             'delta_found50':int(x['matched_50'])-int(y['matched_50'])}
        rows.append(row);by_group[f'{row["size"]}_class{row["class_id"]}'].append(row)
    summary={key:{'objects':len(v),**{k:sum(float(t[k]) for t in v)/len(v)
                    for k in ('delta_score','delta_best_iou','delta_found50')}}
             for key,v in by_group.items()}
    out={'meaning':'greedy nearest same-category GT-to-detection analysis; not COCO AP',
         'summary':summary,'rows':rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
