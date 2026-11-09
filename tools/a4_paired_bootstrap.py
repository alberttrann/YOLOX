"""P0 paired COCO bootstrap, using resampled image instances with unique IDs.

COCO AP is NOT the mean of per-image APs. We re-evaluate a resampled COCO
catalog for every replicate. Slow but definitionally faithful. Requires
pycocotools on the user's evaluation environment.
"""
import argparse
import collections
import contextlib
import io
import json
import random
from pathlib import Path


def score(gt_data, original_detections, sampled_image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    by_id={int(im['id']):im for im in gt_data['images']}
    anns=collections.defaultdict(list)
    for a in gt_data['annotations']:anns[int(a['image_id'])].append(a)
    preds=collections.defaultdict(list)
    for d in original_detections:preds[int(d['image_id'])].append(d)
    subset={'info':gt_data.get('info',{}),'licenses':gt_data.get('licenses',[]),
            'categories':gt_data['categories'],'images':[],'annotations':[]}
    dets=[];ann_id=1
    for index, old_id in enumerate(sampled_image_ids,1):
        im=dict(by_id[old_id]);im['id']=index;subset['images'].append(im)
        for a in anns[old_id]:
            entry=dict(a);entry['id']=ann_id;entry['image_id']=index
            subset['annotations'].append(entry);ann_id+=1
        for d in preds[old_id]:
            entry=dict(d);entry['image_id']=index
            dets.append(entry)
    if not dets:return {'AP':0.,'AR100':0.}
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=subset;coco.createIndex()
        result=coco.loadRes(dets)
        ev=COCOeval(coco,result,'bbox')
        ev.evaluate();ev.accumulate();ev.summarize()
    return {'AP':float(ev.stats[0]),'AR100':float(ev.stats[8]),
            'AP_small':float(ev.stats[3]),'AR100_small':float(ev.stats[9]),
            'AP_medium':float(ev.stats[4]),'AP_large':float(ev.stats[5])}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--left',required=True,help='left branch directory from a4_eval_counterfactual')
    p.add_argument('--right',required=True)
    p.add_argument('--allow-different-checkpoints',action='store_true',
                   help='Use ONLY for paired A0/A1/A2/A3 across different checkpoints')
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--seed',type=int,default=137)
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    if a.iterations<10:raise ValueError('Use at least 10 paired replicates')
    l,r=Path(a.left),Path(a.right)
    lm=json.loads((l/'metadata.json').read_text(encoding='utf-8'))
    rm=json.loads((r/'metadata.json').read_text(encoding='utf-8'))
    for key in ('data_ann','weights','conf','nms','input_size'):
        if lm[key]!=rm[key]:raise RuntimeError('Unpaired comparison: '+key)
    # Legacy FP16 (whole model.half) and CUDA AMP are NOT interchangeable.
    for key in ('fp16','precision_mode'):
        if lm.get(key) != rm.get(key):
            raise RuntimeError('Unpaired precision: '+key+'; rerun both with one mode')
    if not a.allow_different_checkpoints and lm['checkpoint_sha256']!=rm['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs: use --allow-different-checkpoints for model-to-model comparison')
    gt=json.loads(Path(lm['data_ann']).read_text(encoding='utf-8'))
    ls=json.loads((l/'detections.json').read_text(encoding='utf-8'))
    rs=json.loads((r/'detections.json').read_text(encoding='utf-8'))
    image_ids=sorted(int(im['id']) for im in gt['images']);rng=random.Random(a.seed)
    metrics=['AP','AR100','AP_small','AR100_small','AP_medium','AP_large']
    diffs={key:[] for key in metrics}
    for i in range(a.iterations):
        sample=rng.choices(image_ids,k=len(image_ids))
        x=score(gt,ls,sample);y=score(gt,rs,sample)
        for key in metrics:diffs[key].append(x[key]-y[key])
        if (i+1)%25==0:print('PROGRESS',i+1,'/',a.iterations,flush=True)
    summary={}
    for key,values in diffs.items():
        values=sorted(values);n=len(values)
        summary[key]={'delta_mean':sum(values)/n,'ci95':[values[int(.025*(n-1))],values[int(.975*(n-1))]],
                      'prob_positive':sum(v>0 for v in values)/n}
    output={'comparison':f'{lm["branch"]}-{rm["branch"]}',
            'iterations':a.iterations,'seed':a.seed,'num_images':len(image_ids),
            'metrics':summary,'gt_categories':gt['categories']}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output['metrics'],indent=2))


if __name__=='__main__':main()
