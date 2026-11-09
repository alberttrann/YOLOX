"""Fail-closed audit of local experiment prerequisites and data sufficiency."""
import argparse
import collections
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from yolox.exp import get_exp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s_fourier_a4.py')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    e=get_exp(a.model_exp,None)
    meta=e._preflight()
    model=e.get_model()
    with open(e.train_ann,encoding='utf-8') as f:train=json.load(f)
    with open(e.val_ann,encoding='utf-8') as f:valid=json.load(f)
    train_cats=[(int(c['id']),c['name']) for c in train['categories']]
    val_cats=[(int(c['id']),c['name']) for c in valid['categories']]
    if train_cats!=val_cats:raise RuntimeError('Train/val category mapping differs')
    by_id=collections.Counter(x['image_id'] for x in train.get('annotations',[]))
    sizes=collections.Counter()
    for x in train.get('annotations',[]):
        _,_,w,h=x['bbox'];area=float(x.get('area',w*h))
        sizes['small' if area<1024 else 'medium' if area<9216 else 'large']+=1
    expected=max(by_id.values(),default=0)*(4 if e.mosaic_prob>0 else 1)
    if e._capacity<expected:raise RuntimeError('TrainTransform target capacity underestimates 4x mosaic')
    names=set(dict(model.named_modules()))
    if 'backbone.spi_highway' not in names:raise RuntimeError('A4 SPI not constructed')
    if not model.head.enable_engram:raise RuntimeError('Engram not available in A4')
    if model.head.fourier_variant not in ('a4','a4_control'):
        raise RuntimeError('Wrong A4 head')
    if any('ttt_lrs' in k or 'mask_token' in k for k in model.state_dict()):
        raise RuntimeError('Legacy TTT state survived in A4')
    root=Path.cwd()
    result={'experiment':e.exp_name,'variant':e.fourier_variant,
            'dataset_preflight':meta,'classes':train_cats,
            'source_annotations':str(Path(e.train_ann).resolve()),
            'ground_truth_count':len(train.get('annotations',[])),
            'annotation_sizes_original_pixels':dict(sizes),
            'required_target_capacity':expected,'implemented_target_capacity':e._capacity,
            'heads':['P3','P4','P5'],
            'actual_parameter_count':sum(p.numel() for p in model.parameters()),
            'known_gaps':['heterogeneous-aspect batch collation not implemented',
                          '4K low latency unverified','oriented annotations not supported',
                          'evaluation maxDets=100 may hide dense UAV detections']}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
