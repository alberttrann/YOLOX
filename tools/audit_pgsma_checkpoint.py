"""Audit trusted pretrained checkpoint mapping into vNext.3. No training or writes."""
from __future__ import annotations
import argparse
import json
import sys
from ._pgsma_preflight_common import (load_exp,load_torch_checkpoint,normalize_state_keys,
    unwrap_state_dict,checkpoint_coverage,dump_json)


def source_backbone_coverage(src, expected, prefixes):
    """Measure reuse of ORIGINAL backbone tensors, not new architecture size."""
    rows={}
    for prefix in prefixes:
        dot=prefix+'.'
        original={k:v for k,v in src.items() if k.startswith(dot)}
        good={k:v for k,v in original.items()
              if k in expected and tuple(v.shape)==tuple(expected[k].shape)}
        src_num=sum(v.numel() for v in original.values())
        matched_num=sum(v.numel() for v in good.values())
        new_target=[k for k in expected if k.startswith(dot) and k not in original]
        rows[prefix]={'source_tensors':len(original),
                      'matched_source_tensors':len(good),
                      'source_elements':src_num,
                      'matched_source_elements':matched_num,
                      'pretrained_element_ratio':round(matched_num/src_num,6) if src_num else None,
                      'incompatible_source_examples':[k for k in original if k not in good][:12],
                      'new_target_tensors':len(new_target),
                      'new_target_examples':new_target[:8]}
    return rows

def audit(path,min_backbone=0.85,prefer_ema=False):
    exp=load_exp(); model=exp.get_model().cpu()
    ckpt=load_torch_checkpoint(path)
    src,which=unwrap_state_dict(ckpt,prefer_ema=prefer_ema)
    src=normalize_state_keys(src)
    coverage=checkpoint_coverage(model,src)
    groups=coverage['groups']
    blocker=[]; warning=[]
    backbone_prefixes=('backbone.backbone.stem','backbone.backbone.dark2',
                       'backbone.backbone.dark3','backbone.backbone.dark4',
                       'backbone.backbone.dark5')
    source_coverage=source_backbone_coverage(src,model.state_dict(),backbone_prefixes)
    for prefix in backbone_prefixes:
        r=source_coverage[prefix]['pretrained_element_ratio']
        if r is None or r < min_backbone:
            blocker.append(f'{prefix}: only {r} of original pretrained elements loaded (minimum {min_backbone})')
    # Destination-model ratios remain visible as informational diagnostics.
    # New context/attention tensors absent from official weights are expected.
    if coverage['shape_mismatches']:
        warning.append(f'{len(coverage["shape_mismatches"])} same-key incompatible tensors (may be COCO class head mismatch)')
    if not src:
        blocker.append('No usable checkpoint tensors')
    matched=coverage['matched']
    model.load_state_dict(matched,strict=False)
    result={
        'source':path,'source_subkey':which,'source_tensors':len(src),
        'model_tensors':len(model.state_dict()),'matched_tensors':len(matched),
        'matched_elements':coverage['matched_parameter_elements'],
        'coverage_by_prefix':groups,
        'pretrained_source_coverage_by_prefix':source_coverage,
        'shape_mismatch_examples':coverage['shape_mismatches'][:35],
        'unexpected_examples':coverage['unexpected_keys'][:35],
        'missing_examples':coverage['missing_keys'][:35],
        'blockers':blocker,'warnings':warning,'status':'BLOCKED' if blocker else 'CHECKS_PASSED',
        'note':'Source pretrained backbone coverage determines go/no-go. New modules remain untrained; COCO class-head shape mismatch is expected.'
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--min-backbone-ratio',type=float,default=.85)
    p.add_argument('--prefer-ema',action='store_true')
    p.add_argument('--json')
    a=p.parse_args()
    try:result=audit(a.checkpoint,a.min_backbone_ratio,a.prefer_ema)
    except Exception as e:result={'status':'BLOCKED','error':f'{type(e).__name__}: {e}'}
    dump_json(result,a.json)
    sys.exit(2 if result['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
