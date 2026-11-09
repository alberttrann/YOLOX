"""Strict provenance guard for A4/A4-R's NONPERSISTENT frozen filter banks.

A4 and A4-R have identical trainable shapes. Loading the wrong filter-basis
experiment may pass state_dict(strict=True) while silently changing inference.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(model_exp,checkpoint):
    from yolox.exp import get_exp
    exp=get_exp(model_exp,None)
    if not hasattr(exp,'a4_filter_basis'):
        raise RuntimeError('This verifier is for A4 or A4-R only')
    ckpt=Path(checkpoint).resolve()
    if not ckpt.is_file():raise FileNotFoundError(ckpt)
    manifest=ckpt.parent/'a4_provenance.json'
    if not manifest.is_file():
        raise RuntimeError('A4 checkpoint missing required provenance: '+str(manifest))
    rec=json.loads(manifest.read_text(encoding='utf-8'))
    if rec.get('filter_basis')!=exp.a4_filter_basis:
        raise RuntimeError('WRONG A4 FILTER BASIS: stored='+repr(rec.get('filter_basis'))+
                           ', requested='+repr(exp.a4_filter_basis))
    here=exp._preflight()
    if rec.get('dataset_preflight')!=here:
        raise RuntimeError('A4 dataset/geometry preflight differs from stored provenance')
    source_hashes=rec.get('source_sha256')
    if not source_hashes:raise RuntimeError('Missing source SHA manifest')
    root=Path(__file__).resolve().parents[1]
    for relative,digest in source_hashes.items():
        path=root/relative
        if not path.is_file():raise RuntimeError('Missing recorded source: '+relative)
        current=hashlib.sha256(path.read_bytes()).hexdigest()
        if current!=digest:
            raise RuntimeError('Source changed since A4 training: '+relative+
                               '. Do not interpret checkpoint under drifted code')
    print('A4_CHECKPOINT_PROVENANCE_PASS',{'basis':exp.a4_filter_basis,'checkpoint':str(ckpt)})
    return rec


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',required=True)
    p.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    verify(a.model_exp,a.checkpoint)


if __name__=='__main__':main()
