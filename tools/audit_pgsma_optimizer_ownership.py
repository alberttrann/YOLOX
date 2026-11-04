"""No-data, no-training ownership audit for PG-SMA semantic pilot variants.

Run from the actual YOLOX checkout:
  python -m tools.audit_pgsma_optimizer_ownership --json audits/semantic_ownership.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

from ._pgsma_pilot_core import configure_variant, model_parameter_split


def run():
    from exps.default.tde_yolox_s_pgsma import Exp
    variants = ('memory_simota', 'pgsma_simota')
    reports=[]
    for variant in variants:
        exp=Exp()
        if bool(exp.allow_full_training):
            raise RuntimeError('Expected allow_full_training=False')
        exp.scpa=False
        exp.pgsma_variant = 'memory' if variant.startswith('memory') else 'pgsma'
        model=exp.get_model()
        mode=configure_variant(model,variant)
        native_optimizer=exp.get_optimizer(batch_size=2)
        # Same selection as the actual patched pilot runner.
        semantic_params=[p for p in model.semantic_parameters() if p.requires_grad]
        semantic_optimizer=torch.optim.AdamW(semantic_params,lr=1e-4,weight_decay=0.0)
        model_parameter_split(model,native_optimizer,semantic_optimizer)
        selected={id(p) for group in semantic_optimizer.param_groups for p in group['params']}
        frozen={id(p) for p in model.semantic_parameters() if not p.requires_grad}
        teacher={id(p) for p in model.semantic.teacher_projectors.parameters()}
        if selected & (frozen | teacher):
            raise RuntimeError('Frozen teacher was included in the trainable optimizer')
        reports.append({
            'variant':variant,'inference_mode':mode,
            'semantic_trainable_parameter_tensors':len(semantic_params),
            'semantic_frozen_parameter_tensors':len(frozen),
            'teacher_parameter_tensors':len(teacher),
            'semantic_trainable_elements':sum(p.numel() for p in semantic_params),
            'frozen_teacher_in_optimizer':False,
            'native_semantic_overlap':False,
            'optimizer_membership':'exact_trainable_semantic_set',
        })
    return {'status':'CHECKS_PASSED','training_performed':False,
            'native_or_semantic_updates':0,'variants':reports,
            'scope':'optimizer membership only, no CUDA/AMP or data-loader execution'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args(argv)
    try:
        report=run()
        result=0
    except Exception as exc:
        report={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
        result=2
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))
    return result

if __name__=='__main__':
    sys.exit(main())
