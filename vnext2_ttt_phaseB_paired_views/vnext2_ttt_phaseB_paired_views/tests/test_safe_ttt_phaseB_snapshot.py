"""Optional CPU integration test on supplied (older) YOLOX TDE source snapshot.

Requires test_full_snapshot_yolox_audit.py and a source snapshot selected by
VNEXT2_DUMP_SOURCE; this is not a substitute for checkpoint/CUDA validation.
"""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from test_full_snapshot_yolox_audit import Backbone,Head,Full,get_size_from_native
from safe_ttt_model import SafeSpatialTTT
from pilot_safe_ttt_predictor_only import select_predictor_only,exact_frozen_snapshot,check_frozen
from pilot_safe_ttt_paired_views import (paired_meta_step,paired_eval_one,reference_consistency)
from ttt_source_views import sample_appearance,apply_appearance


def model_fixture():
    torch.manual_seed(20261104)
    m=Full(Backbone(depth=.33,width=.5,in_channels=[256,512,1024]),
           Head(9,width=.5,in_channels=[256,512,1024]))
    m.eval().requires_grad_(False)
    m.set_meta_training_state(2,80)
    m.head.training=True
    x=(torch.rand(1,3,64,64)*255)
    labels=torch.zeros(1,10,5)
    labels[0,0]=torch.tensor([2.,30.,30.,22.,22.])
    C=get_size_from_native(m,x)
    safe=SafeSpatialTTT(C,rank=4,step_policy='normalized',basis_gain=16.).train()
    _,frozen=select_predictor_only(safe)
    snapshot=exact_frozen_snapshot(safe)
    return m,safe,x,labels,snapshot


def test_full_snapshot_paired_meta_and_frozen_basis():
    torch.set_num_threads(1)
    for clean_weight,teacher_weight in ((0.0,0.0),(0.5,0.0),(0.5,0.03)):
        m,safe,x,labels,snap=model_fixture()
        xshift=apply_appearance(x,sample_appearance(20261104,300,strength=.85))
        native_snapshot={k:v.detach().clone() for k,v in m.state_dict().items()}
        pred_initial=next(safe.predictor.parameters()).detach().clone()
        optim=torch.optim.AdamW([p for p in safe.predictor.parameters() if p.requires_grad],lr=1e-4)
        report=paired_meta_step(m,safe,x,xshift,labels,optim,
                                clean_weight=clean_weight,teacher_weight=teacher_weight)
        check_frozen(safe,snap)
        assert report['predictor_gradient_l2']>0
        assert (report['clean'] is not None)==(clean_weight>0)
        assert max(report['native_parity'].values()) < 2e-4
        assert max(report['adapted_parity'].values()) < 2e-4
        assert not torch.equal(pred_initial,next(safe.predictor.parameters()).detach())
        assert all(torch.equal(v,native_snapshot[k]) for k,v in m.state_dict().items())
        if teacher_weight:
            assert report['source_teacher_positive_anchors']>0
            assert report['source_teacher_consistency']>=0
        else:
            assert report['source_teacher_positive_anchors']==0
        before=paired_eval_one(m,safe,xshift,labels)
        assert before['counts']['foreground']>0
        assert before['fast_state_l2']>0
        assert all(torch.equal(v,native_snapshot[k]) for k,v in m.state_dict().items())
        print('PASS paired full YOLOX snapshot','clean_weight',clean_weight, 'teacher_weight',teacher_weight,
              'meta gradient',report['predictor_gradient_l2'],
              'task delta',report['adapted_minus_native'],
              'test delta',before['delta'])


if __name__=='__main__':
    test_full_snapshot_paired_meta_and_frozen_basis()
