"""Optional original-source snapshot CPU integration test.

Prerequisites: test_full_snapshot_yolox_audit.py and its source snapshot fixture
from the previous audit bundle. Not required for the current CUDA checkpoint.
"""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from test_full_snapshot_yolox_audit import Backbone,Head,Full,get_size_from_native
from safe_ttt_model import SafeSpatialTTT
from pilot_safe_ttt_predictor_only import select_predictor_only,exact_frozen_snapshot,check_frozen
from pilot_safe_ttt_phaseC import paired_accumulate_one,optimizer_step_predictor,paired_eval_one
from ttt_phaseC_constraints import ConstraintConfig
from ttt_source_views import sample_appearance,apply_appearance


def test_repaired_guard_full_snapshot_two_episode_accumulation():
    torch.set_num_threads(1)
    torch.manual_seed(87)
    m=Full(Backbone(depth=.33,width=.5,in_channels=[256,512,1024]),
           Head(9,width=.5,in_channels=[256,512,1024]))
    m.eval().requires_grad_(False)
    m.set_meta_training_state(2,80)
    m.head.training=True
    x=(torch.rand(1,3,64,64)*255)
    labels=torch.zeros(1,10,5)
    labels[0,0]=torch.tensor([2.,30.,30.,22.,22.])
    ch=get_size_from_native(m,x)
    safe=SafeSpatialTTT(ch,rank=4,step_policy='normalized',basis_gain=16.).train()
    selected,_=select_predictor_only(safe)
    frozen=exact_frozen_snapshot(safe)
    model_snapshot={k:v.detach().clone() for k,v in m.state_dict().items()}
    pred_before=next(safe.predictor.parameters()).detach().clone()
    optimizer=torch.optim.AdamW([p for _,p in selected],lr=0.0001,weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    c=ConstraintConfig(background_weight=.5,scale_weight=.25,clean_total_weight=1.)
    for idx in (300,301):
        # Same GT geometry but two different appearances.
        xview=x if idx==300 else (x+torch.rand_like(x)*2).clamp(0,255)
        shifted=apply_appearance(xview,sample_appearance(20261104,idx,strength=.85))
        report=paired_accumulate_one(m,safe,xview,shifted,labels,
              constraint_config=c,clean_weight=1.,teacher_weight=0.,accumulation_count=2)
        for v in ('clean','shifted'):
            assert report[v]['nonregression']['background_extra_gradient_multiplier_when_active']==.5
            assert max(report[v]['native_parity'].values())<2e-4
            assert max(report[v]['adapted_parity'].values())<2e-4
    gr=optimizer_step_predictor(safe,optimizer,max_grad_norm=.05)
    assert gr['predictor_gradient_l2']>0
    assert not torch.equal(pred_before,next(safe.predictor.parameters()).detach())
    check_frozen(safe,frozen)
    assert all(torch.equal(v,model_snapshot[k]) for k,v in m.state_dict().items())
    h=paired_eval_one(m,safe,x,labels)
    assert h['counts']['foreground']>0
    print('PASS Phase C full TDE snapshot 2-episode meta-grad accumulation; gradient_norm', gr['predictor_gradient_l2'])

if __name__=='__main__':test_repaired_guard_full_snapshot_two_episode_accumulation()
