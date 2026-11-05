"""Portable CPU tests for predictor-only exact-meta pilot. Run from YOLOX root."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'tools'
# For extracted YOLOX tree, use copied tools/; otherwise allow local source package.
if not (TOOLS/'safe_ttt_model.py').exists():
    TOOLS=Path(__file__).resolve().parents[2]/'vnext2_ttt_training_attribution'/'tools'
sys.path.insert(0,str(TOOLS))
sys.path.insert(0,str(ROOT/'tools'))
from safe_ttt_model import SafeSpatialTTT
from pilot_safe_ttt_predictor_only import select_predictor_only,exact_frozen_snapshot,check_frozen,predictor_delta,summarize_holdout


def build():
    torch.manual_seed(20261104)
    m=SafeSpatialTTT(16,rank=4,step_policy='normalized')
    return m


def test_freeze_and_nonpredictor_immutability():
    m=build()
    chosen, frozen=select_predictor_only(m)
    before=exact_frozen_snapshot(m)
    predictor_before={n:p.detach().clone() for n,p in chosen}
    opt=torch.optim.AdamW([p for _,p in chosen],lr=0.0005,weight_decay=0)
    x=torch.randn(1,16,8,8)
    target=x.detach()+torch.randn_like(x)*0.2
    adapted,_=m.run_episode(x,meta_train=True)
    obj=(adapted-target).square().mean()
    opt.zero_grad(set_to_none=True)
    obj.backward()
    pred_grads=[p.grad for _,p in chosen]
    assert all(g is not None and torch.isfinite(g).all() for g in pred_grads)
    assert any(float(g.detach().abs().sum())>0 for g in pred_grads)
    assert all(p.grad is None for _,p in frozen)
    opt.step()
    check_frozen(m,before)
    assert predictor_delta(m,predictor_before)>0


def test_unknown_key_rejected():
    m=build()
    m.rogue=torch.nn.Parameter(torch.ones(1))
    try: select_predictor_only(m)
    except RuntimeError as err:assert 'Unknown parameter' in str(err)
    else:raise AssertionError('Unknown parameter accepted')


def test_holdout_assesses_paired_changes():
    initial=[{'index':10,'native':1.0,'delta':-0.0001,'cosine':-0.1},
             {'index':11,'native':1.0,'delta':0.0002,'cosine':0.2}]
    trained=[{'index':10,'native':1.0,'delta':-0.0002,'cosine':0.1},
             {'index':11,'native':1.0,'delta':0.0003,'cosine':0.1}]
    result=summarize_holdout(initial,trained)
    assert result['train_state_better_count']==1
    assert result['train_state_worse_count']==1
    assert result['sample_size']==2
    assert result['trained_beneficial_count']==1


def test_holdout_rejects_reordered_indices():
    base=[{'index':1,'native':1.,'delta':0.,'cosine':0.},
          {'index':2,'native':1.,'delta':0.,'cosine':0.}]
    try:summarize_holdout(base,list(reversed(base)))
    except RuntimeError as err:assert 'order' in str(err)
    else:raise AssertionError('Holdout reordering accepted')


if __name__=='__main__':
    torch.set_num_threads(1)
    for name,func in list(globals().items()):
        if name.startswith('test_') and callable(func):
            func()
            print('PASS',name)
