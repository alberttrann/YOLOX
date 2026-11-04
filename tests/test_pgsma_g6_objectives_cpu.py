"""Pilot G6 CPU-only: factorial direction/background-guard contract tests."""
from pathlib import Path
import importlib.util
import torch

ROOT=Path(__file__).resolve().parents[1]

def make_cls():
    spec=importlib.util.spec_from_file_location('g6_pgsma',ROOT/'yolox/models/pgsma.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PGSMA


def args():
    base=torch.tensor([[[0.0,0.0,0.0],[0.0,0.0,0.0]]])
    fg=torch.tensor([0,1])
    labels=torch.tensor([1,1])
    quality=torch.tensor([0.55,0.9])
    uid=torch.tensor([7,8])
    return base,fg,labels,quality,uid


def test_bounded_zero_after_target_and_gradient_stops():
    cls=make_cls()
    base,fg,labels,quality,uid=args()
    corr=(base+torch.tensor([[[-0.10,0.12,-0.10],[-0.1,0.12,-0.1]]])).requires_grad_()
    loss=cls._relative_bounded_margin_cost(base,corr,fg,labels,quality,uid)
    assert abs(loss.item()) < 1e-10
    loss.backward()
    assert corr.grad is not None
    assert corr.grad.abs().sum().item() < 1e-9


def test_bounded_initial_gradient_same_direction_as_g4_softplus():
    cls=make_cls()
    base,fg,labels,quality,uid=args()
    for fn in ('_relative_bounded_margin_cost','_relative_margin_cost'):
        corr=base.clone().requires_grad_()
        loss=getattr(cls,fn)(base,corr,fg,labels,quality,uid)
        loss.backward()
        assert torch.isfinite(loss)
        assert (corr.grad[...,1] < 0).all()
        assert (corr.grad[...,[0,2]] > 0).all()


def test_background_guard_identity_improvement_regression_and_grad():
    cls=make_cls()
    base=torch.tensor([[[0.0,-2.,3.],[1.,0.,-1.]]])
    bg=torch.tensor([0,1])
    unchanged=base.clone().requires_grad_()
    val=cls._relative_background_regret(base,unchanged,bg)
    assert val.item()==0.
    lower=(base-0.4).requires_grad_()
    assert cls._relative_background_regret(base,lower,bg).item()==0.
    higher=(base+0.3).requires_grad_()
    penalty=cls._relative_background_regret(base,higher,bg)
    assert penalty.item()>0.
    penalty.backward()
    assert higher.grad is not None and (higher.grad>0).all()


def test_empty_foreground_background_safe():
    cls=make_cls()
    base=torch.randn(1,2,3)
    corr=base.clone().requires_grad_()
    z=cls._relative_bounded_margin_cost(base,corr,torch.empty(0,dtype=torch.long),
        torch.empty(0,dtype=torch.long),torch.empty(0),torch.empty(0,dtype=torch.long))
    b=cls._relative_background_regret(base,corr,torch.empty(0,dtype=torch.long))
    (z+b).backward()
    assert torch.isfinite(z+b)
    assert corr.grad is not None


def test_objective_choices_trainable_no_native_grad():
    cls=make_cls()
    for objective in ('bounded_probe','margin_bg_probe','bounded_bg_probe'):
        torch.manual_seed(0)
        p=cls(classes=3,channels=8,dim=8,modes=2)
        p.candidate_objective=objective
        feats=[torch.randn(1,8,4,4,requires_grad=True),
               torch.randn(1,8,2,2,requires_grad=True),
               torch.randn(1,8,1,1,requires_grad=True)]
        weights=[torch.randn(3,8,1,1,requires_grad=True) for _ in feats]
        biases=[torch.randn(3,requires_grad=True) for _ in feats]
        logits=[torch.nn.functional.conv2d(f,w,b) for f,w,b in zip(feats,weights,biases)]
        obj=[torch.randn(1,1,*f.shape[-2:]) for f in feats]
        match={'fg_ids':torch.tensor([0,1,17,18]),'fg_labels':torch.tensor([0,0,1,2]),
               'fg_quality':torch.tensor([0.5,0.6,0.7,0.9]),'gt_uid':torch.tensor([0,0,1,2]),
               'fg_sizes':torch.tensor([5.,5.,11.,45.]),'bg_ids':torch.tensor([2,3,4])}
        losses=p.semantic_losses(feats,list(zip(weights,biases)),logits,obj,match)
        total=sum(losses[k] for k in ('prototype','candidate','gate','teacher'))
        total.backward()
        assert torch.isfinite(total)
        assert losses['rank_probe_loss'].item()>=0
        assert losses['clean_bg_guard'].item()>=0
        assert any(w.grad is not None and w.grad.abs().sum()>0 for w in p.predictors.parameters())
        assert all(x.grad is None for x in feats+weights+biases)
