"""CPU-only correctness/gradient tests for opt-in candidate margin probe."""
from pathlib import Path
import importlib.util
import torch

ROOT=Path(__file__).resolve().parents[1]

def make():
    path=ROOT/'yolox/models/pgsma.py'
    spec=importlib.util.spec_from_file_location('g4_pgsma',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PGSMA


def test_default_unmodified_and_zero_identity():
    cls=make(); p=cls(classes=3,channels=8,dim=8,modes=2)
    assert p.candidate_objective=='v2'
    f=[torch.randn(1,8,5,6),torch.randn(1,8,3,3),torch.randn(1,8,2,2)]
    logits=[torch.randn(1,3,*x.shape[-2:]) for x in f]
    objs=[torch.randn(1,1,*x.shape[-2:]) for x in f]
    actual=p.forward_levels(f,logits,objs)
    assert all(torch.equal(x,y) for x,y in zip(actual,logits))


def test_relative_margin_cost_gradient_targets():
    cls=make()
    base=torch.tensor([[[0.0,0.0,0.0],[0.0,0.0,0.0]]])
    # Flattened 2 anchors, both class 1
    pred=base.clone().requires_grad_()
    loss=cls._relative_margin_cost(base,pred,torch.tensor([0,1]),
        torch.tensor([1,1]),torch.tensor([0.5,0.8]),torch.tensor([7,8]))
    loss.backward()
    assert torch.isfinite(loss)
    # Gradient descent: raise true-class score, lower competing-class scores.
    assert torch.all(pred.grad[0,:,1] < 0)
    assert torch.all(pred.grad[0,:,[0,2]] > 0)


def test_positive_cost_decreases_for_useful_discriminative_change():
    cls=make()
    base=torch.tensor([[[0.5, 0.1, 0.0]]])
    fg=torch.tensor([0]); labels=torch.tensor([0]);quality=torch.tensor([0.8]);uid=torch.tensor([0])
    b=cls._relative_margin_cost(base,base,fg,labels,quality,uid)
    better=base+torch.tensor([[[0.1,-0.05,-0.05]]])
    worse=base+torch.tensor([[[-0.1,0.05,0.05]]])
    assert cls._relative_margin_cost(base,better,fg,labels,quality,uid) < b
    assert cls._relative_margin_cost(base,worse,fg,labels,quality,uid) > b


def test_single_class_and_empty_safe():
    cls=make();base=torch.randn(1,2,1);cor=(base+0.1).requires_grad_()
    pos=cls._relative_margin_cost(base,cor,torch.tensor([0]),torch.tensor([0]),
                                  torch.tensor([0.7]),torch.tensor([1]))
    assert torch.isfinite(pos)
    pos.backward()
    assert cor.grad is not None and torch.isfinite(cor.grad).all()
    empty=cls._relative_margin_cost(base,cor,torch.empty(0,dtype=torch.long),
                torch.empty(0,dtype=torch.long),torch.empty(0),torch.empty(0,dtype=torch.long))
    assert torch.isfinite(empty)


def test_optin_grad_no_native_leak():
    cls=make();torch.manual_seed(42)
    p=cls(classes=3,channels=8,dim=8,modes=2)
    p.candidate_objective='margin_probe'
    feat=[torch.randn(1,8,4,4,requires_grad=True),torch.randn(1,8,2,2,requires_grad=True),torch.randn(1,8,1,1,requires_grad=True)]
    weight=[torch.randn(3,8,1,1,requires_grad=True) for _ in feat]
    bias=[torch.randn(3,requires_grad=True) for _ in feat]
    logits=[torch.nn.functional.conv2d(f,w,b) for f,w,b in zip(feat,weight,bias)]
    obj=[torch.randn(1,1,*f.shape[-2:]) for f in feat]
    match={'fg_ids':torch.tensor([0,1,17,18]),'fg_labels':torch.tensor([0,0,1,2]),
           'fg_quality':torch.tensor([0.5,0.6,0.7,0.9]),'gt_uid':torch.tensor([0,0,1,2]),
           'fg_sizes':torch.tensor([5.,5.,11.,45.]),'bg_ids':torch.tensor([2,3,4])}
    losses=p.semantic_losses(feat,list(zip(weight,bias)),logits,obj,match)
    total=losses['prototype']+losses['candidate']+losses['gate']+losses['teacher']
    total.backward()
    assert torch.isfinite(total)
    assert losses['rank_probe_loss'].item()>0
    assert any(t.grad is not None and t.grad.abs().sum()>0 for t in p.predictors.parameters())
    assert all(x.grad is None for x in feat+weight+bias)
