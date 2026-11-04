"""Standalone CPU helper checks; no dataset/checkpoint needed."""
import importlib.util
from pathlib import Path
import torch

src=Path(__file__).resolve().parents[1]/'tools'/'audit_pgsma_gate_geometry.py'
spec=importlib.util.spec_from_file_location('g3',src)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_auc_perfect_and_inverted():
    g=torch.tensor([.95,.80,.20,.10]); good=torch.tensor([True,True,False,False]); bad=~good
    assert m._auc_gate(g,good,bad)==1.0
    assert m._auc_gate(1-g,good,bad)==0.0


def test_auc_ties_and_undefined():
    g=torch.tensor([.5,.5,.5,.5]); good=torch.tensor([True,True,False,False]); bad=~good
    assert m._auc_gate(g,good,bad)==.5
    assert m._auc_gate(g,good,~good & False) is None


def test_margin():
    cls=torch.tensor([[1.,3.,2.],[2.,1.,0.]])
    labels=torch.tensor([1,0]); result=m._class_margin(cls,labels)
    assert torch.allclose(result,torch.tensor([1.,1.]))


def test_quantiles():
    x=torch.tensor([0.,1.,2.,3.])
    assert m._quantiles(x)['mean']==1.5
    assert m._quantiles(x.new_empty(0)) is None
