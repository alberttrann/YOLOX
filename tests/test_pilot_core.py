import importlib.util
from pathlib import Path
import sys
import pytest

MOD=Path(__file__).resolve().parents[1] / 'tools' / '_pgsma_pilot_core.py'
spec=importlib.util.spec_from_file_location('_pgsma_pilot_core',MOD)
core=importlib.util.module_from_spec(spec)
sys.modules['_pgsma_pilot_core']=core
spec.loader.exec_module(core)

@pytest.mark.parametrize('steps,eval_images,batch,interval', [
  (0,64,1,10),(2049,64,1,10),(5,0,1,1),(5,513,1,1),(5,2,0,1),(5,2,9,1),(5,2,1,0)])
def test_budget_denies_excess(steps,eval_images,batch,interval):
    with pytest.raises(ValueError): core.validate_budget(steps,eval_images,batch,interval)

def test_budget_accepts():
    core.validate_budget(200,64,2,50)

def test_scales():
    assert [core.group_scale(i) for i in (2.0,5.0,9.0,18.0,33.0)]==['lt4','4to8','8to16','16to32','ge32']

def test_unique_matches_and_confidence_order():
    gt=[(0,[0,0,10,10],0),(0,[20,20,30,30],0)]
    pred=[(0,[0,0,10,10],0.8),(0,[0,0,10,10],0.7),(0,[20,20,30,30],0.6)]
    assert core.greedy_recall_pairs(gt,pred)=={0,1}

def test_class_disagreement():
    assert not core.greedy_recall_pairs([(1,[0,0,10,10],0)], [(0,[0,0,10,10],0.9)])

class Param:
 def __init__(self): self.requires_grad=True
class Model:
 def __init__(self):
  self.native=[Param()];self.semantic=[Param()]
  self.head=type('Head',(),{})()
 def native_parameters(self): return self.native
 def semantic_parameters(self): return self.semantic
 def set_inference_mode(self, m): self.mode=m
class Optim:
 def __init__(self,params): self.param_groups=[{'params':params}]

def test_param_ownership():
    m=Model();o=Optim(m.native);s=Optim(m.semantic)
    assert core.model_parameter_split(m,o,s)==m.native
    with pytest.raises(RuntimeError): core.model_parameter_split(m,o,Optim(m.native))

def test_variants():
    m=Model()
    for variant,matcher,mode in [('native_simota','baseline','native'),('native_scpa','scpa','native'),('native_scpa_instance','scpa','native'),('memory','scpa','memory'),('pgsma','scpa','pgsma')]:
        assert core.configure_variant(m,variant)==mode
        assert m.head.assignment==matcher
    assert core.configure_variant(m,'native_scpa_instance')=='native'
    assert m.head.per_gt_normalization
