"""Numerical step gates and reversible routing in old/current TTT method shapes."""
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from safe_ttt_model import SafeSpatialTTT
from native_dark2_router import Dark2AuditRouter

def test_normalized_state_amplitude_and_zero_gradient_abstention():
    torch.manual_seed(7)
    m=SafeSpatialTTT(8,rank=4,step_policy='normalized',min_signal_norm=1e-8)
    grads=torch.tensor([[1e-6,1e-6,0.,0.],[0.,0.,0.,0.]])
    step=m.compute_step(grads)
    expected=m.max_state_step*float(torch.sigmoid(m.step_logit).detach())
    assert abs(float(step[0].norm().detach())-expected)<1e-6
    assert torch.equal(step[1],torch.zeros_like(step[1]))
    m.step_policy='legacy_floor'
    legacy=m.compute_step(grads)
    assert float(legacy[0].norm().detach())<float(step[0].norm().detach())/10
    print('PASS: gradient-floor regression, normalized step, zero-signal abstention')

class OldStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_stage=nn.Conv2d(8,8,1)
    def forward(self,x,run_ttt=True):
        return self.backbone_stage(x)*(-3 if run_ttt else 1)

class CurrentStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_stage=nn.Conv2d(8,8,1)
    def forward(self,x,run_ttt=True):
        return self._forward_residual_ttt(x,run_ttt)
    def forward_from_features(self,feat,run_ttt=True):
        return self._forward_residual_ttt(None,run_ttt,precomputed_feat=feat)
    def _forward_residual_ttt(self,x_in,run_ttt=True,precomputed_feat=None):
        x=self.backbone_stage(x_in) if precomputed_feat is None else precomputed_feat
        return x*-3 if run_ttt else x

class MockModel(nn.Module):
    def __init__(self,stage):
        super().__init__()
        self.backbone=nn.Module();self.backbone.backbone=nn.Module()
        self.backbone.backbone.dark2=stage

def test_router_preserves_method_and_intercepts_both_layouts():
    torch.manual_seed(2)
    for stage_type in (OldStage,CurrentStage):
        model=MockModel(stage_type())
        stage=model.backbone.backbone.dark2
        x=torch.randn(1,8,4,4)
        prior=stage(x,run_ttt=True)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with Dark2AuditRouter(model) as router:
            router.set_mode('native')
            y=stage(x,run_ttt=True)
            router.require_one_hit(1)
            assert torch.equal(y,stage.backbone_stage(x))
            if isinstance(stage,CurrentStage):
                feat=stage.backbone_stage(x)
                router.set_mode('native')
                yz=stage.forward_from_features(feat,run_ttt=True)
                router.require_one_hit(1)
                assert torch.equal(yz,feat)
        after=stage(x,run_ttt=True)
        assert torch.equal(prior,after)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    print('PASS: old GN and current residual forwarding, method restoration')

if __name__=='__main__':
    torch.set_num_threads(1)
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):fn()
