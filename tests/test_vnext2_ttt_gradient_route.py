"""CPU tests for the non-mutating vNext.2 spatial gradient-route ablation."""
import importlib.util
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parents[1]
def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

probe=load(BASE/'tools'/'probe_vnext2_ttt_gradient_route.py','gradient_route_probe')
# Uses existing test_vnext2_per_image_eval.py already installed by the user.
existing=load(Path(__file__).parent/'test_vnext2_per_image_eval.py','old_vnext2_test')

def test_route_with_actual_vnext2_classes():
    torch.manual_seed(2026)
    net=existing.model().eval().requires_grad_(False)
    x=torch.randn(1,8,12,16)
    with torch.no_grad():
        features=net.backbone_stage(x)
    saved={k:v.detach().clone() for k,v in net.state_dict().items()}
    result=probe.probe_gradient_route(net,features)
    names=['original_token','neighbor_fill_detached_control','neighbor_fill_differentiable']
    for key in names:
        v=result[key]
        assert v['gradient_additivity_error']<1e-6,(key,v['gradient_additivity_error'])
        assert v['full_loss']>0
        assert v['masked_gradient_norm']>=0
        assert v['masked_prediction_spatial_variance'] is not None
    route=result['gradient_path_isolation']
    assert route['forward_equivalence_between_context_and_detached']==0.0
    assert route['direct_neighbor_route_gradient_norm']>0
    assert result['mask_and_context']['fraction_masked_sites_with_visible_neighbor']>0
    assert result['original_token']['masked_prediction_spatial_variance']<1e-10
    assert result['neighbor_fill_differentiable']['masked_prediction_spatial_variance']>0
    assert all(torch.equal(v,saved[k]) for k,v in net.state_dict().items())
    print('PASS: direct masked gradient routing, equal-value detach control, no state mutation')

if __name__=='__main__':
    test_route_with_actual_vnext2_classes()
