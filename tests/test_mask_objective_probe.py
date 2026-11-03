"""Standalone CPU tests for the read-only masked/unmasked TTT gradient probe."""
from pathlib import Path
import importlib.util
import sys
import os

import torch

HERE=Path(__file__).resolve().parents[1]
p=HERE/'tools'/'diagnose_vnext2_ttt_objective.py'
s=importlib.util.spec_from_file_location('mask_probe',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

# Reuse model class from the self-contained TTT per-image test bootstrap.
t=Path(__file__).parent/'test_vnext2_per_image_eval.py'
spec=importlib.util.spec_from_file_location('per_image_test',t)
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)


def test_masked_visible_decomposition():
    net=ref.model().eval().requires_grad_(False)
    x=torch.randn(1,8,12,16)
    with torch.no_grad():
        feats=net.backbone_stage(x)
    report=m.probe_inner_objective(net,feats)
    assert 0<report['masked_fraction']<1
    assert abs(report['loss_masked_contribution']+
               report['loss_visible_contribution']-report['loss_total'])<1e-6
    assert report['gradient_additivity_absolute_error']<1e-6
    assert report['full_grad_norm']>0
    assert report['masked_grad_norm']>=0
    assert report['visible_grad_norm']>=0
    assert report['masked_output_spatial_variance'] is not None
    assert report['masked_target_spatial_variance'] is not None
    assert report['masked_output_spatial_variance'] < 1e-10
    assert report['masked_target_spatial_variance'] > 1e-12
    print('PASS masked/visible loss and gradients, spatial mask-token limitation')


if __name__=='__main__':
    test_masked_visible_decomposition()
