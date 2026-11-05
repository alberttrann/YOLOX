"""Actual supplied source snapshot: CSPDarknet -> Tribrid PAFPN -> TDE_Head.

This is deliberately small (64px CPU) and unlabeled weights, so it verifies
integration, not BDD100K AP or adverse generalization.
"""
import os,sys,types,torch
from pathlib import Path
from test_actual_tde_head_fixed_targets import SOURCE,load,utils
ROOT=Path(__file__).resolve().parents[1]
for name in ['tribrid_neck','darknet','yolo_pafpn','yolox']:
    load('yolox.models.'+name,SOURCE/'yolox/models'/f'{name}.py')
from native_dark2_router import Dark2AuditRouter
from fixed_detection_targets import NativeMatchReplay
from safe_ttt_model import SafeSpatialTTT
from audit_safe_ttt_yolox import audit_one,get_size_from_native
from pilot_safe_ttt_meta import fixed_native_targets_step
Head=sys.modules['yolox.models.engram_head'].TDE_Head
Backbone=sys.modules['yolox.models.yolo_pafpn'].YOLOPAFPN
Full=sys.modules['yolox.models.yolox'].YOLOX

def test_full_forward_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(999)
    body=Backbone(depth=.33,width=.5,in_channels=[256,512,1024])
    head=Head(9,width=.5,in_channels=[256,512,1024])
    model=Full(body,head)
    model.eval().requires_grad_(False)
    model.set_meta_training_state(2,80)
    model.head.training=True
    images=torch.randn(1,3,64,64)
    labels=torch.zeros(1,10,5)
    labels[0,0]=torch.tensor([2.,30.,30.,22.,22.])
    C=get_size_from_native(model,images)
    assert C==64
    safe=SafeSpatialTTT(C,rank=4,step_policy="normalized",basis_gain=16.0).eval()
    with torch.no_grad():
        initial={k:v.clone() for k,v in model.state_dict().items()}
    report=audit_one(model,safe,images,labels,meta_probe=True)
    assert report['matching']['fg']['foreground']>0
    assert len(report['gradients'])>=10
    assert report['meta_probe']['detector_to_ssl_predictor_grad_norm']>0
    assert report['fast_state_norm'][0]>0
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    assert report['native_parity']['total']<1e-4
    print('PASS: full vNext2 snapshot backbone/neck/TDE head/native replay/actual detector meta gradient')
    print('META_COMPARISON',report['meta_probe'])
    print('RAW_LOGIT_EFFECT',report['prediction_effects'])
    print('ALIGNMENT',report['gradients']['total'])
    print('TASK_COMPONENT_DIFFS', {k: report['updated_losses'][k]-v for k,v in report['native_losses'].items()})
    print('TTT_SIGNALS', {k:report[k] for k in ('fast_state_norm','ssl_initial','grad_norm','relative_feature_change_postcast','postcast_changed_fraction')})
    optimizer=torch.optim.AdamW(safe.parameters(),lr=5e-4)
    before=next(safe.predictor.parameters()).detach().clone()
    tr=fixed_native_targets_step(model,safe,images,labels,optimizer)
    assert tr['meta_predictor_grad_norm']>0
    assert not torch.equal(before,next(safe.predictor.parameters()).detach())
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    print('PILOT',tr)
    tr_bal=fixed_native_targets_step(model,safe,images,labels,optimizer,objective_mode='scale_balanced')
    assert tr_bal['meta_predictor_grad_norm']>0
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    print('BALANCED_PILOT',tr_bal)
    print('TASK_CHANGE',report['updated_losses']['total']-report['native_losses']['total'],
        'GRAD_COS',report['gradients']['total']['cosine'],
        'PREDICTOR_MIXED_GRAD',report['meta_probe']['detector_to_ssl_predictor_grad_norm'])


def test_full_snapshot_inference_bypass_and_batch_independence():
    torch.set_num_threads(1)
    torch.manual_seed(2031)
    body=Backbone(depth=.33,width=.5,in_channels=[256,512,1024])
    head=Head(9,width=.5,in_channels=[256,512,1024])
    model=Full(body,head).eval().requires_grad_(False)
    model.set_meta_training_state(2,80)
    A=torch.randn(1,3,64,64)
    B=torch.randn(1,3,64,64)+1.5
    C=get_size_from_native(model,A)
    safe=SafeSpatialTTT(C,rank=4,step_policy='normalized',basis_gain=16).eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with Dark2AuditRouter(model,safe) as router:
        router.set_mode('native')
        with torch.no_grad():native=model(A)
        router.require_one_hit(1)
        router.set_mode('episode',meta_train=False)
        with torch.no_grad():
            solo=model(A)
            ab=model(torch.cat([A,B],0))
            aa=model(torch.cat([A,A],0))
        assert torch.allclose(solo,ab[0],atol=1e-5,rtol=1e-5)
        assert torch.allclose(ab[0],aa[0],atol=1e-5,rtol=1e-5)
        assert solo.shape==native.shape
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    print('PASS: full snapshot decoded inference, independent batch composition, native state immutable')

if __name__=="__main__":
    test_full_forward_and_gradients()
    test_full_snapshot_inference_bypass_and_batch_independence()
