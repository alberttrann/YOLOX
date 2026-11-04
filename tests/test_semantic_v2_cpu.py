"""CPU-only contract tests for PG-SMA semantic-v2 candidate remediation.

Run: python -m pytest -q tests/test_semantic_v2_cpu.py
No dataset, checkpoint, CUDA or full YOLOX package is needed for these probes.
"""
from pathlib import Path
import ast
import importlib.util
import torch

ROOT = Path(__file__).resolve().parents[1]


def _pgsma():
    spec = importlib.util.spec_from_file_location('pgsma_candidate_v2', ROOT / 'yolox/models/pgsma.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PGSMA


def _negative_selector():
    # Pull out the method without importing heavy YOLOX dependencies.
    tree = ast.parse((ROOT / 'yolox/models/pgsma_head.py').read_text())
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PGSMAHead')
    method = next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == '_hard_background')
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'torch':torch}
    exec(compile(module, 'extracted_negative_selector', 'exec'), namespace)
    return namespace['_hard_background']


def _fixtures():
    torch.manual_seed(19)
    features = [torch.randn(1, 16, 8, 8),torch.randn(1,16,4,4),torch.randn(1,16,2,2)]
    w = [torch.randn(3,16,1,1)*.01 for _ in features]
    biases = [torch.randn(3)*.01 for _ in features]
    native = [torch.nn.functional.conv2d(f,a,b) for f,a,b in zip(features,w,biases)]
    obj = [torch.randn(1,1,*f.shape[-2:]) for f in features]
    # Three GTs covering micro/medium/macro effective-size bins.
    match = {'fg_ids':torch.tensor([2,3,5,66,75,82]),
             'fg_labels':torch.tensor([0,0,1,1,2,2]),
             'fg_quality':torch.tensor([.6,.7,.8,.8,.7,.9]),
             'gt_uid':torch.tensor([0,0,1,1,2,2]),
             'fg_sizes':torch.tensor([6.,6.,12.,12.,43.,43.]),
             'bg_ids':torch.tensor([9,10,11,12,14,20,24,29,31,40,46,51,58,61,69,80])}
    return features, native,obj,w,biases,match


def test_identity_at_initialization():
    PGSMA=_pgsma()
    m=PGSMA(classes=3,channels=16,dim=8,modes=2)
    features,native,obj,*_= _fixtures()
    out=m.forward_levels(features,native,obj)
    for a,b in zip(out,native):
        assert torch.equal(a,b), 'Zero-initialized predictor must preserve native output exactly.'
    assert all(float(torch.sigmoid(g[-1].bias).mean().detach())<0.05 for g in m.gates)


def test_grad_routes_and_differentiability():
    PGSMA=_pgsma()
    m=PGSMA(classes=3,channels=16,dim=8,modes=2)
    features,native,obj,w,biases,match=_fixtures()
    for tensor in features+native+obj+w+biases:
        tensor.requires_grad_(True)
    losses=m.semantic_losses(features,list(zip(w,biases)),native,obj,match,seed=121)
    for key,value in losses.items():
        assert torch.isfinite(value).all(), key
    total=losses['prototype']+losses['teacher']+losses['candidate']+losses['gate']
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.predictors.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.gates.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.projectors.parameters())
    assert all(p.grad is None for p in m.teacher_projectors.parameters())
    assert all(t.grad is None for t in features+obj+w+biases)
    # Native logits are non-leaf and can retain grad only if explicitly requested.
    assert all(t.grad is None for t in native)


def test_separate_optimizer_step_and_teacher():
    PGSMA=_pgsma()
    m=PGSMA(classes=3,channels=16,dim=8,modes=2)
    features,native,obj,w,biases,match=_fixtures()
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=0.001)
    before=[p.detach().clone() for p in m.predictors.parameters()]
    optimizer.zero_grad(set_to_none=True)
    v=m.semantic_losses(features,list(zip(w,biases)),native,obj,match)
    (v['prototype']+v['candidate']+v['gate']+v['teacher']).backward()
    optimizer.step()
    m.update_teacher()
    assert any(not torch.equal(a,p) for a,p in zip(before,m.predictors.parameters()))


def test_hard_negative_excludes_gt_and_halo():
    select=_negative_selector()
    # Two images, 10x10 anchor grid; highest-objectness boxes sit inside GT.
    xy=torch.tensor([[4.+8*x,4.+8*y] for y in range(10) for x in range(10)])
    obj=torch.zeros(2,100,1)
    obj[:,44:56]=10
    positives=torch.zeros((2,100),dtype=torch.bool)
    boxes=[torch.tensor([[44.,44.,40.,40.]]), torch.tensor([[44.,44.,40.,40.]])]
    bg=select(obj,positives,None,max_neg=15,centers=xy,gt_by_image=boxes,halo=1.25)
    for b in range(2):
        ids=bg[bg//100==b]%100
        ctr=xy[ids]
        assert not (((ctr-boxes[b][0,:2]).abs() <= boxes[b][0,2:]*1.25/2).all(-1)).any()
        assert len(ids)==15


def test_dynamic_geometry_negative_selector():
    select=_negative_selector()
    for h,w in [(7,11),(16,13),(37,41)]:
        xy=torch.tensor([[4.+8*x,4.+8*y] for y in range(h) for x in range(w)])
        obj=torch.randn(1,len(xy),1)
        pos=torch.zeros((1,len(xy)),dtype=torch.bool)
        res=select(obj,pos,None,max_neg=20,centers=xy,gt_by_image=[torch.tensor([[38.,26.,12.,12.]])])
        assert res.ndim==1 and len(res)<=20
        assert (res<len(xy)).all()


def test_exact_init_retraction_and_memory():
    PGSMA=_pgsma()
    features,native,obj,w,biases,match=_fixtures()
    for correction in ('direct','retraction'):
        model=PGSMA(classes=3,channels=16,dim=8,modes=2,correction=correction)
        for mode in ('pgsma','memory'):
            outputs=model.forward_levels(features,native,obj,mode=mode)
            assert all(torch.equal(v,original) for v,original in zip(outputs,native))


def test_no_foreground_and_dynamic_shapes():
    PGSMA=_pgsma()
    model=PGSMA(classes=3,channels=16,dim=8,modes=2)
    features=[torch.randn(1,16,7,11),torch.randn(1,16,4,6),torch.randn(1,16,2,3)]
    w=[torch.randn(3,16,1,1)*0.01 for _ in features]
    biases=[torch.randn(3)*.01 for _ in features]
    native=[torch.nn.functional.conv2d(f,a,b) for f,a,b in zip(features,w,biases)]
    obj=[torch.randn(1,1,*f.shape[-2:]) for f in features]
    empty=torch.empty(0,dtype=torch.long)
    match={'fg_ids':empty,'fg_labels':empty,'fg_quality':torch.empty(0),
           'gt_uid':empty,'fg_sizes':torch.empty(0),'bg_ids':torch.tensor([0,3,6,78,101])}
    outs=model.forward_levels(features,native,obj)
    assert all(out.shape==inp.shape for out,inp in zip(outs,native))
    losses=model.semantic_losses(features,list(zip(w,biases)),native,obj,match)
    assert all(torch.isfinite(v).all() for v in losses.values())
    (losses['prototype']+losses['candidate']+losses['gate']+losses['teacher']).backward()


def test_native_only_warmstart_filters_legacy_semantic_weights():
    # Isolate the real loader function so that the stripped textual checkout
    # does not need the external original YOLOX package.
    tree=ast.parse((ROOT/'tools/pgsma_bounded_pilot.py').read_text())
    method=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load_weights')
    ns={'safe_checkpoint_state': lambda filename: (saved,weights),'file_digest':lambda filename:'hash'}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'loader_test','exec'),ns)
    class Model(torch.nn.Module):
        def __init__(self,width):
            super().__init__()
            self.backbone=torch.nn.Linear(2,2)
            self.head=torch.nn.Module()
            self.head.pgsma_engine=torch.nn.Linear(width,3)
    legacy=Model(3)
    revised=Model(4)
    revised_head_before=revised.head.pgsma_engine.weight.detach().clone()
    saved={'pgsma_pilot':{'variant':'native_simota','successful_native_updates':512}}
    weights=legacy.state_dict()
    result=ns['load_weights'](revised,'pretend_native_pilot.pth',True)
    assert 'backbone.weight' in revised.state_dict()
    assert torch.equal(revised.backbone.weight,legacy.backbone.weight)
    assert torch.equal(revised.head.pgsma_engine.weight,revised_head_before)
    assert result['is_pilot_warmstart']
    broken=dict(weights)
    broken['backbone.weight']=torch.randn(3,3)
    weights=broken
    try:
        ns['load_weights'](Model(4),'broken_native.pth',True)
    except RuntimeError as e:
        assert 'Native warm-start mismatch' in str(e)
    else:
        raise AssertionError('Incompatible native weights must fail closed')
