import copy
import torch
import torch.nn.functional as F


def _fake_records(b, sizes, ncls):
    n=sum(h*w for h,w in sizes)
    device='cpu'
    pos=torch.tensor([4, 10, n+1, n+8],device=device)
    bg=torch.tensor([2,3,7,n+5,n+7],device=device)
    return dict(fg_ids=pos, fg_labels=torch.tensor([0,1% ncls,0,1%ncls]),
                fg_quality=torch.tensor([.9,.2,.7,.4]),
                gt_uid=torch.tensor([0,1,2,3]), bg_ids=bg)


def test_identity_geometry_and_bounds(src):
    torch.manual_seed(13)
    P=src['pgsma'].PGSMA(3,channels=12,dim=8,modes=4)
    features=[torch.randn(2,12,8,8),torch.randn(2,12,4,4),torch.randn(2,12,2,2)]
    classes=[torch.randn(2,3,*x.shape[2:]) for x in features]
    objs=[torch.randn(2,1,*x.shape[2:]) for x in features]
    out, d=P.forward_levels(features,classes,objs,collect=True)
    assert all(torch.max((y-x).abs()) < 2e-5 for x,y in zip(classes,out))
    assert all(torch.isfinite(y).all() for y in out)
    assert all((v['delta'].abs() <= P.budget+1e-6).all() for v in d)
    for mode in ('native','memory'):
        result=P.forward_levels(features,classes,objs,mode=mode)
        assert len(result)==3
    # No eval state update (per image, no optimizer).
    state=copy.deepcopy(P.state_dict())
    P.eval()
    with torch.no_grad(): P.forward_levels(features, classes, objs)
    assert all(torch.equal(x,state[k]) for k,x in P.state_dict().items())


def test_gradient_ownership_and_training_step(src):
    torch.manual_seed(30)
    P=src['pgsma'].PGSMA(2,channels=12,dim=8,modes=3)
    features=[torch.randn(2,12,8,8,requires_grad=True),
              torch.randn(2,12,4,4,requires_grad=True),
              torch.randn(2,12,2,2,requires_grad=True)]
    native=torch.nn.ModuleList([torch.nn.Conv2d(12,2,1) for _ in range(3)])
    clss=[m(x) for m,x in zip(native,features)]
    objs=[torch.randn(2,1,*x.shape[-2:]) for x in features]
    sizes=[x.shape[2:] for x in features]
    match=_fake_records(2,sizes,2)
    losses=P.semantic_losses(features,[(m.weight,m.bias) for m in native],
                             clss,objs,match,seed=400)
    assert all(torch.isfinite(v) for v in losses.values())
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    fn=Path(__file__).resolve().parents[1]/'overlay/yolox/core/pgsma_step.py'
    spec=spec_from_file_location('pgstep',fn);module=module_from_spec(spec);spec.loader.exec_module(module)
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__();self.semantic=P;self.native=native;self.features=features;self.clss=clss
        def semantic_losses(self,seed,correction_mode):
            return P.semantic_losses(features,[(m.weight,m.bias) for m in native],
                                     clss,objs,match,seed=seed,correction_mode=correction_mode)
    W=Wrapper()
    opt=torch.optim.AdamW(P.parameters(),lr=.01)
    before=[p.detach().clone() for p in native.parameters()]
    done,m=module.semantic_optimizer_step(W,opt,seed=400)
    assert done,m
    assert all(torch.equal(x,p) for x,p in zip(before,native.parameters()))
    assert all(x.grad is None for x in features)
    assert all(p.grad is None for p in native.parameters())
    assert int(P.semantic_step)==1
    assert P.predictors[0].out.weight.grad is not None
    assert torch.isfinite(P.predictors[0].out.weight.grad).all()
    assert P.predictors[0].out.weight.grad.abs().sum().item()>0
    # Gate begins acquiring gradient after nonzero correction has developed.
    done,m=module.semantic_optimizer_step(W,opt,seed=401)
    assert done
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in P.gates.parameters())


def test_memory_score_permutation_and_one_class(src):
    torch.manual_seed(7)
    P=src['pgsma'].PGSMA(1,channels=4,dim=8,modes=4)
    z=F.normalize(torch.randn(1,8,2,3),dim=1)
    left=P._prototype_scores(z).detach()
    with torch.no_grad(): P.prototypes.copy_(P.prototypes[:,torch.tensor([2,0,3,1])])
    assert torch.allclose(left,P._prototype_scores(z),atol=1e-6)
    assert P._sample_losses(torch.randn(1,5,1),torch.tensor([1]),
                           torch.tensor([0]),torch.tensor([.9]),torch.tensor([1]),
                           torch.tensor([0,3]),True)[0].isfinite()


def test_intervention_on_context(src):
    P=src['pgsma'].PGSMA(3,channels=8,dim=8)
    x=torch.randn(1,8,8,8)
    z,_=P._project(0,x)
    # Changing a neighbor impacts the predictor when its terminal layer is enabled.
    with torch.no_grad(): P.predictors[0].out.weight.normal_(std=.02)
    original=P.predictors[0](z)
    shuffled=torch.roll(z,shifts=1,dims=3)
    alternate=P.predictors[0](shuffled)
    assert not torch.allclose(original,alternate)


def test_per_image_batch_independence_and_half_input(src):
    torch.manual_seed(401)
    P=src['pgsma'].PGSMA(3,channels=8,dim=8,modes=3).eval()
    features=[torch.randn(2,8,8,8),torch.randn(2,8,4,4),torch.randn(2,8,2,2)]
    logits=[torch.randn(2,3,*x.shape[-2:]) for x in features]
    objectness=[torch.randn(2,1,*x.shape[-2:]) for x in features]
    with torch.no_grad():
        combined=P.forward_levels(features,logits,objectness)
        solo=P.forward_levels([x[:1] for x in features],
                              [x[:1] for x in logits],[x[:1] for x in objectness])
        half=P.forward_levels([x.half() for x in features],
                              [x.half() for x in logits],[x.half() for x in objectness])
    assert all(torch.allclose(y[:1],z,atol=3e-5,rtol=3e-5) for y,z in zip(combined,solo))
    assert all(torch.isfinite(y).all() and y.dtype==torch.float16 for y in half)


def test_individual_loss_gradient_ownership(src):
    P=src['pgsma'].PGSMA(2,channels=8,dim=8,modes=2)
    x=[torch.randn(1,8,4,4),torch.randn(1,8,2,2),torch.randn(1,8,1,1)]
    native=[torch.nn.Conv2d(8,2,1) for _ in x]
    cl=[m(t) for m,t in zip(native,x)]
    ob=[torch.randn(1,1,*t.shape[-2:]) for t in x]
    match={'fg_ids':torch.tensor([0,3]),'fg_labels':torch.tensor([0,1]),
           'fg_quality':torch.tensor([.8,.4]),'gt_uid':torch.tensor([0,1]),
           'bg_ids':torch.tensor([1,5,9])}
    losses=P.semantic_losses(x,[(m.weight,m.bias) for m in native],cl,ob,match,seed=7)
    named={n:p for n,p in P.named_parameters() if p.requires_grad}
    expected={
      'prototype':('projectors.','prototypes','log_concentration'),
      'teacher':('predictors.',),
      'candidate':('predictors.','log_gain'),
    }
    for objective,prefixes in expected.items():
        grads=torch.autograd.grad(losses[objective],named.values(),retain_graph=True,
                                  allow_unused=True)
        active=[name for (name,_),g in zip(named.items(),grads)
                if g is not None and g.abs().sum().item()>1e-12]
        assert active,objective
        assert all(any(n.startswith(prefix) for prefix in prefixes) for n in active), (objective,active)
    # Gate loss is identity at zero movement. Once the predictor moves it may
    # train only the gate, never the candidate or the reference geometry.
    with torch.no_grad(): P.predictors[0].out.weight.normal_(std=.1)
    losses=P.semantic_losses(x,[(m.weight,m.bias) for m in native],cl,ob,match,seed=8)
    grads=torch.autograd.grad(losses['gate'],named.values(),allow_unused=True)
    active=[name for (name,_),g in zip(named.items(),grads)
            if g is not None and g.abs().sum().item()>1e-12]
    assert active and all(n.startswith('gates.') for n in active), active
