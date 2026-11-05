import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_ttt.core import PRISMTTT,relations

torch.set_num_threads(1)

def features(n=1):
    return tuple(torch.randn(n,c,max(2,16//(2**i)),max(2,16//(2**i))) for i,c in enumerate((16,24,32)))

def make():return PRISMTTT((16,24,32),rank=4).float()

def test_identity():
    torch.manual_seed(123);m=make();fs=features(2);z=torch.zeros(2,20)
    for f,g in zip(fs,m.adapt(fs,z)):assert torch.equal(f,g)
    out,_=m(fs,mode='off')
    for f,g in zip(fs,out):assert torch.equal(f,g)

def test_batch_and_state():
    torch.manual_seed(444);m=make().eval().requires_grad_(False);fs=features(2)
    ck={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad():
        p=m.propose(fs)
        a=m.propose(tuple(x[:1] for x in fs))
        b=m.propose(tuple(x[1:] for x in fs))
    assert torch.allclose(p.state[:1],a.state,atol=1e-6,rtol=1e-5)
    assert torch.allclose(p.state[1:],b.state,atol=1e-6,rtol=1e-5)
    assert all(torch.equal(ck[k],v) for k,v in m.state_dict().items())

def test_meta_gradient():
    torch.manual_seed(125);m=make();fs=features()
    p=m.propose(fs,meta_train=True)
    pred=m.apply(fs,p,'on',require_calibration=False)[0]
    # Simulated head loss; real TDE head integration in separate test.
    loss=sum((x*x).mean() for x in pred)
    g=torch.autograd.grad(loss,tuple(m.ssl_weight_net.parameters()),allow_unused=True)
    assert sum(float(v.abs().sum()) for v in g if v is not None)>0

def test_calibration_and_failclosed():
    torch.manual_seed(127);m=make().eval();fs=features()
    p=m.propose(fs,meta_train=False)
    try:m.apply(fs,p,'on')
    except RuntimeError:pass
    else:raise AssertionError('Uncalibrated online adaptation accepted')
    m.source_calibrate_gradients(torch.randn(24,20)*1e-5)
    out,g=m.apply(fs,p,'auto')
    assert all(torch.equal(f,x) for f,x in zip(fs,out)) and g.sum()==0
    assert not bool(m.gate.calibrated)

def test_budgets():
    m=make().eval();fs=features();z=torch.randn(1,20)*5
    out=m.adapt(fs,z)
    for f,x in zip(fs,out):
        diff=(x-f).square().mean(1,keepdim=True).sqrt()
        bound=.06*(f.square().mean(1,keepdim=True).sqrt()+.01)
        assert torch.all(diff<=bound+1e-6)

def test_ssl_nonzero_and_eval():
    m=make().eval().requires_grad_(False);fs=features()
    with torch.no_grad():
        p=m.propose(fs)
        assert torch.isfinite(p.state).all()
        assert torch.isfinite(p.ssl_before).all()
        assert p.grad.norm()>0

def test_inference_mode_refusal():
    m=make().eval();fs=features()
    with torch.inference_mode():
        try:m.propose(fs)
        except RuntimeError:pass
        else:raise AssertionError('inference_mode must be refused')


def test_reliability_calibration_and_identity():
    from prism_ttt.source_protocol import gate_calibrate_from_reports
    m=make().eval()
    fs=features()
    m.source_calibrate_gradients(torch.randn(32,20)*1e-5)
    # No successful source adaptation -> calibrated but all scales abstain.
    records=[([.65,.68,.77],[0,0,0],[1,1,1]) for _ in range(24)]
    report=gate_calibrate_from_reports(m,records)
    assert report['all_rejected']
    with torch.no_grad():
        p=m.propose(fs)
        out,g=m.apply(fs,p,'auto')
    assert g.sum()==0 and all(torch.equal(f,o) for f,o in zip(fs,out))

def test_invalid_image_isolated():
    torch.manual_seed(85);m=make().eval().requires_grad_(False)
    f=features(2);f=tuple(x.clone() for x in f)
    f[1][1,0,0,0]=float('nan')
    with torch.no_grad():p=m.propose(f)
    assert bool(p.valid[0]) and not bool(p.valid[1])
    adapted,r=m.apply(f,p,'on',require_calibration=False)
    assert r[1].sum()==0
    # Invalid source is returned, unchanged including original NaN positions.
    for old,new in zip(f,adapted):
        assert torch.equal(torch.nan_to_num(old[1]),torch.nan_to_num(new[1]))

def test_state_is_per_sample_not_batch_norm():
    torch.manual_seed(222);m=make().eval().requires_grad_(False);f=features(2)
    with torch.no_grad():
        a=m.propose(f)
        b=m.propose(tuple(x[[1,0]] for x in f))
    assert torch.allclose(a.state,b.state[[1,0]],atol=1e-6,rtol=1e-6)

if __name__=="__main__":
    for k,v in list(globals().items()):
        if k.startswith("test_") and callable(v):v();print("PASS",k)
