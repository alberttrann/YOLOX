"""Reconstructed vNext.2 source snapshot, tiny CPU forward/meta integration."""
import os,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['VNEXT2_DUMP_SOURCE']=str(ROOT/'snapshot_src' if (ROOT/'snapshot_src/yolox/models/engram_head.py').exists() else ROOT)
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'prism_ttt'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _snapshot_loader as snap
for name in ['tribrid_neck','darknet','yolo_pafpn','yolox']:
    snap.load('yolox.models.'+name,snap.SOURCE/'yolox/models'/f'{name}.py')
from prism_ttt.bridge import PRISMYOLOX
from prism_ttt.core import PRISMTTT
from prism_ttt.meta import source_view_result,parameter_groups,source_pair_step,optimizer_step
Head=sys.modules['yolox.models.engram_head'].TDE_Head
Body=sys.modules['yolox.models.yolo_pafpn'].YOLOPAFPN
Full=sys.modules['yolox.models.yolox'].YOLOX

def make():
    torch.manual_seed(203)
    body=Body(depth=.33,width=.5,in_channels=[256,512,1024])
    head=Head(9,width=.5,in_channels=[256,512,1024])
    native=Full(body,head).eval().requires_grad_(False)
    native.set_meta_training_state(2,80)
    native.head.training=True
    sidecar=PRISMTTT((128,256,512),rank=8)
    return PRISMYOLOX(native,sidecar)

def sample():
    x=torch.randn(1,3,64,64)
    y=torch.zeros(1,10,5)
    y[0,0]=torch.tensor([2.,30.,30.,22.,22.])
    return x,y

def test_snapshot_identity_and_meta():
    torch.set_num_threads(1)
    model=make();x,y=sample()
    with torch.no_grad():
        native=model.native_features(x)
        out=model.sidecar.adapt(native,torch.zeros(1,20))
    assert all(torch.equal(a,b) for a,b in zip(native,out))
    orig={k:v.detach().clone() for k,v in model.native.state_dict().items()}
    report=source_view_result(model,x,y,meta_train=True)
    print('PARITY',report['parity'])
    print('META task delta',float(report['parts']['total_delta'].detach()))
    print('SUPPORT',report['native_proxy']['gt_support'])
    loss=report['objective']
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.sidecar.ssl_weight_net.parameters() if p.grad is not None)>0,'Missing detector->SSL mixed derivative'
    assert sum(float(p.grad.abs().sum()) for p in model.sidecar.bases.parameters() if p.grad is not None)>0
    assert all(torch.equal(orig[k],v) for k,v in model.native.state_dict().items())
    print('PASS snapshot meta-gradient and native immutability')

def test_snapshot_head_parity():
    torch.set_num_threads(1)
    model=make();x,y=sample()
    with torch.no_grad():
        f=model.native_features(x)
        a=model.head_predictions(f,x,y)
        b=model(x,y,mode='off')
    assert torch.equal(a[0],b[0])
    print('PASS identical native bypass loss')


def test_snapshot_source_training_step():
    torch.set_num_threads(1)
    model=make();x,y=sample();optimizer=torch.optim.AdamW(parameter_groups(model.sidecar,1e-4))
    # Simulated source calibration values; no inference calibration claim.
    model.sidecar.source_calibrate_gradients(torch.rand(32,20)*1e-4)
    start={k:v.detach().clone() for k,v in model.native.state_dict().items()}
    before=model.sidecar.ssl_weight_net[0].weight.detach().clone()
    changed=x*(1+.02*torch.linspace(-1,1,x.shape[2])[None,None,:,None])
    res=source_pair_step(model,x,changed,y,clean_weight=1.,gate_weight=.1)
    norm=optimizer_step(model.sidecar,optimizer)
    assert norm>0
    assert not torch.equal(before,model.sidecar.ssl_weight_net[0].weight.detach())
    assert all(torch.equal(start[k],v) for k,v in model.native.state_dict().items())
    print('PASS one paired-view optimizer update',norm,res)

def test_snapshot_empty_gt():
    torch.set_num_threads(1)
    model=make();x,y=sample();y.zero_()
    report=source_view_result(model,x,y,meta_train=True)
    assert all(v==0 for v in report['native_proxy']['gt_support'].values())
    assert report['native']['counts']['empty_images']==1
    loss=report['objective'];assert torch.isfinite(loss)
    print('PASS snapshot empty GT handling and background source loss')


def test_snapshot_all_gt_scales():
    torch.set_num_threads(1)
    model=make()
    x=torch.randn(1,3,128,128)
    y=torch.zeros(1,10,5)
    y[0,0]=torch.tensor([2.,25.,25.,14.,14.])
    y[0,1]=torch.tensor([3.,65.,65.,52.,45.])
    y[0,2]=torch.tensor([1.,63.,65.,108.,104.])
    r=source_view_result(model,x,y,meta_train=False)
    assert r['native_proxy']['gt_support']=={'small':1,'medium':1,'large':1}
    assert all(r['native_proxy']['scale_gt_detection'][k] is not None for k in ('small','medium','large'))
    assert all(r['native_proxy']['scale_proxy'][k] is not None for k in ('small','medium','large'))
    assert torch.isfinite(r['objective'])
    print('PASS all-GT small/medium/large coverage and GT-balanced risk')

if __name__=="__main__":
    for k,v in list(globals().items()):
        if k.startswith("test_") and callable(v):v();print("PASS",k)
