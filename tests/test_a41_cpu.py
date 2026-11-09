"""Real CPU full model tests against the included archived YOLOX source ABI.

Uses the user's archived code, not a dummy nn.Sequential detector; genuine
CSP->SPI->Tribrid->P3/P4/P5->SimOTA->multi-loss backward.
"""
from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'reference'
BASE = ARCHIVE / 'reference/vnext2_flat_source' if (ARCHIVE/'reference').exists() else ARCHIVE/'vnext2_flat_source'
LEGACY = ARCHIVE / 'reference/base_overlay' if (ARCHIVE/'reference').exists() else ARCHIVE/'base_overlay'
A4 = ARCHIVE / 'overlay'
NEW = ROOT / 'overlay'

py = types.ModuleType('yolox'); py.__path__ = [str(BASE/'yolox')]; sys.modules['yolox'] = py
models = types.ModuleType('yolox.models'); models.__path__ = [
    str(NEW/'yolox/models'), str(A4/'yolox/models'),
    str(LEGACY/'yolox/models'), str(BASE/'yolox/models')]
sys.modules['yolox.models'] = models
u=types.ModuleType('yolox.utils')
def meshgrid(xs):return torch.meshgrid(*xs,indexing='ij')
def xyxy(x):return torch.cat((x[...,:2]-x[...,2:]/2,x[...,:2]+x[...,2:]/2),-1)
def iou(a,b,xyxy_format=True):
    if not xyxy_format:a,b=xyxy(a),xyxy(b)
    A=a.unsqueeze(-2); B=b.unsqueeze(0)
    lo=torch.maximum(A[...,:2],B[...,:2]); hi=torch.minimum(A[...,2:],B[...,2:])
    inter=(hi-lo).clamp_min(0).prod(-1)
    aa=(a[...,2:]-a[...,:2]).clamp_min(0).prod(-1).unsqueeze(-1)
    ab=(b[...,2:]-b[...,:2]).clamp_min(0).prod(-1).unsqueeze(0)
    return inter/(aa+ab-inter).clamp_min(1e-9)
u.meshgrid=meshgrid; u.cxcywh2xyxy=xyxy; u.bboxes_iou=iou; u.visualize_assign=lambda *a,**k:None
sys.modules['yolox.utils']=u

from yolox.models.a41_backbone import prepare_a41_neck, A41Tribrid, A41DetailSPI
from yolox.models.a41_spectral import A41SpectralResidual
from yolox.models.a41_head import A41Head
from yolox.models.a4_detector import A4Detector
from yolox.models.yolo_pafpn import YOLOPAFPN
from yolox.models.network_blocks import BaseConv,CSPLayer


def model(K=3,width=.25,basis='dct'):
    neck=YOLOPAFPN(depth=.33,width=width)
    c=int(64*width)
    stage=torch.nn.Sequential(BaseConv(c,2*c,3,2),CSPLayer(2*c,2*c,n=1,use_gn=False))
    class NativeStage(torch.nn.Module):
        def __init__(self, s): super().__init__();self.backbone_stage=s
        def forward(self,x,run_ttt=False): return self.backbone_stage(x)
    neck.backbone.dark2=NativeStage(stage)
    neck=prepare_a41_neck(neck)
    head=A41Head(K,width=width,basis=basis)
    return A4Detector(neck,head)


def test_trainable_at_step_zero():
    torch.manual_seed(29)
    spi=A41DetailSPI(16,32).train()
    detail=torch.randn(2,16,16,16,requires_grad=True)
    native=torch.randn(2,32,8,8,requires_grad=True)
    corrected=spi(detail,native)
    (corrected.square().mean()).backward()
    spi_grads={n:float(p.grad.abs().sum()) for n,p in spi.named_parameters() if p.grad is not None}
    assert spi_grads['detail_gate_logit']>0
    assert any(g>0 for n,g in spi_grads.items() if 'projector' in n),spi_grads
    assert any(g>0 for n,g in spi_grads.items() if 'router' in n),spi_grads
    spi.enabled=False
    torch.testing.assert_close(spi(detail,native),native,atol=0,rtol=0)
    a=A41SpectralResidual(16,'dct').train()
    features=torch.randn(2,16,8,12)
    (a(features).square().mean()).backward()
    for name in ('band_logits','mix.weight','project.weight','gate.weight'):
        p=dict(a.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    print('PASS SPI/router/projector and spectral band/mix/project/gate gradients at initial step')


def test_full_model_memory_gradient_isolation():
    torch.manual_seed(11)
    m=model().train()
    x=torch.randn(2,3,64,64)
    y=torch.zeros(2,6,5)
    y[0,0]=torch.tensor([0.,24.,24.,20.,20.]); y[1,0]=torch.tensor([2.,40.,40.,16.,16.])
    outputs=m(x,y)
    assert torch.isfinite(outputs['total_loss'])
    assert 0 < float(outputs['mem_loss'].detach()) < 1.0,outputs['mem_loss']
    named=dict(m.named_parameters())
    shared=next(p for n,p in named.items() if 'backbone.backbone.dark3' in n)
    proto=[p for n,p in named.items() if 'memory_banks' in n and 'prototypes' in n]
    query=[p for n,p in named.items() if 'latent_projectors' in n and p.ndim>1]
    spectral=[p for n,p in named.items() if 'spectral_adapters' in n and 'project.weight' in n]
    allparams=[shared]+proto+query+spectral
    g=torch.autograd.grad(outputs['mem_loss'],allparams,retain_graph=True,allow_unused=True)
    assert g[0] is None or not bool(g[0].abs().sum()),'aux backprop into backbone'
    pgrads=g[1:1+len(proto)]
    qgrads=g[1+len(proto):1+len(proto)+len(query)]
    sgrads=g[1+len(proto)+len(query):]
    assert all(gg is None or not bool(gg.abs().sum()) for gg in sgrads),'aux backprop into spectral features'
    assert any(gg is not None and bool(gg.abs().sum()) for gg in pgrads),'prototypes not grounded'
    assert any(gg is not None and bool(gg.abs().sum()) for gg in qgrads),'query projector still starved'
    outputs['total_loss'].backward()
    for needle in ('spi_highway.projector','spi_highway.router',
                   'C3_p3.global_mixer','spectral_adapters'):
        grads=[p.grad for n,p in m.named_parameters() if needle in n and p.grad is not None]
        assert grads and any(g.abs().sum()>0 for g in grads),needle
    assert m.head.last_proto_diagnostic['positive_assignments']>=2
    for level,d in [('C3_p3',2),('C3_p4',1),('C3_n3',1),('C3_n4',1)]:
        block=getattr(m.backbone,level)
        assert block.rmb_dilation==d
        inner=[p for n,p in block.named_parameters() if 'global_mixer.3.qkv.weight' in n]
        assert len(inner)==1 and inner[0].grad is not None and inner[0].grad.abs().sum()>0, level
    print('PASS complete YOLOX gradient flow, weighted class-grounding, no direct auxiliary leakage')


def test_control_and_rectangular_bypass():
    a=model(basis='dct');b=model(basis='orthogonal_control')
    ap={n:tuple(p.shape) for n,p in a.named_parameters()}
    bp={n:tuple(p.shape) for n,p in b.named_parameters()}
    assert ap==bp
    assert not torch.equal(a.head.spectral_adapters[0].dct_kernels,
                           b.head.spectral_adapters[0].dct_kernels)
    a.eval()
    with torch.no_grad():
        p=a(torch.randn(1,3,64,96))
        a.head.memory_enabled=False
        for adapter in a.head.spectral_adapters: adapter.enabled=False
        a.backbone.spi_highway.enabled=False
        for layer in ('C3_p3','C3_p4','C3_n3','C3_n4'):
            getattr(a.backbone,layer).context_enabled=False
        q=a(torch.randn(1,3,96,160))
    assert p.shape[-1]==8 and q.shape[-1]==8
    assert torch.isfinite(q).all()
    print('PASS DCT/orthogonal control param matching, dynamic aspect, and all independent bypasses')


if __name__=='__main__':
    torch.set_num_threads(2)
    test_trainable_at_step_zero()
    test_full_model_memory_gradient_isolation()
    test_control_and_rectangular_bypass()
