import torch
import pytest


def test_synthetic_native_and_semantic_integration(src):
    torch.manual_seed(19); torch.set_num_threads(2)
    neck=src['yolo_pafpn'].YOLOPAFPN(.33,.5,
            pretrained_compatible_dark2=True, ttt_residual_adapter=False)
    head=src['pgsma_head'].PGSMAHead(3,.5,loss_type='iou',assignment='scpa')
    model=src['pgsma_model'].PGSMAYOLOX(neck,head,3,.5,semantic_dim=16,
                                         semantic_modes=2)
    assert not any('mask_token' in n or 'memory_banks' in n or 'residual_scale' in n
                   for n,_ in model.named_parameters())
    # Native-only forward receives no gradients from semantic, by construction.
    model.train()
    x=torch.randn(2,3,64,96)
    y=torch.zeros(2,8,5)
    y[0,0]=torch.tensor([0,25.,25.,15.,12.])
    y[1,0]=torch.tensor([1,48.,36.,10.,8.])
    out=model(x,y)
    assert torch.isfinite(out['total_loss'])
    assert out['num_gt_total']==2
    assert out['uncovered_gt']==0
    assert out['core_positives']==2
    sem=model.semantic_losses(seed=17)
    assert all(torch.isfinite(v) for v in sem.values())
    sem_sum=sum(sem[k] for k in ['prototype','teacher','candidate','gate'])
    sem_sum.backward()
    assert all(p.grad is None for p in model.native_parameters())
    assert any(p.grad is not None for p in model.semantic_parameters())
    model.zero_grad(set_to_none=True)
    out['total_loss'].backward()
    assert all(p.grad is None for p in model.semantic_parameters())
    assert any(p.grad is not None for p in model.native_parameters())
    model.eval()
    with torch.no_grad():
        model.set_inference_mode('native')
        clean=model(torch.randn(1,3,64,96))
        model.set_inference_mode('pgsma')
        pg=model(torch.randn(1,3,96,64))
        model.set_inference_mode('memory')
        mb=model(torch.randn(1,3,64,96))
    assert clean.ndim==3 and clean.shape[-1]==8
    assert pg.ndim==3 and pg.shape[-1]==8
    assert mb.ndim==3 and mb.shape[-1]==8
    assert torch.isfinite(pg).all() and torch.isfinite(mb).all()


def test_spi_zero_gate(src):
    torch.manual_seed(11)
    spi=src['pgsma_spi'].SPIResidual(8,16)
    a=torch.randn(2,8,8,8,requires_grad=True)
    p=torch.randn(2,16,4,4,requires_grad=True)
    v=spi(a,p)
    assert torch.equal(v,p)
    v.square().mean().backward()
    assert spi.gate.grad is not None and spi.gate.grad.abs()>0


def test_dynamic_context_neck_small_map(src):
    attention=src['ttt_modules'].ContinuousSaliencyAttention(32,num_heads=2,saliency_floor=0)
    for hw in ((1,1),(2,2),(9,11)):
        x=torch.randn(1,32,*hw)
        y=attention(x)
        assert y.shape==x.shape and torch.isfinite(y).all()


def test_full_native_optimizer_step_isolation(src):
    import copy
    import importlib.util
    from pathlib import Path
    torch.manual_seed(23);torch.set_num_threads(2)
    model=src['pgsma_model'].PGSMAYOLOX(
        src['yolo_pafpn'].YOLOPAFPN(.33,.5,pretrained_compatible_dark2=True,
                                  ttt_residual_adapter=False),
        src['pgsma_head'].PGSMAHead(2,.5,assignment='scpa'),
        2,.5,semantic_dim=8,semantic_modes=2)
    twin=copy.deepcopy(model)
    x=torch.randn(2,3,64,64)
    y=torch.zeros(2,6,5)
    y[0,0]=torch.tensor([0.,28,28,14,10]);y[1,0]=torch.tensor([1.,28,28,12,14])
    opt1=torch.optim.SGD(model.native_parameters(),lr=.002,momentum=.9)
    opt2=torch.optim.SGD(twin.native_parameters(),lr=.002,momentum=.9)
    native_out=model.train()(x,y)
    twin_out=twin.train()(x,y)
    assert torch.equal(native_out['total_loss'],twin_out['total_loss'])
    spec=importlib.util.spec_from_file_location('pgstep',
       Path(__file__).resolve().parents[1]/'overlay/yolox/core/pgsma_step.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    semopt=torch.optim.AdamW(model.semantic_parameters(),lr=.001)
    ok,_=mod.semantic_optimizer_step(model,semopt,seed=912)
    assert ok
    opt1.zero_grad();opt2.zero_grad()
    native_out['total_loss'].backward();twin_out['total_loss'].backward()
    for a,b in zip(model.native_parameters(),twin.native_parameters()):
        assert torch.equal(a.grad,b.grad)
    opt1.step();opt2.step()
    for a,b in zip(model.native_parameters(),twin.native_parameters()):
        assert torch.equal(a,b)
    native_state=model.state_dict();twin_state=twin.state_dict()
    for key in twin_state:
        if key.startswith('head.pgsma_engine.'):
            continue
        assert torch.equal(native_state[key], twin_state[key]),key
