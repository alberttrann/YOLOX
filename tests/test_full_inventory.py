"""Standalone parameter inventory smoke test without external YOLOX dependencies.
The actual user's repo should use tools/audit_architecture.py for final evidence.
"""
import importlib.util, sys, types
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
for n in ('yolox','yolox.models'):
    pkg=types.ModuleType(n);pkg.__path__=[];sys.modules[n]=pkg
utils=types.ModuleType('yolox.utils')
for name in ('bboxes_iou','cxcywh2xyxy','visualize_assign'):
    setattr(utils,name,lambda *a,**k: None)
utils.meshgrid=lambda axes: torch.meshgrid(*axes,indexing='ij')
def bboxes_iou(a,b,xyxy=False):
    a=a.float();b=b.float()
    if not xyxy:
        a=torch.cat([a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2],dim=-1)
        b=torch.cat([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],dim=-1)
    x1=torch.maximum(a[:,None,:2],b[None,:,:2]);x2=torch.minimum(a[:,None,2:],b[None,:,2:])
    inter=(x2-x1).clamp(min=0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp(min=0).prod(-1)
    area_b=(b[:,2:]-b[:,:2]).clamp(min=0).prod(-1)
    return inter/(area_a[:,None]+area_b[None,:]-inter+1e-8)
utils.bboxes_iou=bboxes_iou
sys.modules['yolox.utils']=utils
for name in ('network_blocks','losses','ttt_modules','tribrid_neck','darknet','yolo_head','engram_head','yolo_pafpn','yolox'):
    path=root/'yolox'/'models'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(f'yolox.models.{name}',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
mods=lambda name:sys.modules[f'yolox.models.{name}']
neck=mods('yolo_pafpn').YOLOPAFPN(.33,.5,pretrained_compatible_dark2=True,ttt_residual_adapter=True)
head=mods('engram_head').TDE_Head(9,.5,loss_type='oiou')
model=mods('yolox').YOLOX(neck,head)
model.train_ttt_enabled=model.eval_ttt_enabled=True
neck.enable_spi=True
head.enable_memory_restoration=True
head.use_baseline_assignments=False
for block in model.modules():
    if isinstance(block,mods('tribrid_neck').C2f_Tribrid):block.enable_global_context=True
    if isinstance(block,mods('network_blocks').C2fA):block.enable_attention=True
pairs=list(model.named_parameters());total=sum(p.numel() for _,p in pairs)
print('FULL MODEL PARAMETER COUNT:',format(total,','))
for prefix in ('backbone.backbone','backbone.spi_highway','backbone.C3_p4','backbone.C3_p3','backbone.C3_n3','backbone.C3_n4','head.memory_banks','head.latent_projectors','head.cls_convs','affordance_head'):
    count=sum(p.numel() for n,p in pairs if n.startswith(prefix))
    print(prefix,format(count,','))
    assert count>0,prefix
assert total>9_000_000 # Expected active modules registered (not vanilla YOLOX)
model.eval()
with torch.no_grad():
    result=model(torch.randn(1,3,64,64))
assert result.isfinite().all()
print('FULL EVAL FORWARD PASS:',tuple(result.shape),'finite OK')

model.train()
model.set_meta_training_state(1, 80)
model.aux_ramp_epochs=12
model.mem_loss_weight=0.10
model.proto_loss_weight=0.10
model.intra_loss_weight=0.05
model.rec_loss_weight=0.05
model.afford_loss_weight=0.25
labels=torch.zeros(2,120,5)
labels[0,0]=torch.tensor([0,28,25,20,12],dtype=torch.float32)
labels[1,0]=torch.tensor([3,38,35,11,16],dtype=torch.float32)
pred=model(torch.randn(2,3,64,64), labels)
print('TRAIN LOSSES:', {k:round(v.item(),4) for k,v in pred.items() if torch.is_tensor(v) and v.numel()==1})
assert torch.isfinite(pred['total_loss']).all()
pred['total_loss'].backward()
for prefix in ('backbone.spi_highway','backbone.C3_p3.gate','backbone.backbone.dark2.residual_scale',
               'head.latent_projectors','head.memory_banks','affordance_head'):
    assigned=[v for n,v in model.named_parameters() if n.startswith(prefix) and v.grad is not None]
    assert assigned, f'no gradient for {prefix}'
    assert all(torch.isfinite(g.grad).all() for g in assigned)
    norm=sum(g.grad.abs().sum().item() for g in assigned)
    print('GRADIENT GROUP',prefix,round(norm,8))
    assert norm>0,f'zero group gradient {prefix}'
print('PASS: full all-module CPU synthetic train backward')
all_grads=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
invalid=[n for n,p in all_grads if not torch.isfinite(p.grad).all().item()]
missing=[n for n,p in model.named_parameters() if p.grad is None]
zero=[n for n,p in all_grads if p.grad.abs().sum().item() == 0]
print('GRADIENT COVERAGE',len(all_grads),'/',len(pairs),'missing',len(missing),'zero',len(zero),'invalid',len(invalid))
print('MISSING EXAMPLES',missing[:22])
print('ZERO EXAMPLES',zero[:22])
assert not invalid
optim=torch.optim.SGD(model.parameters(), lr=1e-3)
with torch.no_grad():
    gate_names=('backbone.C3_p3.gate','backbone.C3_p3.dense_gate',
                'backbone.backbone.dark4.1.ema.gamma',
                'backbone.backbone.dark4.1.dense_gate')
    for name,p in model.named_parameters():
        if name in gate_names:
            print('INITIAL GATE GRAD',name,float(p.grad))
optim.step();optim.zero_grad(set_to_none=True)
pred2=model(torch.randn(2,3,64,64), labels)
assert torch.isfinite(pred2['total_loss']).all()
pred2['total_loss'].backward()
for prefix in ('backbone.C3_p3.global_mixer.3.qkv.weight',
               'backbone.C3_p3.dense_local.cv1.conv.weight',
               'backbone.backbone.dark4.1.ema.conv3x3.weight',
               'backbone.backbone.dark4.1.dense_local.cv1.conv.weight',
               'backbone.backbone.dark2.projector.net.0.weight'):
    vals=[p.grad for n,p in model.named_parameters() if n == prefix]
    assert vals and vals[0] is not None
    print('SECOND STEP GRAD',prefix,vals[0].abs().sum().item())
    assert vals[0].abs().sum().item()>0,prefix
print('PASS: previously zero-gated branch weights activate after first optimizer step')
# Validate real Exp.get_model() configuration, not only manually constructed modules.
exp_stub=types.ModuleType('yolox.exp')
class _BaseExp:
    def __init__(self):
        self.model=None
exp_stub.Exp=_BaseExp
sys.modules['yolox.exp']=exp_stub
sys.modules['yolox.models'].YOLOX=mods('yolox').YOLOX
sys.modules['yolox.models'].YOLOPAFPN=mods('yolo_pafpn').YOLOPAFPN
spec=importlib.util.spec_from_file_location('test_user_exp',root/'exps/default/tde_yolox_s.py')
exp_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp_module)
real_exp=exp_module.Exp()
actual=real_exp.get_model()
assert real_exp.enable_ttt and real_exp.enable_memory and real_exp.enable_spi
assert real_exp.enable_global_context and real_exp.enable_dense_local
assert actual.train_ttt_enabled and actual.eval_ttt_enabled
assert actual.head.enable_memory_restoration
assert actual.backbone.backbone.dark2.residual_adapter
assert sum(p.numel() for p in actual.parameters()) == total
print('PASS: real experiment bootstrap and all-full-mode flags')
