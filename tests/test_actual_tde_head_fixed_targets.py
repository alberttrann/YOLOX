"""Actual supplied vNext.2 TDE_Head matching/target parity, CPU.

Pass VNEXT2_DUMP_SOURCE=/path/to/extracted/dump to select the supplied old source.
Intentionally uses the real engram_head.py, yolo_head.py, losses.py and
memory/SimOTA code; only the normal upstream yolox.utils toolbox is stubbed.
This verifies the target math and refusal modes before running user's CURRENT
(not uploaded as a full code snapshot) checkpoint on their machine.
"""
import importlib.util, math, os, sys, types
from pathlib import Path
import torch
import torch.nn as nn
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('VNEXT2_DUMP_SOURCE') or
      (Path(__file__).resolve().parents[1] if
       (Path(__file__).resolve().parents[1]/'yolox/models/engram_head.py').exists()
       else '/mnt/data/vnext2_ttt_integration_workspace/source'))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

yolox=types.ModuleType('yolox');yolox.__path__=[];sys.modules['yolox']=yolox
mods=types.ModuleType('yolox.models');mods.__path__=[];sys.modules['yolox.models']=mods
utils=types.ModuleType('yolox.utils')
def boxiou(gt,pred,xyxy=False):
    if not xyxy:
        gx0=gt[:,:2]-gt[:,2:]/2;gx1=gt[:,:2]+gt[:,2:]/2
        px0=pred[:,:2]-pred[:,2:]/2;px1=pred[:,:2]+pred[:,2:]/2
    else: gx0,gx1,px0,px1=gt[:,:2],gt[:,2:],pred[:,:2],pred[:,2:]
    intersection=(torch.minimum(gx1[:,None,:],px1[None,:,:])-torch.maximum(gx0[:,None,:],px0[None,:,:])).clamp(min=0).prod(-1)
    ag=(gx1-gx0).clamp(min=0).prod(-1);ap=(px1-px0).clamp(min=0).prod(-1)
    return intersection/(ag[:,None]+ap[None,:]-intersection+1e-8)
utils.bboxes_iou=boxiou
utils.cxcywh2xyxy=lambda x:x
utils.meshgrid=lambda seq:torch.meshgrid(*seq,indexing='ij')
utils.visualize_assign=lambda *a,**k:None
sys.modules['yolox.utils']=utils
for name in ['network_blocks','losses','yolo_head','ttt_modules','engram_head']:
    load('yolox.models.'+name,SOURCE/'yolox/models'/f'{name}.py')
Head=sys.modules['yolox.models.engram_head'].TDE_Head
from fixed_detection_targets import NativeMatchReplay,reconstruct_tde_loss,assert_head_loss_parity


def tiny_head(epoch=1):
    torch.manual_seed(50)
    head=Head(num_classes=9,width=.5,in_channels=[256,512,1024])
    head.eval()
    head.training=True  # loss-mode selection; all BN/memory children stay eval
    head.set_epoch_state(epoch,80)
    feat=[torch.randn(1,128,8,8),torch.randn(1,256,4,4),torch.randn(1,512,2,2)]
    images=torch.randn(1,3,64,64)
    labels=torch.zeros(1,5,5)
    labels[0,0]=torch.tensor([2.,30.,30.,22.,22.])
    return head,feat,images,labels


def test_real_head_native_fixed_targets_and_gradient():
    for epoch in [1,20,50]:
        head,feat,images,labels=tiny_head(epoch)
        with NativeMatchReplay(head) as cap:
            with torch.no_grad():native=head(feat,labels,images)
            cap.finalize()
            recon=reconstruct_tde_loss(head,cap)
            errors=assert_head_loss_parity(native,recon,atol=5e-5)
            assert max(errors.values())<5e-5
            assert recon['counts']['foreground']>0
            first_target=cap.stored[0].match
            cap.replay()
            z=torch.tensor(0.,requires_grad=True)
            changed=[f+z*torch.randn_like(f)*0.1 for f in feat]
            adapted=head(changed,labels,images)
            cap.finalize()
            re_a=reconstruct_tde_loss(head,cap)
            err=assert_head_loss_parity(adapted,re_a,atol=5e-5)
            assert all(x==0 for x in re_a['counts'].values() if x=='this-key-never-exists')
            g=torch.autograd.grad(re_a['total'],z,retain_graph=True)[0]
            assert torch.isfinite(g)
            assert torch.equal(cap.stored[0].match[1],first_target[1])
            assert re_a['counts']['small']>0
            assert len(re_a['levels'])==3
            print('PASS: actual TDE head fixed native matches, epoch',epoch,'max parity error',max(err.values()),'task grad',float(g.detach()))

def test_real_head_empty_gt_handling():
    head,feat,images,labels=tiny_head(12)
    labels.zero_()
    with NativeMatchReplay(head) as cap:
        with torch.no_grad():native=head(feat,labels,images)
        cap.finalize()
        recon=reconstruct_tde_loss(head,cap)
        err=assert_head_loss_parity(native,recon,atol=5e-5)
        assert recon['counts']['foreground']==0
        assert recon['counts']['empty_images']==1
        assert float(recon['components']['background_objectness'].detach())>0
        cap.replay()
        again=head(feat,labels,images)
        cap.finalize()
        assert_head_loss_parity(again,reconstruct_tde_loss(head,cap),atol=5e-5)
        print('PASS: actual TDE head empty GT, background loss isolated',err)

def test_replay_rejects_modified_gt():
    head,feat,images,labels=tiny_head(2)
    with NativeMatchReplay(head) as cap:
        with torch.no_grad():head(feat,labels,images)
        cap.finalize();cap.replay()
        forged=labels.clone();forged[0,0,1]+=2
        try:head(feat,forged,images)
        except RuntimeError as e:
            assert 'changed' in str(e).lower()
        else:raise AssertionError('Replayer silently accepted modified GT')
    print('PASS: GT change rejected')

if __name__=='__main__':
    torch.set_num_threads(1)
    for k,fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):fn()
