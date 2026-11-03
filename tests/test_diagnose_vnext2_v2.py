"""Standalone CPU regression checks for vNext.2 diagnostic-v2 helpers.

Run: python tests/test_diagnose_vnext2_v2.py
Does not import YOLOX or require GPU/checkpoints.
"""
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

file=Path(__file__).resolve().parents[1]/'tools/diagnose_vnext2_v2.py'
spec=importlib.util.spec_from_file_location('vnext2_diag_v2',file)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_tensor_math():
    a = torch.tensor([1., 2., 3.])
    d=m.tensor_delta(a,a)
    assert d['relative_l2']==0 and d['max_abs']==0
    d=m.tensor_delta(a,a+1)
    assert abs(d['mean_abs']-1)<1e-6 and d['rms_abs']>0
    assert m.tensor_summary(a)['l2']>0
    try:
        m.tensor_delta(a,torch.ones(5))
        assert False
    except ValueError:
        pass



def test_matched_batch_controls_really_match_same_image():
    def report(a,b):
        return {'predictions':torch.tensor([[[a]],[[b]]], dtype=torch.float32),
                'features':torch.tensor([[[[a]]],[[[b]]]],dtype=torch.float32)}
    aa, ab, ba, bb = report(1,1), report(1,2), report(2,1), report(2,2)
    r=m.matched_batch_controls(aa,ab,ba,bb)
    assert all(v['max_abs']==0. for v in r.values()),r
    # Only B in AB is modified. It must be detected by B companion control
    # and must not appear as an A companion effect.
    ab['predictions'][1]+=0.25
    ab['features'][1]+=0.25
    r=m.matched_batch_controls(aa,ab,ba,bb)
    assert r['B_second_slot_companion_effect']['max_abs']==0.25
    assert r['B_second_slot_dark2_effect']['max_abs']==0.25
    assert r['A_first_slot_companion_effect']['max_abs']==0.
    assert r['A_first_slot_dark2_effect']['max_abs']==0.


def test_dataset_support():
    with tempfile.TemporaryDirectory() as t:
        path=Path(t)/'test.json'
        path.write_text(json.dumps({'images':[{'id':1},{'id':2}],
            'categories':[{'id':1,'name':'car'}, {'id':2,'name':'sign'}],
            'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[1,2,3,4]},
                           {'id':2,'image_id':1,'category_id':1,'bbox':[5,6,7,8]}]}))
        rep=m.annotation_report(path)
        assert rep['images']==2 and rep['images_without_annotations']==1
        assert rep['class_support'][0]['num_annotations']==2
        assert rep['class_support'][1]['num_annotations']==0


class MockProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token=nn.Parameter(torch.zeros(1,8,1,1))
        self.conv=nn.Conv2d(8,8,1,bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(8).view(8,8,1,1)*0.15)
    def forward(self,x):
        return x+self.conv(x)


class MockTTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_adapter=True
        self.residual_scale=nn.Parameter(torch.randn(1,8,1,1)*0.01)
        self.residual_bias=nn.Parameter(torch.randn(1,8,1,1)*0.01)
        self.projector=MockProjector()
        self.ttt_lrs=nn.ParameterDict({'residual_scale':nn.Parameter(torch.tensor(-1.)),
                                     'residual_bias':nn.Parameter(torch.tensor(-1.))})
        self.residual_gain=0.1
        self.residual_grad_clip=5.
        self.mask_ratio=.2
        self.current_epoch=2
        self.max_epochs=80
        self.min_meta_lr=1e-4
        self.base_max_lr=.08
    def _get_noise_clipped_target(self,feat):
        smooth=F.avg_pool2d(feat,3,1,1)
        return smooth+(feat-smooth).clamp(-.15,.15)
    def _get_relative_contrast_mask(self,feat,ratio):
        mask=torch.zeros_like(feat[:,0:1])
        mask[:,:,:2,:2]=1
        return mask


def test_static_fastweight():
    torch.manual_seed(5)
    t=MockTTT().eval().requires_grad_(False)
    native=torch.randn(1,8,7,9)
    before={n:p.clone() for n,p in t.state_dict().items()}
    static=m.make_static_adapter(t)(native,run_ttt=True)
    assert not static.requires_grad
    assert m.tensor_delta(static,native)['relative_l2']>0
    probe=m.inner_step_probe(t,native)
    assert math.isfinite(probe['inner_loss_before'])
    assert math.isfinite(probe['inner_loss_after'])
    assert probe['gradient_total_l2']>0
    assert probe['fast_scale_step_l2']>0 or probe['fast_bias_step_l2']>0
    assert probe['gradient_clip_factor']<=1
    rep=m.tensor_delta(probe.pop('_reconstructed_ttt_feature'),static)
    assert rep['relative_l2']>=0
    for n,p in t.state_dict().items():
        assert torch.equal(p,before[n]),f'Unexpected state mutation: {n}'


def test_gt_and_spatial():
    class C:
        def loadImgs(self, ids):
            assert ids==[42]
            return [{'height':100,'width':200}]
        imgToAnns={42:[{'id':3,'category_id':1,'bbox':[10,20,20,30]},
                         {'id':4,'iscrowd':1,'category_id':1,'bbox':[2,3,4,5]}]}
    class D:
        coco=C()
        ids=[42]
    gt=m.gt_boxes_for_index(D(),0,[1,3,64,128])
    assert gt['supported'] and len(gt['boxes'])==1
    assert abs(gt['boxes'][0]['bbox_input_xyxy'][0]-6.4)<1e-4
    gt['canvas_hw']=[64,128]
    before=torch.zeros(1,8,16,32)
    after=before.clone()
    after[:,:,:,3:8]=1
    rep=m.spatial_effects(before,after,gt)
    assert rep['supported'] and rep['reported_gt_count']==1
    assert rep['foreground_union_fraction']>0
    assert rep['gt_regions'][0]['effect']['max_abs']==1


def test_matching_and_batch_control():
    a={'detections':[[0,0,10,10,0.9,0.9,2],[20,20,30,30,0.8,0.8,1]]}
    b={'detections':[[1,1,11,11,0.9,0.9,2]]}
    rep=m.match_detections(a,b)
    assert rep['paired']==1 and rep['a_unpaired']==1 and rep['b_unpaired']==0
    assert rep['mean_pair_iou']>.5
    # If image A's forward depends on the *other* batch member, the matched-shape
    # [A,A] versus [A,B] control detects it without batch-size confounding.
    A=torch.ones(1,3,5,5)
    B=torch.zeros_like(A)
    def coupled(batch):
        return batch+batch.mean(dim=0,keepdim=True)
    def independent(batch):
        return batch*2
    aa=torch.cat([A,A],0)
    ab=torch.cat([A,B],0)
    assert m.tensor_delta(coupled(ab)[:1],coupled(aa)[:1])['relative_l2']>0
    assert m.tensor_delta(independent(ab)[:1],independent(aa)[:1])['relative_l2']==0


def test_end_to_end_fake_checkpoint():
    """Mock full checkpoint/data/eval interfaces, exercise the *whole* CLI on CPU.

    This does not replace CUDA/in-repository verification, but catches import,
    patch restoration, hook, JSON, and orchestration defects.
    """
    import sys
    import types
    from unittest.mock import patch
    class FakeTTT(MockTTT):
        def __init__(self):
            super().__init__()
            self.backbone_stage=nn.Conv2d(3,8,3,padding=1)
        def forward_from_features(self,native,run_ttt=True):
            if not run_ttt:
                return native
            return torch.cat([m.inner_step_probe(self,native[j:j+1])['_reconstructed_ttt_feature']
                              for j in range(native.size(0))],0)
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.dark2=FakeTTT()
            self.dark3=nn.Identity()
    class FakeWrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=FakeBackbone()
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=FakeWrapped()
            self.eval_ttt_enabled=False
        def set_meta_training_state(self,epoch,max_epochs):
            t=self.backbone.backbone.dark2
            t.current_epoch=epoch
            t.max_epochs=max_epochs
        def forward(self,x):
            dark=self.backbone.backbone
            native=dark.dark2.backbone_stage(x)
            adapted=dark.dark2.forward_from_features(native,run_ttt=self.eval_ttt_enabled)
            z=dark.dark3(adapted)
            pooled=z.mean(dim=(1,2,3))
            out=pooled[:,None,None].expand(-1,1,7).clone()
            out[:,:,0:4]=10.
            out[:,:,4:]=0.8
            return out
    class FakeCoco:
        def loadImgs(self,ids):
            return [{'height':16,'width':16}]
        imgToAnns={1:[{'id':10,'category_id':1,'bbox':[3,3,6,7]}],
                   2:[{'id':11,'category_id':1,'bbox':[2,2,3,5]}]}
    class FakeData:
        ids=[1,2]
        coco=FakeCoco()
        def __len__(self):
            return 2
        def __getitem__(self,i):
            return (torch.ones(3,16,16)*(i+1),None,None,self.ids[i])
    class FakeExp:
        def __init__(self,annotation):
            self.data_dir=''
            self.val_ann=str(annotation)
            self.test_ann=str(annotation)
            self.test_size=(16,16)
            self.max_epoch=80
            self.num_classes=2
            self.nmsthre=.6
        def get_model(self):
            return FakeModel()
        def get_eval_dataset(self,**kwargs):
            return FakeData()
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        ann=td/'ann.json'
        ann.write_text(json.dumps({'images':[{'id':1},{'id':2}],
            'categories':[{'id':1,'name':'fake'}],
            'annotations':[{'id':10,'image_id':1,'category_id':1,'bbox':[3,3,6,7]},
                           {'id':11,'image_id':2,'category_id':1,'bbox':[2,2,3,5]}]}))
        model=FakeModel()
        ckpt=td/'weights.pth'
        torch.save({'ema_model':model.state_dict(),'start_epoch':2},ckpt)
        dest=td/'report.json'
        exp_stub=types.ModuleType('yolox.exp')
        exp_stub.get_exp=lambda *_: FakeExp(ann)
        utils_stub=types.ModuleType('yolox.utils')
        utils_stub.postprocess=lambda p,n,c,t: [torch.tensor([[1.,2.,8.,9.,.8,.8,1.]]) for _ in range(len(p))]
        pkg_stub=types.ModuleType('yolox')
        pkg_stub.__path__=[]
        argv=['diagnose_vnext2_v2.py','--checkpoint',str(ckpt),'--model-exp','mock-model',
              '--data-exp','mock-data','--output',str(dest)]
        with patch.dict(sys.modules,{'yolox':pkg_stub,'yolox.exp':exp_stub,'yolox.utils':utils_stub}), \
             patch.object(sys,'argv',argv):
            m.main()
        report=json.loads(dest.read_text())
        assert report['registered_state_mutations']==[]
        assert report['post_nms_counts']=={'native':1,'static':1,'adapted':1}
        assert report['inner_step_replica_vs_actual_ttt_feature']['relative_l2']<1e-6
        assert report['three_way_feature_effect']['native_to_static']['relative_l2']>=0
        assert set(report['batch_controls_same_shape'])=={'native','static','adapted'}



if __name__=='__main__':
    for t in [test_tensor_math,test_dataset_support,test_static_fastweight,
              test_gt_and_spatial,test_matching_and_batch_control,test_end_to_end_fake_checkpoint]:
        t()
        print('PASS:',t.__name__)
