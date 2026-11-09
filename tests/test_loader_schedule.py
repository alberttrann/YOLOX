"""Data-loader interface regression using a simulated YOLOX DataLoader API.

This is NOT a substitute for --sample-mosaic against installed YOLOX.
"""
from pathlib import Path
import importlib.util
import sys
import types
import torch

ROOT=Path(__file__).resolve().parents[1]

def mod(name, package=False):
    result=types.ModuleType(name)
    if package:result.__path__=[]
    sys.modules[name]=result
    return result

source=mod('tde_yolox_s')
source.Exp=object
source._yolo_worker_init_fn=lambda worker_id:None
class A4Exp:
    def __init__(self):
        self.data_num_workers=0
        self.input_size=(576,1024)
        self.data_dir='test'
        self.train_ann='dummy.json'
        self._capacity=364
        self.flip_prob=.5
        self.hsv_prob=1.
        self.mosaic_prob=.5
        self.mixup_prob=0.
        self.enable_mixup=False
        self.degrees=0.;self.translate=.1
        self.mosaic_scale=(.5,1.5);self.mixup_scale=(.5,1.5);self.shear=0.
        self.max_epoch=80;self.no_aug_epochs=15;self.multiscale_range=1
    def _preflight(self):return {'capacity':self._capacity}
mod('tde_yolox_s_fourier_a4').Exp=A4Exp
mod('yolox',package=True)
mod('yolox.models',package=True)
for name,attr in [('a41_backbone','prepare_a41_neck'),('a41_head','A41Head'),('a4_detector','A4Detector')]:
    setattr(mod('yolox.models.'+name),attr,object())

class COCO:
    def __init__(self,**kwargs):self.preproc=kwargs['preproc'];self.kwargs=kwargs
    def __len__(self):return 10
class TrainTransform:
    def __init__(self,**kw):self.kw=kw
class MosaicDetection:
    def __init__(self,source,**kwargs):
        self._dataset=source;self.preproc=kwargs['preproc'];self.kwargs=kwargs
        self.mosaic_prob=kwargs['mosaic_prob']
    def __len__(self):return len(self._dataset)
class BatchSampler:
    def __init__(self,**kwargs):self.mosaic=kwargs['mosaic'];self.kwargs=kwargs
class DataLoader:
    def __init__(self,dataset,**kwargs):self.dataset=dataset;self.batch_sampler=kwargs['batch_sampler'];self.kwargs=kwargs
    def close_mosaic(self):self.batch_sampler.mosaic=False
fake=mod('yolox.data')
fake.COCODataset=COCO;fake.TrainTransform=TrainTransform
fake.MosaicDetection=MosaicDetection;fake.YoloBatchSampler=BatchSampler
fake.DataLoader=DataLoader

s=importlib.util.spec_from_file_location('test_a41_exp',ROOT/'overlay/exps/default/tde_yolox_s_fourier_a41.py')
e=importlib.util.module_from_spec(s);s.loader.exec_module(e)

def test_no_double_preproc():
    ex=e.Exp()
    loader=ex.get_data_loader(batch_size=8,is_distributed=False,no_aug=False)
    assert isinstance(loader.dataset,MosaicDetection)
    assert loader.dataset._dataset.preproc is None
    assert isinstance(loader.dataset.preproc,TrainTransform)
    assert loader.dataset.preproc.kw['max_labels']==364
    assert loader.dataset.kwargs['mosaic_prob']==.5
    assert loader.dataset.kwargs['enable_mixup'] is False
    assert loader.batch_sampler.mosaic
    loader.close_mosaic()
    assert not loader.batch_sampler.mosaic
    print('PASS exactly one TrainTransform, actual wrapper and end-of-training close_mosaic API')

def test_resize_close():
    pair=e.a41_resize_choice((576,1024),1,epoch=10,max_epoch=80,no_aug_epochs=15,randint=lambda a,b:1)
    assert pair==(608,1088),pair
    for epoch in (65,66,79):
        assert e.a41_resize_choice((576,1024),1,epoch,80,15,randint=lambda a,b:1)==(576,1024)
    assert e.a41_resize_choice((576,1024),0,0,80,15)==(576,1024)
    print('PASS stride32, dynamic aspect and frozen terminal resize, epoch 66–80')

if __name__=='__main__':
    test_no_double_preproc();test_resize_close()
