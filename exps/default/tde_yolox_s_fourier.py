"""Full-fidelity tdevNext.3 Fourier experiment; no legacy source patching.

Model construction uses the installed vNext.2 CSP+SPI+Tribrid components but
replaces the entire head with a fully defined implementation in fourier_head.
The GN/Bn stage mismatch is explicitly rejected. A separate experiment name
prevents overwriting the successful vNext.2 results.
"""
import json
import math
import random
from collections import Counter
import torch
from torch import nn
import torch.distributed as dist
from tde_yolox_s import Exp as SourceExp
from yolox.models.fourier_backbone import prepare_backbone
from yolox.models.fourier_head import FourierTDEHead
from yolox.models.fourier_detector import FourierDetector


class Exp(SourceExp):
    def __init__(self):
        super().__init__()
        self.exp_name='tdevnext3_fourier_spectral'
        self.fourier_variant='spectral'
        self.seed=0
        self.pixel_budget=None
        self.auto_geometry=True
        # Fully disable weather-specific training changes and legacy TTT rates.
        self.enable_mixup=False
        self.mixup_prob=0.0
        # Keep Mosaic optional but identical across all variant experiments.
        self._preflight_result=None
        self._capacity=None

    def _preflight(self):
        if self._preflight_result is not None:
            return self._preflight_result
        with open(self.train_ann,encoding='utf-8') as handle:
            coco=json.load(handle)
        images=coco.get('images',[])
        categories=coco.get('categories',[])
        if not images or not categories:
            raise ValueError('Expected nonempty images and categories in COCO JSON')
        ids=[cat['id'] for cat in categories]
        if len(set(ids))!=len(ids):
            raise ValueError('COCO category IDs must be unique')
        ratios=sorted(float(im['width'])/float(im['height']) for im in images
                      if im.get('height',0)>0 and im.get('width',0)>0)
        if len(ratios)!=len(images):
            raise ValueError('One or more images have invalid geometry')
        counts=Counter(int(ann['image_id']) for ann in coco.get('annotations',[]))
        max_labels=max(counts.values(),default=0)
        # Conservative source upper bound; actual ragged collate is unavailable
        # without replacing the installed YOLOX upstream dataloader.
        n_mosaic=4 if self.mosaic_prob>0 else 1
        n_mix=1 if self.enable_mixup and self.mixup_prob>0 else 0
        self._capacity=max(1,max_labels*(n_mosaic+n_mix))
        m=len(ratios)//2
        ratio=ratios[m] if len(ratios)%2 else (ratios[m-1]+ratios[m])/2
        budget=self.pixel_budget if self.pixel_budget is not None else self.input_size[0]*self.input_size[1]
        if self.auto_geometry:
            stride=32
            h=max(stride,int(round(math.sqrt(budget/ratio)/stride))*stride)
            w=max(stride,int(round(math.sqrt(budget*ratio)/stride))*stride)
            self.input_size=(h,w)
            self.test_size=(h,w)
        self.num_classes=len(categories)
        self._preflight_result={'num_images':len(images),'classes':self.num_classes,
                                'max_source_boxes':max_labels,'lossless_label_capacity':self._capacity,
                                'aspect_min':ratios[0],'aspect_median':ratio,
                                'aspect_max':ratios[-1],'canvas':list(self.input_size)}
        return self._preflight_result

    def get_model(self):
        if getattr(self,'model',None) is not None:
            return self.model
        self._preflight()
        if self.fourier_variant not in FourierTDEHead.SUPPORTED:
            raise ValueError(f'Unknown fourier_variant={self.fourier_variant}')
        # Construct vNext2's actual pretrained-compatible backbone + full neck.
        # SourceExp.get_model builds old head but it is NEVER used for forward.
        source_model=super().get_model()
        neck=prepare_backbone(source_model.backbone,remove_spi_clipping=True)
        head=FourierTDEHead(self.num_classes, width=self.width,
                            in_channels=(256,512,1024),act=self.act,
                            variant=self.fourier_variant,loss_type='oiou')
        # Match the BN configuration of the existing pretrained-friendly YOLOX
        # experiment instead of accidentally using PyTorch default eps/momentum.
        for layer in head.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.eps=1e-3
                layer.momentum=0.03
        self.model=FourierDetector(neck,head)
        self.model.max_epochs=int(self.max_epoch)
        return self.model

    def get_dataset(self,cache=False,cache_type='ram'):
        from yolox.data import COCODataset,TrainTransform
        self._preflight()
        ds=COCODataset(data_dir=self.data_dir,json_file=self.train_ann,name='',
                       img_size=self.input_size,
                       preproc=TrainTransform(max_labels=self._capacity,
                           flip_prob=self.flip_prob,hsv_prob=self.hsv_prob),
                       cache=cache,cache_type=cache_type)
        ds.cache=cache;ds.cache_type=cache_type
        return ds

    def get_eval_dataset(self,**kwargs):
        self._preflight()
        return super().get_eval_dataset(**kwargs)

    def random_resize(self,data_loader,epoch,rank,is_distributed):
        self._preflight()
        # Same aspect and compute-bounded multi-scale policy for all domains.
        # Not an implementation of arbitrary heterogeneous batch aspect ratios.
        h0,w0=self.input_size
        if rank==0:
            if self.multiscale_range>0:
                d=random.randint(-int(self.multiscale_range),int(self.multiscale_range))
                h=max(32,h0+32*d)
                w=max(32,round(h*w0/h0/32)*32)
            else:
                h,w=h0,w0
            value=torch.tensor((h,w),dtype=torch.long,device='cuda')
        else:
            value=torch.zeros(2,dtype=torch.long,device='cuda')
        if is_distributed:dist.broadcast(value,src=0)
        return tuple(int(v) for v in value.tolist())
