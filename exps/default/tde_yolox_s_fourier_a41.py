"""tdevNext.3 A4.1: *actual* Mosaic, trainable detail/context, grounded memory.

A4.1 is a separate train-from-COCO-pretrained research treatment. Do not
resume an A4 or A4-R optimizer or compare it as architecture-only ablation:
the augmentation distribution is deliberately corrected as well.
"""
import random
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import RandomSampler, DistributedSampler
from tde_yolox_s import Exp as SourceExp, _yolo_worker_init_fn
from tde_yolox_s_fourier_a4 import Exp as A4Exp
from yolox.models.a41_backbone import prepare_a41_neck
from yolox.models.a41_head import A41Head
from yolox.models.a4_detector import A4Detector


def a41_resize_choice(input_size, multiscale_range, epoch, max_epoch,
                      no_aug_epochs, randint=random.randint):
    """Pure, testable choice; `epoch` is ZERO-based as in native Trainer."""
    h0, w0 = [int(v) for v in input_size]
    if h0 % 32 or w0 % 32:
        raise ValueError('Canvas must be stride-32 aligned')
    if epoch >= int(max_epoch) - int(no_aug_epochs) or multiscale_range <= 0:
        return h0, w0
    d = randint(-int(multiscale_range), int(multiscale_range))
    height = max(32, h0 + 32 * d)
    width = max(32, round((height * w0 / h0) / 32) * 32)
    return height, width


class Exp(A4Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = 'tdevnext3_a41_dct_grounded'
        self.fourier_variant = 'a41'
        self.a4_filter_basis = 'dct'
        self.a41_architecture = 'real_mosaic_live_spi_context_spectral_query_grounding'
        self.a41_proto_weight = 0.05
        self.a41_data_policy = 'MosaicDetection_single_TrainTransform'
        self.enable_ttt = False
        self.enable_legacy_ttt = False
        self.enable_rec_aux = False
        self.enable_proto_aux = False
        self.enable_intra_aux = False
        self.enable_afford_aux = False
        self.enable_memory_aux = True
        self.enable_memory = True
        self.enable_spi = True
        self.enable_mixup = False
        self.mixup_prob = 0.0

    def get_model(self):
        if getattr(self, 'model', None) is not None:
            return self.model
        self._preflight()
        source_model = SourceExp.get_model(self)
        neck = prepare_a41_neck(source_model.backbone)
        head = A41Head(self.num_classes, width=self.width,
                       in_channels=(256, 512, 1024), act=self.act,
                       loss_type='oiou', basis=self.a4_filter_basis,
                       proto_weight=self.a41_proto_weight)
        for module in head.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
        self.model = A4Detector(neck, head)
        self.model.max_epochs = int(self.max_epoch)
        return self.model

    def get_dataset(self, cache=False, cache_type='ram'):
        """Raw COCO images/boxes; MosaicDetection owns the ONLY TrainTransform.

        Its pull_item() fetches raw boxes. Giving this dataset a TrainTransform
        as well is unnecessary and risks silent double preprocessing if the
        installed implementation differs from standard YOLOX.
        """
        from yolox.data import COCODataset
        self._preflight()
        ds = COCODataset(data_dir=self.data_dir, json_file=self.train_ann,
                         name='', img_size=self.input_size, preproc=None,
                         cache=cache, cache_type=cache_type)
        ds.cache, ds.cache_type = cache, cache_type
        return ds

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (MosaicDetection, TrainTransform,
                                YoloBatchSampler, DataLoader as YoloDataLoader)
        self._preflight()
        raw = self.get_dataset(cache=cache_img, cache_type='ram')
        if getattr(raw, 'preproc', 'missing') is not None:
            raise RuntimeError('A4.1 COCO source must have preproc=None')
        if self.enable_mixup or float(self.mixup_prob):
            raise RuntimeError('A4.1 mixup not certified; keep it disabled')
        data = MosaicDetection(
            raw, img_size=self.input_size, mosaic=not no_aug,
            preproc=TrainTransform(max_labels=self._capacity,
                                   flip_prob=self.flip_prob,
                                   hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate,
            mosaic_scale=self.mosaic_scale, mixup_scale=self.mixup_scale,
            shear=self.shear, enable_mixup=False,
            mosaic_prob=self.mosaic_prob, mixup_prob=0.0)
        if not isinstance(data, MosaicDetection) or getattr(data, '_dataset', None) is not raw:
            raise RuntimeError('MosaicDetection failed to wrap raw COCO dataset')
        self.dataset = data
        if is_distributed:
            if not dist.is_initialized():
                raise RuntimeError('Distributed loader requested without initialized process group')
            world_size = dist.get_world_size()
            if batch_size % world_size:
                raise RuntimeError('Batch size must divide the distributed world size')
            batch_size = batch_size // world_size
            sampler = DistributedSampler(data)
        else:
            sampler = RandomSampler(data)
        batch_sampler = YoloBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False,
            mosaic=not no_aug)
        kwargs = dict(num_workers=self.data_num_workers, pin_memory=True,
                      batch_sampler=batch_sampler,
                      worker_init_fn=_yolo_worker_init_fn)
        if self.data_num_workers > 0:
            kwargs['persistent_workers'] = True
        loader = YoloDataLoader(data, **kwargs)
        return loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        self._preflight()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if rank == 0:
            pair = a41_resize_choice(
                self.input_size, self.multiscale_range, epoch,
                self.max_epoch, self.no_aug_epochs)
            size = torch.tensor(pair, device=device, dtype=torch.long)
        else:
            size = torch.zeros(2, device=device, dtype=torch.long)
        if is_distributed:
            dist.broadcast(size, src=0)
        return tuple(int(v) for v in size.tolist())
