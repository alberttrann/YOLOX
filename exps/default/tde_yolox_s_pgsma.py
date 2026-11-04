"""Isolated vNext.3 experiment; inherits dataset paths from local vNext.2 exp.

IMPORTANT: the vNext.2 data pipeline still uses fixed-capacity labels. Use
`tools/audit_pgsma_integration.py` before any real-data run; this experiment
fails early on dataset overflow where `get_dataset` can inspect counts.
"""
import os
import json
import torch
from exps.default.tde_yolox_s import Exp as VNext2Exp


class Exp(VNext2Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = 'tdevnext_vnext3_pgsma_isolated'
        self.enable_ttt = False
        self.enable_legacy_ttt = False
        self.enable_memory = False
        self.enable_memory_aux = False
        self.enable_proto_aux = False
        self.enable_intra_aux = False
        self.enable_rec_aux = False
        self.enable_afford_aux = False
        self.enable_backbone_attention = False  # optional extras remain ablations
        self.regression_loss_type = 'iou'  # OIoU is a separate ablation
        self.use_baseline_assignments = False
        self.pgsma_dim = 64
        self.pgsma_modes = 4
        self.pgsma_correction = 'direct'
        self.pgsma_variant = 'pgsma'  # or 'memory' for a separately trained control
        self.scpa = False  # SimOTA is the current favored controlled matcher; keep SCPA as explicit ablation.
        self.per_gt_normalization = False
        self.semantic_lr = 0.001
        # NO unverified long training should be started with this setting;
        # run all preflight probes and data-pipeline overflow audit first.
        self.allow_full_training = False

    def get_model(self):
        if 'model' in self.__dict__ and self.model is not None:
            return self.model
        from yolox.models.pgsma_head import PGSMAHead
        from yolox.models.pgsma_model import PGSMAYOLOX
        from yolox.models.yolo_pafpn import YOLOPAFPN
        from yolox.models.tribrid_neck import C2f_Tribrid
        from yolox.models.network_blocks import C2fA
        from yolox.models.darknet import CSPDarknet
        neck = YOLOPAFPN(self.depth,self.width,in_channels=[256,512,1024], act=self.act,
            pretrained_compatible_dark2=True, ttt_residual_adapter=False)
        # The archived YOLOPAFPN creates its own CSPDarknet; no duplicate backbone.
        head = PGSMAHead(self.num_classes,self.width,loss_type=self.regression_loss_type,
            assignment='scpa' if self.scpa else 'baseline',
            per_gt_normalization=self.per_gt_normalization)
        model = PGSMAYOLOX(neck,head,self.num_classes,self.width,self.pgsma_dim,
                           self.pgsma_modes,self.pgsma_correction)
        for block in model.modules():
            if isinstance(block, C2f_Tribrid):
                block.enable_global_context = self.enable_global_context
                block.enable_dense_local = self.enable_dense_local
            if isinstance(block, C2fA):
                block.enable_attention = self.enable_backbone_attention
                block.enable_dense_local = self.enable_dense_local
        model.head.initialize_biases(1e-2)
        model.set_inference_mode(self.pgsma_variant)
        self.model = model
        return model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        # Keep configured aspect ratio rather than silently switching square UAV
        # inputs back to the inherited driving-specific rectangular presets.
        import random
        import torch.distributed as dist
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shape = torch.zeros(2, device=device, dtype=torch.long)
        if rank == 0:
            factor = random.choice((0.875, 1.0, 1.125)) if (
                self.multiscale_range > 0 and epoch < self.max_epoch-self.no_aug_epochs) else 1.0
            h = max(32, round(self.input_size[0] * factor / 32) * 32)
            w = max(32, round(self.input_size[1] * factor / 32) * 32)
            shape[0]=h;shape[1]=w
        if is_distributed: dist.broadcast(shape,src=0)
        return int(shape[0]),int(shape[1])

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Actually invoke MosaicDetection; preserve letterbox masks as 4th channel.

        Mosaic's composite canvas is valid synthetic training content. A plain
        image is top-left letterboxed, so only its resized source rectangle is
        eligible for objectness and hard-background training.
        """
        import torch.distributed as dist
        from torch.utils.data import RandomSampler, DistributedSampler
        from yolox.data import COCODataset, MosaicDetection, YoloBatchSampler
        from yolox.data import DataLoader as YoloDataLoader
        from yolox.data.pgsma_train_transform import PGSMAChannelMaskTransform
        from exps.default.tde_yolox_s import _yolo_worker_init_fn

        # Compute conservative capacity from the source COCO annotation counts.
        meta=self.get_dataset(cache=False,cache_type='ram')
        del meta
        capacity=int(self.annotation_capacity)
        source=COCODataset(data_dir=self.data_dir,json_file=self.train_ann,
                           name='',img_size=self.input_size,
                           preproc=None,cache=False,cache_type='ram')
        source.cache=False;source.cache_type='ram'
        dataset=MosaicDetection(source,img_size=self.input_size,mosaic=not no_aug,
            preproc=PGSMAChannelMaskTransform(capacity,self.flip_prob,self.hsv_prob),
            degrees=self.degrees,translate=self.translate,
            mosaic_scale=self.mosaic_scale,mixup_scale=self.mixup_scale,
            shear=self.shear,enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,mixup_prob=self.mixup_prob)
        self.dataset=dataset
        if is_distributed:
            batch_size=batch_size//dist.get_world_size()
            sampler=DistributedSampler(dataset)
        else:
            sampler=RandomSampler(dataset)
        batch_sampler=YoloBatchSampler(sampler=sampler,batch_size=batch_size,
                                       drop_last=False,mosaic=not no_aug)
        kwargs=dict(num_workers=self.data_num_workers,pin_memory=True,
                    batch_sampler=batch_sampler,worker_init_fn=_yolo_worker_init_fn)
        if self.data_num_workers>0: kwargs['persistent_workers']=True
        return YoloDataLoader(dataset,**kwargs)

    def preprocess(self, inps, targets, tsize):
        """Resize mask using nearest, RGB using inherited YOLOX preprocessing."""
        import torch
        from torch.nn import functional as F
        if inps.ndim!=4 or inps.shape[1]!=4:
            raise RuntimeError('Training loader must return RGB + validity mask in channel 4')
        rgb, validity=inps[:,:3],inps[:,3:4].float()
        rgb,targets=super().preprocess(rgb,targets,tsize)
        if tuple(validity.shape[-2:])!=tuple(rgb.shape[-2:]):
            validity=F.interpolate(validity,size=rgb.shape[-2:],mode='nearest')
        return torch.cat([rgb,validity.to(dtype=rgb.dtype)],dim=1),targets

    def get_optimizer(self, batch_size):
        if 'optimizer' in self.__dict__:
            return self.optimizer
        model = self.get_model()
        lr = self.basic_lr_per_img * batch_size
        decay=[]; no_decay=[]
        for name, p in model.named_parameters():
            if not p.requires_grad or name.startswith('head.pgsma_engine.'):
                continue
            if ('bias' in name or p.ndim <= 1 or 'norm' in name or 'gn.' in name):
                no_decay.append(p)
            else:
                decay.append(p)
        self.optimizer = torch.optim.SGD([
            {'params':decay,'weight_decay':self.weight_decay},
            {'params':no_decay,'weight_decay':0}],
            lr=lr,momentum=self.momentum,nesterov=True)
        return self.optimizer

    def get_dataset(self, cache=False, cache_type='ram'):
        from yolox.data import COCODataset, TrainTransform
        # Conservative padded batching: bound raw annotations plus 4-image
        # mosaic + optional mixup fifth source, rather than silently drop GTs.
        # This is NOT ragged collation or heterogeneous aspect-bucket support.
        max_count = 0
        if os.path.isfile(self.train_ann):
            with open(self.train_ann, encoding='utf8') as fd:
                ann = json.load(fd)['annotations']
            from collections import Counter
            counts = Counter(a['image_id'] for a in ann if not a.get('iscrowd', 0))
            max_count = max(counts.values(), default=0)
        else:
            raise FileNotFoundError('Training annotation JSON required for lossless-capacity audit: '
                                    +str(self.train_ann))
        multiplier = 5 if self.enable_mixup else 4
        capacity = max(160, max_count * multiplier)
        if capacity > 4096:
            raise RuntimeError('Padded annotation capacity exceeds safety envelope: '+str(capacity)
                               +'. Implement real ragged collation; do not truncate.')
        self.annotation_capacity = capacity
        dataset = COCODataset(
            data_dir=self.data_dir, json_file=self.train_ann, name='',
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=capacity, flip_prob=self.flip_prob,
                                  hsv_prob=self.hsv_prob), cache=cache, cache_type=cache_type)
        dataset.cache=cache; dataset.cache_type=cache_type
        return dataset
