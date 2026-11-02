#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.exp import Exp as MyExp


# GLOBAL LEVEL FUNCTION: Must reside at module scope to be picklable on Windows spawn
def _yolo_worker_init_fn(worker_id):
    import cv2
    from yolox.data import worker_init_reset_seed
    worker_init_reset_seed(worker_id)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


class Exp(MyExp):
    """
    TDE-YOLOX Master Experiment Configuration (Hardened vFinal).
    """

    def __init__(self):
        super(Exp, self).__init__()
        # --- MODEL SCALE: STANDARD YOLOX-S ---
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "tde_yolox_s_full_vnext2"  # separate checkpoints from prior runs
        self.act = "silu"

        # --- TDE-YOLOX RESEARCH CONFIGURATION ---
        self.num_classes = 9           # BDD100K 9-class detection subset
        self.init_ttt_lr = 0.02        # Calibrated initial step size (logit link)
        self.ttt_noise_std = 0.08      # Denoising perturbation
        self.ttt_mask_ratio = 0.20     # Exact 20% relative contrast Sobel masking
        self.afford_loss_weight = 0.25  # Training-only dense affordance supervision weight
        self.recovery_mode = False  # Full research architecture; not the prior ablation-only recovery run
        # Independently toggle one proposed capability per ablation.
        self.pretrained_compatible_dark2 = True
        self.enable_ttt = True  # pretrained-compatible feature-space residual TTT
        self.enable_legacy_ttt = False  # legacy GN weight adaptation remains disallowed
        self.enable_spi = True
        self.enable_global_context = True
        self.enable_dense_local = True
        self.enable_backbone_attention = True
        self.enable_memory = True
        self.enable_memory_aux = True
        self.enable_proto_aux = True
        self.enable_intra_aux = True
        self.enable_rec_aux = True
        self.enable_afford_aux = True
        self.use_baseline_assignments = False
        self.regression_loss_type = "oiou"

        # --- NATIVE 16:9 RULE-OF-64 RESOLUTION PROTOCOL ---
        self.input_size = (576, 1024)   # (Height, Width) - 0% Letterbox Padding Waste
        self.test_size = (576, 1024)
        self.multiscale_range = 1

        # --- AUGMENTATION CURRICULUM ---
        self.mosaic_prob = 0.50
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.shear = 0.0
        self.translate = 0.1

        # --- STRICT ZSDA PROTOCOL PATHS ---
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_clean_coco.json"
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_val_clean_coco.json"
        self.test_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"

        # --- TRAINING & CURRICULUM SCHEDULE ---
        # --- TRAINING & CURRICULUM SCHEDULE (Calibrated for Pre-trained Fine-Tuning) ---
        self.max_epoch = 80
        # When fine-tuning from yolox_s.pth, 1-2 epochs of warmup is standard (10 epochs is for scratch)
        self.warmup_epochs = 10 if self.recovery_mode else 1
        self.no_aug_epochs = 15
        self.ramp_epochs = 50
        self.aux_ramp_epochs = 12  # active at step one, smoothly reach nominal auxiliary weights
        self.prob_start = 0.10
        self.prob_end = 0.60

        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.ema = True

        self.data_num_workers = 4
        self.batch_size = 16

        # Crowd-Safe Evaluation Thresholds
        self.test_conf = 0.01
        self.nmsthre = 0.65 if self.recovery_mode else 0.60

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()
        if rank == 0:
            if self.multiscale_range > 0 and (self.recovery_mode or epoch < (self.max_epoch - self.no_aug_epochs)):
                scale_choices = [(512, 896), (576, 1024), (640, 1152)]
                h, w = random.choice(scale_choices)
                tensor[0] = h
                tensor[1] = w
            else:
                tensor[0] = self.input_size[0]
                tensor[1] = self.input_size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        return (tensor[0].item(), tensor[1].item())

    def get_optimizer(self, batch_size):
        if "optimizer" in self.__dict__:
            return self.optimizer

        lr = self.basic_lr_per_img * batch_size
        pg_decay, pg_no_decay = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                "bias" in name
                or p.ndim <= 1
                or "norm" in name
                or "gn." in name
                or "gamma" in name
                or "ttt_lrs" in name
                or "mask_token" in name
                or "prototypes" in name
            ):
                pg_no_decay.append(p)
            else:
                pg_decay.append(p)

        optimizer_grouped_parameters = [
            {"params": pg_decay, "weight_decay": self.weight_decay, "lr": lr},
            {"params": pg_no_decay, "weight_decay": 0.0, "lr": lr},
        ]

        self.optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            momentum=self.momentum,
            nesterov=True,
        )
        return self.optimizer

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120 if self.recovery_mode else 160,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )
        # Defense against CacheDataset.__del__ attribute error
        dataset.cache = cache
        dataset.cache_type = cache_type
        return dataset

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        testdev = kwargs.get("testdev", False)
        target_ann = self.test_ann if testdev else self.val_ann

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=target_ann,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )
        dataset.cache = False
        dataset.cache_type = "ram"
        return dataset

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        return torch.utils.data.DataLoader(
            valdataset,
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
        )

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN
        from yolox.models.darknet import CSPDarknet
        from yolox.models.engram_head import TDE_Head

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            if self.enable_memory_aux and not self.enable_memory:
                raise ValueError("Memory CE requires enable_memory=True")
            if self.enable_rec_aux and not (self.enable_ttt or self.enable_legacy_ttt):
                raise ValueError("Reconstruction loss requires an enabled TTT implementation")
            if self.pretrained_compatible_dark2 and self.enable_legacy_ttt:
                raise ValueError("Legacy TTT requires GN dark2 and destroys native BN parity. "
                                 "Implement a separate residual GN adapter before enabling it.")
            backbone = CSPDarknet(
                self.depth,
                self.width,
                depthwise=False,
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std,
                ttt_mask_ratio=self.ttt_mask_ratio,
                pretrained_compatible_dark2=self.pretrained_compatible_dark2,
                ttt_residual_adapter=self.enable_ttt and self.pretrained_compatible_dark2,
            )

            neck = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=[256, 512, 1024],
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std,
                ttt_mask_ratio=self.ttt_mask_ratio,
                pretrained_compatible_dark2=self.pretrained_compatible_dark2,
                ttt_residual_adapter=self.enable_ttt and self.pretrained_compatible_dark2,
            )
            neck.backbone = backbone

            head = TDE_Head(
                self.num_classes,
                self.width,
                in_channels=[256, 512, 1024],
                act=self.act,
                loss_type=self.regression_loss_type,
            )

            self.model = YOLOX(
                neck,
                head,
                afford_loss_weight=self.afford_loss_weight,
            )
            from yolox.models.network_blocks import C2fA
            from yolox.models.tribrid_neck import C2f_Tribrid
            self.model.train_ttt_enabled = self.enable_ttt
            self.model.eval_ttt_enabled = self.enable_ttt
            neck.enable_spi = self.enable_spi
            head.enable_memory_restoration = self.enable_memory
            head.use_baseline_assignments = self.use_baseline_assignments
            self.model.mem_loss_weight = 0.10 if self.enable_memory_aux else 0.0
            self.model.proto_loss_weight = 0.10 if self.enable_proto_aux else 0.0
            self.model.intra_loss_weight = 0.05 if self.enable_intra_aux else 0.0
            self.model.rec_loss_weight = 0.05 if self.enable_rec_aux else 0.0
            self.model.afford_loss_weight = self.afford_loss_weight if self.enable_afford_aux else 0.0
            for module in self.model.modules():
                if isinstance(module, C2f_Tribrid):
                    module.enable_global_context = self.enable_global_context
                    module.enable_dense_local = self.enable_dense_local
                elif isinstance(module, C2fA):
                    module.enable_attention = self.enable_backbone_attention
                    module.enable_dense_local = self.enable_dense_local

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        self.model.max_epochs = self.max_epoch
        self.model.warmup_epochs = self.warmup_epochs
        self.model.ramp_epochs = self.ramp_epochs
        self.model.aux_ramp_epochs = self.aux_ramp_epochs
        self.model.prob_start = self.prob_start
        self.model.prob_end = self.prob_end

        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from torch.utils.data import DistributedSampler, RandomSampler
        from yolox.data import (
            YoloBatchSampler,
            DataLoader as YoloDataLoader,
        )

        dataset = self.get_dataset(cache=cache_img, cache_type="ram")

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": _yolo_worker_init_fn,  # Uses picklable global function
        }

        if self.data_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True

        return YoloDataLoader(dataset, **dataloader_kwargs)
