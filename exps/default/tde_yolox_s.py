#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    """
    TDE-YOLOX Master Experiment Configuration (Surgical vNext).
    
    Surgical Fixes Applied:
      - Patch 6.1: Expands TrainTransform max_labels from 120 to 160 (prevents annotation truncation in dense frames).
      - Patch 6.2: Decouples prototypes, gates, and ttt_lrs from weight decay to protect hyperspherical geometry.
      - Full Integration: Binds loss_type="oiou" to TDE_Head and afford_loss_weight=0.25 to YOLOX.
      - Protocol Safety: Enforces strict ZSDA evaluation split paths (zero adverse data in training/val).
    """

    def __init__(self):
        super(Exp, self).__init__()
        # --- MODEL SCALE: STANDARD YOLOX-S ---
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "silu"

        # --- TDE-YOLOX RESEARCH CONFIGURATION ---
        self.num_classes = 9           # BDD100K 9-class detection subset
        self.init_ttt_lr = 0.02        # Aggressive start for GroupNorm instance adaptation
        self.ttt_noise_std = 0.08      # Contrastive DINO-style denoising perturbation
        self.ttt_mask_ratio = 0.20     # Exact 20% relative contrast Sobel masking (Fix 14.1)
        self.afford_loss_weight = 0.25  # Training-only dense affordance supervision weight

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
        self.max_epoch = 80
        self.warmup_epochs = 10
        self.no_aug_epochs = 15
        self.ramp_epochs = 50
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
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        """
        Quantized 16:9 Rule-of-64 multiscale resizing.
        Locks resolution to exact target (576, 1024) during the no_aug fine-tuning phase.
        """
        tensor = torch.LongTensor(2).cuda()
        if rank == 0:
            if self.multiscale_range > 0 and epoch < (self.max_epoch - self.no_aug_epochs):
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
        """
        PATCH 6.2 APPLIED:
        Isolates biases, 1D norm parameters, gates, ttt_lrs, and prototypes from weight decay.
        Protects hyperspherical prototype norms ||P||_2 from collapsing toward zero.
        """
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
                or "gate" in name
                or "ttt_lrs" in name
                or "mask_token" in name
                or "prototypes" in name  # PATCH 6.2: Prototype Weight Decay Immunity
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

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",  # Prevents YOLOX from prepending 'train2017' into paths
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=160,  # PATCH 6.1 APPLIED: Expands label capacity from 120 to 160
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        testdev = kwargs.get("testdev", False)
        target_ann = self.test_ann if testdev else self.val_ann

        return COCODataset(
            data_dir=self.data_dir,
            json_file=target_ann,
            name="",  # Prevents YOLOX from prepending 'val2017' into paths
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        return torch.utils.data.DataLoader(
            valdataset,
            num_workers=0,  # Prevents Windows shared-memory allocation crashes
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
            # 1. Instantiate CSPDarknet with GroupNorm Stage 1 (dark2) and deep C2fA stages (dark4, dark5)
            backbone = CSPDarknet(
                self.depth,
                self.width,
                depthwise=False,
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std,
                ttt_mask_ratio=self.ttt_mask_ratio,
            )

            # 2. Instantiate YOLOPAFPN with integrated SPI Highway and C2f_Tribrid neck
            neck = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=[256, 512, 1024],
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std,
                ttt_mask_ratio=self.ttt_mask_ratio,
            )
            neck.backbone = backbone

            # 3. Instantiate TDE_Head with OIoU loss and decoupled non-destructive FiLM Engram memory
            head = TDE_Head(
                self.num_classes,
                self.width,
                in_channels=[256, 512, 1024],
                act=self.act,
                loss_type="oiou",
            )

            # 4. Instantiate YOLOX Master Wrapper with Affordance Decoder
            self.model = YOLOX(
                neck,
                head,
                afford_loss_weight=self.afford_loss_weight,
            )

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        # Synchronize dynamic curriculum parameters to the model
        self.model.max_epochs = self.max_epoch
        self.model.warmup_epochs = self.warmup_epochs
        self.model.ramp_epochs = self.ramp_epochs
        self.model.prob_start = self.prob_start
        self.model.prob_end = self.prob_end

        return self.model