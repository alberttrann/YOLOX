#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.7: Conservative Augmentation (Zone 8 Mixup Disabled) & Calibrated Optimization

import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from yolox.exp import Exp as MyExp
from yolox.optimizers.muon import MuSGD

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Model Scale: YOLOX-S
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "silu"

        # Dataset & Training Parameters
        self.num_classes = 9         # BDD100K 9 detection categories
        self.init_ttt_lr = 0.05      # Stage 1 GroupNorm Covariate Moment Alignment rate
        self.ttt_noise_std = 0.08

        # Native 16:9 Rule-of-64 Resolution Standard (576 x 1024, Zero Gray Padding)
        self.input_size = (576, 1024)   # (Height, Width)
        self.test_size = (576, 1024)
        
        # Multiscale range over integer multiples of 64: {(512, 896), (576, 1024), (640, 1152)}
        self.multiscale_range = 1    
        
        # ZONE 8 CONSERVATIVE AUGMENTATION (TDE 2.0.4 & Official Megvii YOLOX-S Standard):
        # 100% 4-quadrant opaque Mosaic preserves high-density scene context;
        # Mixup is DISABLED to prevent synthetic translucency on small 9M YOLOX-S capacity!
        self.mosaic_prob = 0.50
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.mixup_prob = 0.0

        # Rigid Geometry Constraints (Dashcam Perspective Standard)
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 0.0       # Zero rotation: keeps bounding boxes tight on vertical targets
        self.shear = 0.0         # Zero shear: eliminates synthetic affine skewing
        self.translate = 0.1

        # Strict ZSDA Clean-to-Adverse Protocol Paths
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_clean_coco.json"
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_val_clean_coco.json"
        self.test_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"

        # Optimization & Schedule Settings
        self.max_epoch = 80
        self.warmup_epochs = 5
        self.no_aug_epochs = 15    
        self.min_lr_ratio = 0.05
        
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.ema = True

        self.data_num_workers = 4
        self.batch_size = 16

        # Standard training validation threshold (prevents early noise from flooding eval)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()
        if rank == 0:
            if self.multiscale_range > 0:
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

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN
        from yolox.models.engram_head import TDE_Head
        from yolox.models.darknet import CSPDarknet

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
                elif isinstance(m, nn.GroupNorm):
                    m.eps = 1e-5

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]

            backbone = CSPDarknet(
                self.depth,
                self.width,
                depthwise=False,
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std
            )

            neck = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std
            )
            neck.backbone = backbone

            head = TDE_Head(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act
            )

            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.max_epochs = self.max_epoch
        return self.model

    def get_optimizer(self, batch_size):
        """
        CALIBRATED MuSGD OPTIMIZER PARTITION v3.7:
          - Group 0 (Muon): Prototypes on S^127 calibrated to LR = base_lr * 1.5 (0.45 deg/step velocity)
          - Group 1 (SGD Decay): Convolutions, Projections, Decoders (WD = 5e-4)
          - Group 2 (SGD No-Decay): Biases, Norms, aligner.gamma/beta, w_hf, and scalar gates (WD = 0.0)
          - Group 3 (Meta-SGD): Stage 1 TTT Step Sizes ttt_lrs_* (LR = base_lr * 0.05, WD = 0.0)
        """
        if "optimizer" in self.__dict__:
            return self.optimizer

        base_lr = self.basic_lr_per_img * batch_size
        
        all_trainable_params = {p for p in self.model.parameters() if p.requires_grad}
        registered_params = set()

        pg_muon = []
        pg_meta = []
        pg_sgd_decay = []
        pg_sgd_no_decay = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # RULE 1: Prototypes on S^127 -> Muon Engine
            if "prototypes" in name:
                pg_muon.append(p)
                registered_params.add(p)
                continue

            # RULE 2: Stage 1 TTT Step Sizes -> Meta-SGD (Slow Learning Rate)
            if "ttt_lrs" in name:
                pg_meta.append(p)
                registered_params.add(p)
                continue

            # RULE 3: Biases, Norm Affines, Scalar Gates, w_hf, and attnres.w (NO DECAY)
            is_no_decay = (
                "bias" in name
                or any(k in name for k in ["gn.", "bn.", "norm.", "bn1.", "net.1."])
                or any(k in name for k in [
                    "aligner.gamma", "aligner.beta", "gamma", 
                    "gate", "omega", "lambda_fog", "k_sigma", "pool_alpha",
                    "w_hf", "attnres"
                ])
                or (p.ndim <= 1)
            )

            if is_no_decay:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            else:
                # RULE 4: Convolutions and Linear Projections (WITH DECAY)
                pg_sgd_decay.append(p)
                registered_params.add(p)

        unassigned = all_trainable_params - registered_params
        assert len(unassigned) == 0, f"FATAL: Unassigned parameters in optimizer setup: {unassigned}"

        muon_lr = base_lr * 1.5

        logger.info("=" * 80)
        logger.info("CALIBRATED MuSGD OPTIMIZER PARTITION v3.7:")
        logger.info(f"  • Group 0 (Muon S^127)      : {len(pg_muon):3d} tensors | LR={muon_lr:.5f} (1.5x) | WD=0.0")
        logger.info(f"  • Group 1 (SGD with Decay)  : {len(pg_sgd_decay):3d} tensors | LR={base_lr:.5f} (1.0x) | WD={self.weight_decay}")
        logger.info(f"  • Group 2 (SGD No-Decay)    : {len(pg_sgd_no_decay):3d} tensors | LR={base_lr:.5f} (1.0x) | WD=0.0")
        logger.info(f"  • Group 3 (Meta-SGD TTT)    : {len(pg_meta):3d} tensors | LR={base_lr * 0.05:.5f} (0.05x)| WD=0.0")
        logger.info("=" * 80)

        param_groups = [
            # Group 0: Muon (Prototypes on S^127)
            {
                "params": pg_muon, 
                "use_muon": True, 
                "lr": muon_lr, 
                "lr_factor": 1.5, 
                "momentum": 0.95, 
                "weight_decay": 0.0, 
                "nesterov": True
            },
            # Group 1: Convolutions and Linear Projections
            {
                "params": pg_sgd_decay, 
                "use_muon": False, 
                "lr": base_lr, 
                "lr_factor": 1.0, 
                "momentum": self.momentum, 
                "weight_decay": self.weight_decay, 
                "nesterov": True
            },
            # Group 2: Protected Biases, Norms, w_hf, and ScaleAttnRes.w
            {
                "params": pg_sgd_no_decay, 
                "use_muon": False, 
                "lr": base_lr, 
                "lr_factor": 1.0, 
                "momentum": self.momentum, 
                "weight_decay": 0.0, 
                "nesterov": True
            },
            # Group 3: Meta-SGD Step Sizes (ttt_lrs_*)
            {
                "params": pg_meta, 
                "use_muon": False, 
                "lr": base_lr * 0.05, 
                "lr_factor": 0.05, 
                "momentum": self.momentum, 
                "weight_decay": 0.0, 
                "nesterov": True
            },
        ]

        self.optimizer = MuSGD(param_groups, muon=1.0, sgd=0.0)
        return self.optimizer

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        target_ann = self.test_ann if testdev else self.val_ann
        eval_data_dir = getattr(self, "eval_data_dir", self.data_dir)
        
        return COCODataset(
            data_dir=eval_data_dir,
            json_file=target_ann,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {
            "num_workers": 0,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)