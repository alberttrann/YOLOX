#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.3: Calibrated MuSGD Multi-Engine Optimizer & Anti-Sag Configuration

import os
import torch
import torch.nn as nn
from yolox.exp import Exp as MyExp
from yolox.optimizers.muon import MuSGD


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # --- MODEL SCALE: YOLOX-S ---
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "silu"

        # --- RESEARCH HYPERPARAMETERS (TDE-YOLOX v3.3) ---
        self.num_classes = 9         # BDD100K 9 detection categories
        self.init_ttt_lr = 0.05      # Stage 1 GroupNorm adaptation rate
        self.ttt_noise_std = 0.08    # Contrastive Gaussian denoising intensity

        # --- DATASET & RESOLUTION LOCKING ---
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0    # Locked canvas: prevents grid dimension mismatches
        self.mosaic_prob = 0.0       # Zero crosshair seams (protects Stage 1 TTT variance masker)
        self.enable_mixup = False    # Zero ghost-box artifacts

        # --- SEAM-FREE IN-CANVAS CONTINUOUS AUGMENTATION ---
        # Prevents Epoch 30 clean-domain memorization without introducing artificial border seams:
        self.hsv_prob = 1.0          # Active HSV jitter (photometric invariance)
        self.flip_prob = 0.5         # Horizontal mirroring (50% probability)
        self.degrees = 8.0           # Continuous rotation in [-8°, +8°]
        self.translate = 0.10        # Continuous translation in [-10%, +10%]
        self.mosaic_scale = (0.85, 1.15)  # In-canvas scale jitter (simulates vehicle distance shifts)
        self.shear = 2.0             # Continuous shear in [-2°, +2°]

        # --- STRICT ZSDA PROTOCOL PATHS ---
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_clean_coco.json"
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_val_clean_coco.json"
        self.test_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"

        # --- TRAINING SCHEDULE ---
        self.max_epoch = 80
        self.warmup_epochs = 10
        self.no_aug_epochs = 15      # Activates L1 box offset regression loss (Epochs 65-80)
        self.min_lr_ratio = 0.0      # Full Cosine Annealing to 0.0 (smooth landing; absorbs L1 shock)
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.ema = True

        self.data_num_workers = 0
        self.batch_size = 16

        self.test_conf = 0.001
        self.nmsthre = 0.65

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

            # 1. Adaptive Backbone with GroupNorm Focus & Functional TTT-Dark2
            backbone = CSPDarknet(
                self.depth,
                self.width,
                depthwise=False,
                act=self.act,
                ttt_lr=self.init_ttt_lr,
                ttt_noise_std=self.ttt_noise_std
            )

            # 2. Tribrid BQSA Neck + Scale-AttnRes Highway
            neck = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
            )
            neck.backbone = backbone

            # 3. Decoupled Anti-Collision Engram Head (144 Prototypes on S^127)
            head = TDE_Head(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act
            )

            # 4. Master Model Host & Multi-Task Loss Engine
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.max_epochs = self.max_epoch

        return self.model

    def get_optimizer(self, batch_size):
        """
        CALIBRATED MuSGD OPTIMIZER PARTITION:
          - Group 0 (Muon): Prototypes on S^127 (144 vectors, lr=0.010, lr_factor=4.0, WD=0.0)
          - Group 1 (SGD): Convolutions & Projections (lr=base_lr, lr_factor=1.0, WD=5e-4)
          - Group 2 (SGD): Biases, Normalizations, and Scalers (lr=base_lr, lr_factor=1.0, WD=0.0)
          - Group 3 (Meta-SGD): Stage 1 TTT step sizes (lr=base_lr*0.50, lr_factor=0.50, WD=0.0) -> Unfrozen!
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

            # 1. Prototypes on S^127 -> Muon (Newton-Schulz Polar Retraction)
            if "prototypes" in name:
                pg_muon.append(p)
                registered_params.add(p)
                continue

            # 2. Stage 1 TTT Step Sizes -> Meta-SGD (Unfrozen with lr_factor=0.50)
            if "ttt_lrs" in name:
                pg_meta.append(p)
                registered_params.add(p)
                continue

            # 3. 1D Biases, Normalizations, and Scalar Gates -> SGD No-Decay
            if "bias" in name:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            elif (any(k in name for k in ["gn.", "bn.", "norm.", "bn1.", "net.1."])) and "weight" in name:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            elif any(k in name for k in ["mask_token", "gr_gate", "w_r1", "w_r2", "gamma", "beta", "v_q", "v_k", "omega", "w_hf"]) or name.endswith(".w"):
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            else:
                # 4. Convolutions -> SGD with Weight Decay (5e-4)
                pg_sgd_decay.append(p)
                registered_params.add(p)

        unassigned = all_trainable_params - registered_params
        assert len(unassigned) == 0, f"Unassigned parameters detected: {unassigned}"

        param_groups = [
            # Group 0: Muon (144 Prototypes on S^127: lr=0.010, lr_factor=4.0x)
            {"params": pg_muon, "use_muon": True, "lr": 0.010, "lr_factor": 0.010 / base_lr, "momentum": 0.95, "weight_decay": 0.0, "nesterov": True},
            # Group 1: Convolutions & Projections (Bounded by weight decay 5e-4)
            {"params": pg_sgd_decay, "use_muon": False, "lr": base_lr, "lr_factor": 1.0, "momentum": 0.9, "weight_decay": self.weight_decay, "nesterov": True},
            # Group 2: Biases, Normalizations, and Scalers
            {"params": pg_sgd_no_decay, "use_muon": False, "lr": base_lr, "lr_factor": 1.0, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True},
            # Group 3: Meta-SGD Step Sizes (Unfrozen: lr_factor=0.50x)
            {"params": pg_meta, "use_muon": False, "lr": base_lr * 0.50, "lr_factor": 0.50, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True},
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