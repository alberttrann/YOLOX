#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Unified MuSGD Optimizer & Strict ZSDA Protocol

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

        # --- RESEARCH HYPERPARAMETERS (TDE-YOLOX v3.1) ---
        self.num_classes = 9         # BDD100K 9 detection categories
        self.init_ttt_lr = 0.05      # Stage 1 GroupNorm adaptation rate
        self.ttt_noise_std = 0.08    # Contrastive Gaussian denoising intensity

        # --- DATASET & RESOLUTION LOCKING ---
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0    # Strictly lock canvas to 640x640 (preserves 10x10 block grid)
        self.mosaic_prob = 0.0       # Disabled from Epoch 1 (prevents seam variance corruption)
        self.enable_mixup = False    # Disabled from Epoch 1

        # --- STRICT ZSDA PROTOCOL PATHS ---
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_clean_coco.json"
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_val_clean_coco.json"
        self.test_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"

        # --- TRAINING SCHEDULE ---
        self.max_epoch = 80
        self.warmup_epochs = 10
        self.no_aug_epochs = 15      # Activates L1 box offset regression loss
        self.min_lr_ratio = 0.05
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

            # 3. Decoupled Anti-Collision Engram Head
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
        NATIVE MuSGD OPTIMIZER PARTITION:
          - Group 0 (Muon): Prototypes [576, 128], BQSA projections, manifold decoders (lr=0.02, lr_factor=8.0, WD=0.0)
          - Group 1 (SGD): 4D Convolutions (lr=base_lr, lr_factor=1.0, WD=5e-4)
          - Group 2 (SGD): Normalizations, Biases, and Scalers (lr=base_lr, lr_factor=1.0, WD=0.0)
          - Group 3 (Meta-SGD): Stage 1 TTT learning rates (lr=base_lr*0.05, lr_factor=0.05, WD=0.0)
        """
        if "optimizer" in self.__dict__:
            return self.optimizer

        base_lr = self.basic_lr_per_img * batch_size
        all_params = set(self.model.parameters())
        registered_params = set()

        pg_muon = []
        pg_meta = []
        pg_sgd_decay = []
        pg_sgd_no_decay = []

        # Partition parameters strictly
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # 1. 2D Matrices and Prototypes -> Muon
            if "prototypes" in name:
                pg_muon.append(p)
                registered_params.add(p)
                continue

            if any(k in name for k in ["manifold_decoders", "anchor_projectors", "local_projectors", "p4_bqsa.qkv", "p4_bqsa.proj", "p3_bqsa.qkv", "p3_bqsa.proj", "p5_dense_attn.qkv", "p5_dense_attn.proj"]):
                if p.dim() in [2, 4]:
                    pg_muon.append(p)
                    registered_params.add(p)
                    continue

            # 2. Stage 1 TTT Step Sizes -> Meta-SGD
            if "ttt_lrs" in name:
                pg_meta.append(p)
                registered_params.add(p)
                continue

            # 3. 1D Biases, Normalizations, and Scalers -> SGD No-Decay
            if "bias" in name:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            elif (any(k in name for k in ["gn.", "bn.", "norm.", "bn1.", "net.1."])) and "weight" in name:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            elif any(k in name for k in ["attnres", ".w", "mask_token", "gr_gate", "w_r1", "w_r2", "gamma", "beta", "v_q", "v_k", "omega", "w_hf"]):
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            else:
                # 4. Standard 4D Convolution Weights -> SGD Decay
                pg_sgd_decay.append(p)
                registered_params.add(p)

        # Zero parameter omissions assertion
        unassigned = all_params - registered_params
        assert len(unassigned) == 0, f"Unassigned parameters detected: {unassigned}"

        # Native Unified Parameter Groups
        param_groups = [
            # Group 0: Muon (lr_factor = 0.02 / 0.0025 = 8.0x)
            {"params": pg_muon, "use_muon": True, "lr": 0.02, "lr_factor": 0.02 / base_lr, "momentum": 0.95, "weight_decay": 0.0},
            # Group 1: Standard 4D Convolutions
            {"params": pg_sgd_decay, "use_muon": False, "lr": base_lr, "lr_factor": 1.0, "momentum": 0.9, "weight_decay": self.weight_decay, "nesterov": True},
            # Group 2: Biases, Normalizations, and Scalers
            {"params": pg_sgd_no_decay, "use_muon": False, "lr": base_lr, "lr_factor": 1.0, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True},
            # Group 3: Meta-SGD Step Sizes
            {"params": pg_meta, "use_muon": False, "lr": base_lr * 0.05, "lr_factor": 0.05, "momentum": 0.9, "weight_decay": 0.0, "nesterov": True},
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
                max_labels=120,  # Restores full capacity for dense traffic scenes
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