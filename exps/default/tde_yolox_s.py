#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Triple-Engine Optimizer & Strict ZSDA Protocol

import os
import torch
import torch.nn as nn
from yolox.exp import Exp as MyExp
from yolox.optimizers.muon import Muon, CombinedOptimizer


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

        # --- DATASET & RESOLUTION LOCKING (Defect M4, F2) ---
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0    # Strictly lock canvas to 640x640 (prevents shape crashes)
        self.mosaic_prob = 0.0       # Disabled from Epoch 1 (prevents seam variance corruption)
        self.enable_mixup = False    # Disabled from Epoch 1

        # --- STRICT ZSDA PROTOCOL PATHS (Defect D7, S6, K6) ---
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        # Train strictly on clean daylight images from BDD train set
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_clean_coco.json"
        # Validate strictly on clean images for model selection (No transductive leak!)
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_val_clean_coco.json"
        # Adverse test set evaluated only once on frozen final checkpoint
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
        TRIPLE-ENGINE OPTIMIZER PARTITION (Defects K1, Z2, O1, Factor 4).
        Partitions 100% of parameters with zero omissions:
          - Engine 1 (Muon): Prototypes [576, 128], BQSA linear projections, manifold decoders
          - Engine 2 (SGD): Convolutions (decay=5e-4), Normalization & Biases (decay=0.0)
          - Engine 3 (Meta-SGD): Stage 1 TTT learning rates (lr=5e-4, decay=0.0)
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

        # 1. Partition Muon Parameters (2D Matrices: Prototypes, 1x1 Projections, Decoders)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # Identify Engram Prototypes [576, 128]
            if "prototypes" in name:
                pg_muon.append(p)
                registered_params.add(p)
                continue

            # Identify BQSA 1x1 Projections and Head Decoders
            if any(k in name for k in ["manifold_decoders", "anchor_projectors", "p4_bqsa.qkv", "p4_bqsa.proj", "p3_bqsa.qkv", "p3_bqsa.proj", "p5_dense_attn.qkv", "p5_dense_attn.proj"]):
                if "weight" in name and p.dim() in [2, 4]:
                    pg_muon.append(p)
                    registered_params.add(p)
                    continue

            # 2. Partition Meta-SGD Parameters (Stage 1 TTT Learning Rates)
            if "ttt_lrs" in name:
                pg_meta.append(p)
                registered_params.add(p)
                continue

            # 3. Partition Momentum SGD Parameters
            if "bias" in name:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            elif any(k in name for k in ["gn.", "bn.", "norm."]) and "weight" in name:
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            elif any(k in name for k in ["attnres", ".w", "mask_token", "gr_gate", "gamma", "beta", "v_q", "v_k", "omega", "w_hf"]):
                pg_sgd_no_decay.append(p)
                registered_params.add(p)
            else:
                pg_sgd_decay.append(p)
                registered_params.add(p)

        # CRITICAL ASSERTION: Zero parameters dropped!
        unassigned = all_params - registered_params
        assert len(unassigned) == 0, f"Defect K1 failure: {len(unassigned)} parameters dropped from optimizer: {unassigned}"

        # Instantiate Engine 1: Muon Optimizer
        opt_muon = Muon(
            pg_muon,
            lr=0.02,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            weight_decay=0.0
        )

        # Instantiate Engine 2 & 3: Multi-Group Momentum SGD
        sgd_groups = [
            {"params": pg_sgd_decay, "weight_decay": self.weight_decay, "lr": base_lr},
            {"params": pg_sgd_no_decay, "weight_decay": 0.0, "lr": base_lr},
            {"params": pg_meta, "weight_decay": 0.0, "lr": base_lr * 0.05}  # Meta-LR velocity reduction (O1)
        ]
        opt_sgd = torch.optim.SGD(sgd_groups, momentum=self.momentum, nesterov=True)

        # Wrap into unified CombinedOptimizer
        self.optimizer = CombinedOptimizer([opt_muon, opt_sgd])
        return self.optimizer

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )