#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Model Scale: Small (Standard YOLOX-S)
        self.depth = 0.33  
        self.width = 0.50 
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # --- TDE-YOLOX Research Config ---
        self.num_classes = 9 # BDD Detection categories filtered in preprocessing
        self.ttt_lr = 0.005        
        self.ttt_noise_std = 0.05  
        
        # --- DATASET OVERRIDES ---
        self.data_dir = "C:/TDE-YOLOX/bdd100k/bdd100k/images"
        self.train_ann = "tde_train_coco.json"
        # use the adverse set for 'val' to see mAP on OOD images during training
        self.val_ann = "tde_test_adverse_coco.json" 
        
        # Training Schedule
        self.max_epoch = 80 # TTT needs fewer epochs because meta-learning is efficient
        self.warmup_epochs = 10    # Longer warmup for Engram stability
        self.no_aug_epochs = 15    # Longer 'pure' training at the end
        self.ema = True            # Keep EMA on to stabilize OOD features
        self.basic_lr_per_img = 0.01 / 64.0 # Standard YOLOX LR

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
            # P3, P4, P5 Channels for YOLOX-S
            in_channels = [128, 256, 512] 
            
            # 1. Instantiate TTT-Aware Backbone (Phase 1)
            # Operates Functional Meta-Learning on Stage 1 Features
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.ttt_lr,           
                ttt_noise_std=self.ttt_noise_std 
            )
            
            # 2. Instantiate Tribrid Neck (Phase 2)
            # Replaces standard CSPLayer with C2f_Tribrid (MaxViT + DSA + SimAM)
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=[256, 512, 1024], # Input from darknet stages
                act=self.act,
            )
            
            # 3. Instantiate the Engram Head (Phase 3)
            # Differentiable Associative Memory restoration
            head = TDE_Head(
                self.num_classes, 
                self.width, 
                in_channels=[256, 512, 1024], 
                act=self.act
            )
            
            # backbone here is neck because YOLOPAFPN wraps the backbone
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model