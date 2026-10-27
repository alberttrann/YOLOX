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
        self.act = "silu"
        self.num_classes = 9

        # --- MATCH BASELINE NATIVE 16:9 RESOLUTION (ZERO GRAY PADDING) ---
        self.input_size = (576, 1024)  # (Height, Width)
        self.test_size = (576, 1024)

        # --- ACDC RAIN DATASET PATHS ---
        self.data_dir = r"D:\ACDC\rgb_anon" 
        self.val_ann = r"D:\YOLOX-3rd\acdc_labels\acdc_rain_bdd.json"
        self.test_ann = r"D:\YOLOX-3rd\acdc_labels\acdc_rain_bdd.json"

        # TDE Meta-Parameters
        self.ttt_lr = 0.02        
        self.ttt_noise_std = 0.08
        self.ttt_mask_ratio = 0.20

        # Evaluation thresholds matching baseline
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann,
            name="",  # Prevents YOLOX from inserting "\val2017\" into the path
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def get_dataset(self, **kwargs):
        return self.get_eval_dataset(**kwargs)

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
            # 1. Instantiate Stage 1 TTT-Aware Backbone
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.ttt_lr,           
                ttt_noise_std=self.ttt_noise_std,
                ttt_mask_ratio=self.ttt_mask_ratio
            )
            
            # 2. Instantiate Tribrid Neck (C2f_Tribrid with Continuous Saliency Linear Attention)
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=[256, 512, 1024],
                act=self.act,
            )
            # Explicitly wire our custom adaptive backbone into the neck
            neck.backbone = backbone
            
            # 3. Instantiate the Engram Head
            head = TDE_Head(
                self.num_classes, 
                self.width, 
                in_channels=[256, 512, 1024], 
                act=self.act
            )
            
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model