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
        self.data_dir = r"D:\ACDC\rgb_anon" 
        self.val_ann = r"D:\YOLOX-3rd\acdc_labels\acdc_rain_bdd.json"
        self.test_ann = r"D:\YOLOX-3rd\acdc_labels\acdc_rain_bdd.json"
        self.input_size = (576, 1024)   # (Height, Width)
        self.test_size = (576, 1024)
        
        # UNIFIED EVALUATION THRESHOLDS (Fix 28):
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
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
            
            # 1. Instantiate Backbone with calibrated test-time TTT learning rate
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.ttt_lr,           # Passes calibrated 0.005!
                ttt_noise_std=self.ttt_noise_std 
            )
            
            # 2. Instantiate Neck and EXPLICITLY HOOK the configured backbone
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=in_channels,
                act=self.act,
                ttt_lr=self.ttt_lr,
                ttt_noise_std=self.ttt_noise_std
            )
            neck.backbone = backbone  # <--- WIRED EXPLICITLY (Fixes Flaw 25!)
            
            # 3. Instantiate Engram Head
            head = TDE_Head(
                self.num_classes, 
                self.width, 
                in_channels=in_channels, 
                act=self.act
            )
            
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model