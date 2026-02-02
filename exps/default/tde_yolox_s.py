#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33  
        self.width = 0.50 
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # TDE-YOLOX Hyperparameters
        self.ttt_lr = 0.005        # Learning rate for the inner TTT loop
        self.ttt_noise_std = 0.05  # Standard deviation for DINO-style denoising
        self.num_classes = 80

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
            
            # 1. Instantiate TTT-Aware Backbone
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.ttt_lr,           
                ttt_noise_std=self.ttt_noise_std 
            )
            
            # 2. Instantiate Tribrid Neck
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=in_channels, 
                act=self.act,
            )
            
            # 3. Instantiate the Engram Head
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