#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33  # S-size model depth
        self.width = 0.50  # S-size model width
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # TDE-YOLOX Specific Hyperparameters
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
            
            # 1. Instantiate the TTT-Aware Backbone
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.ttt_lr,           # Passing TTT param
                ttt_noise_std=self.ttt_noise_std # Passing TTT param
            )
            
            # 2. Instantiate the Tribrid Neck (Logic inside YOLOPAFPN needs to support C2f)
            # Note: Ensure you modified YOLOPAFPN to use the Tribrid block as discussed
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=in_channels, 
                act=self.act,
                # If you parameterized the neck class choice, pass it here.
                # Otherwise, if you hardcoded C2f_Tribrid in yolo_pafpn.py, this is fine.
            )
            
            # 3. Instantiate the Engram Head
            head = TDE_Head(
                self.num_classes, 
                self.width, 
                in_channels=in_channels, 
                act=self.act
            )
            
            # 4. Bind Backbone to Neck (Standard YOLOX practice puts backbone inside PAFPN)
            # *Correction*: In YOLOX code, YOLOPAFPN usually *contains* the backbone.
            # You must ensure your modified YOLOPAFPN accepts the custom backbone instance 
            # OR creates the custom CSPDarknet internally. 
            # *Quickest Fix*: Modify YOLOPAFPN.__init__ to use your new CSPDarknet class.
            
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model