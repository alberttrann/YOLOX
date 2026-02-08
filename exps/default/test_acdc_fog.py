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
        
        self.num_classes = 9 
        self.ttt_lr = 0.005        
        self.ttt_noise_std = 0.05  
        
        self.data_dir = r"D:\ACDC\rgb_anon" 
        self.val_ann = r"D:\YOLOX-3rd\acdc_labels\acdc_fog_bdd.json"
        self.test_ann = r"D:\YOLOX-3rd\acdc_labels\acdc_fog_bdd.json"
        self.test_size = (640, 640) # STRICT CONSISTENCY
        self.input_size = (640, 640)
        
        # --- e1->e72 ---
        """
        This phase is also paired with an annealing ramp of TTT probability from 10% to 50%:
        Schedule:
        - Epoch 0-10: Fixed 0.1 (Warmup stability)
        - Epoch 11-40: Linear Ramp 0.1 -> 0.5 (Curriculum Learning)
        - Epoch 40-72: Fixed 0.5 (High-fidelity Adaptation)

        if not self.training:
            return 1.0 # Always TTT during inference/validation
            
        if self.current_epoch < self.warmup_end:
            return self.prob_start
        
        if self.current_epoch >= self.ramp_end:
            return self.prob_end
        
            
        # Linear Ramp
        progress = (self.current_epoch - self.warmup_end) / (self.ramp_end - self.warmup_end)
        return self.prob_start + (self.prob_end - self.prob_start) * progress
        """
        #self.max_epoch = 80 # TTT needs fewer epochs because meta-learning is efficient
        #self.max_epoch = 80 # TTT needs fewer epochs because meta-learning is efficient
        #self.no_aug_epochs = 15    # Longer 'pure' training at the end
        # --- e1->e72 ---

        # --- 73++ RECONFIGURATION ---
        """
        This phase simulates a stress-test by pushing TTT probability to 70% in the final epochs:
        # --- 73++ STRESS TEST ---
        #if self.current_epoch > 72:
            #return 0.7 # Push to 70% TTT in final phase for stress testing
        # --- 73++ STRESS TEST ---
        """
        #self.max_epoch = 100        # Extend the timeline
        #self.no_aug_epochs = 35     # Ensure Mosaic stays OFF for the rest of training
        #self.min_lr_ratio = 0.05    # Lock LR at 5% floor (Precision Mode)
        #self.weight_decay = 0.00025 # Reduce regularization to preserve identity
        # --- 73++ RECONFIGURATION ---

        # --- 80++ RECONFIGURATION ---
        """
        This phase implements a 'gentle landing', with the intention to stabilize the Engram memory:
        - From Epoch 80 onwards, reduce TTT probability to 20% to avoid feature
        # --- GENTLE LANDING ---
        if self.current_epoch >= 79:
            return 0.2  # Low probability to stabilize features
        """
        self.max_epoch = 100        
        self.no_aug_epochs = 35     # Keep Augmentation OFF       
        # RESTORE STANDARD REGULARIZATION
        self.weight_decay = 0.0005 
        # LET LR DECAY TO ZERO (Remove the floor clamp)
        self.min_lr_ratio = 0.0
        # --- 80++ RECONFIGURATION ---

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