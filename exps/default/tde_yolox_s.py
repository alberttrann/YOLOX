#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # --- MODEL SCALE: SMALL (YOLOX-S) ---
        self.depth = 0.33  
        self.width = 0.50 
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "silu"

        # --- TDE-YOLOX RESEARCH CONFIG ---
        self.num_classes = 9       # BDD Detection categories
        self.init_ttt_lr = 0.02    # Aggressive start for Norm adaptation
        self.ttt_noise_std = 0.1  # Harder contrastive denoising task
        
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_coco.json"
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"
        self.test_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"

        # --- TRAINING SCHEDULE ---
        self.max_epoch = 80        
        self.warmup_epochs = 10
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05     
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 0.0005 
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1     
        self.ema = True            

        self.data_num_workers = 0
        self.batch_size = 16
        self.accum_steps = 8

        self.multiscale_range = 5    
        self.test_size = (640, 640)
        self.test_conf = 0.01        
        self.nmsthre = 0.65

    def get_model(self):
        """
        Wires Stage 1 (TTT) -> Neck (Tribrid) -> Head (Engram).
        """
        from yolox.models import YOLOX, YOLOPAFPN
        from yolox.models.engram_head import TDE_Head
        from yolox.models.darknet import CSPDarknet

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            # P3, P4, P5 Channels for YOLOX-S (Width = 0.5)
            # Standard YOLOX-S outputs: 128, 256, 512
            in_channels = [128, 256, 512] 
            
            # 1. THE ADAPTIVE BACKBONE (Phase 1)
            # Uses GroupNorm + Functional TTT + Learnable LRs
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.init_ttt_lr,           
                ttt_noise_std=self.ttt_noise_std 
            )
            
            # 2. THE TRIBRID NECK (Phase 2)
            # Replaces standard CSPLayer with C2f_Tribrid (DSA + SimAM + MBConv)
            # pass the backbone to PAFPN to maintain YOLOX structure
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=[256, 512, 1024], # Darknet stages before width multiplier
                act=self.act,
            )
            # Override the internal backbone with TTT version
            neck.backbone = backbone
            
            # 3. THE ENGRAM HEAD (Phase 3)
            # Differentiable Identity Restoration + Uncertainty Gating
            head = TDE_Head(
                self.num_classes, 
                self.width, 
                in_channels=[256, 512, 1024], 
                act=self.act
            )
            
            # 4. THE UNIFIED TDE-YOLOX WRAPPER
            # Manages the TTT annealing schedule and Memory Anchor Loss
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        
        # Set the total training epochs for the TTT probability scheduler
        self.model.max_epochs = self.max_epoch
        
        return self.model

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
    
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg_no_decay = []      # Biases and Norms (No Weight Decay)
            pg_decay_head = []    # New components (Full LR, Weight Decay)
            pg_decay_backbone = [] # Pre-trained components (Half LR, Weight Decay)

            for k, v in self.model.named_modules():
                # 1. NO DECAY GROUP: Biases and all Normalization layers
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg_no_decay.append(v.bias)
                
                # Exclude ALL Norms (BatchNorm, GroupNorm, GRN, LayerNorm) from WD
                if isinstance(v, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) or "bn" in k or "norm" in k or "GRN" in v.__class__.__name__:
                    if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                        pg_no_decay.append(v.weight)
                        
                # 2. DECAY GROUPS: Convolutions, Linear layers, Attention weights
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    # LLRD Logic: Check if parameter is in the backbone
                    if "backbone.backbone" in k: 
                        # This targets CSPDarknet inside YOLOPAFPN
                        pg_decay_backbone.append(v.weight)
                    else:
                        # This targets Tribrid Neck, Engram Head, and TTT Projector
                        pg_decay_head.append(v.weight)

            # Initialize SGD
            optimizer = torch.optim.SGD(
                pg_no_decay, lr=lr, momentum=self.momentum, nesterov=True
            )
            
            # Group 1: New Components (Full LR, standard WD)
            optimizer.add_param_group(
                {"params": pg_decay_head, "weight_decay": self.weight_decay}
            )
            
            # Group 2: Backbone Components (LLRD: 50% LR, standard WD)
            optimizer.add_param_group(
                {"params": pg_decay_backbone, "weight_decay": self.weight_decay, "lr": lr * 0.5}
            )

            self.optimizer = optimizer

        return self.optimizer