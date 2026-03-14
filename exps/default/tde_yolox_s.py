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
        self.ttt_noise_std = 0.08  # Hard contrastive denoising task
        
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
    
    """
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            # BASE LR for AdamW (usually 1e-3 or 5e-4)
            base_lr = lr 

            # Parameter Groups
            pg_backbone = [] # Pre-trained (Low LR)
            pg_neck_head = [] # Scratch (High LR)
            pg_norms = []    # No Decay

            for k, v in self.model.named_parameters():
                if not v.requires_grad:
                    continue
                
                # group by component
                if "backbone" in k and "dark2" not in k: # Static Backbone parts
                    target_group = pg_backbone
                else:
                    # Dark2 (Adaptive), Neck, Head are all "Active Learning" components
                    target_group = pg_neck_head

                # handle weight decay
                if "bias" in k or "bn" in k or "norm" in k or "gn" in k:
                     # Add to Norm group (No Decay) but keep LR scaling
                     pg_norms.append(v)
                else:
                    target_group.append(v)

            optimizer = torch.optim.AdamW([
                # 1. Static Backbone: Low LR, Standard Decay
                {"params": pg_backbone, "lr": base_lr * 0.1, "weight_decay": self.weight_decay},
                
                # 2. Active Components (TTT/Neck/Head): High LR, Standard Decay
                {"params": pg_neck_head, "lr": base_lr, "weight_decay": self.weight_decay},
                
                # 3. Norms/Biases: Base LR, NO Decay
                {"params": pg_norms, "lr": base_lr, "weight_decay": 0.0},
            ])
            
            self.optimizer = optimizer

        return self.optimizer
        """
    
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            optimized_ids = set()
            pg_norm, pg_weight, pg_bias = [], [], []
            pg_ttt_lrs, pg_gates = [], []
            pg_engram = []  # Will use separate AdamW

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if id(param) in optimized_ids:
                    continue
                optimized_ids.add(id(param))

                if 'ttt_lrs' in name:
                    pg_ttt_lrs.append(param)
                elif any(k in name for k in [
                    'prototypes', 'latent_projectors',
                    'uncertainty_gates', 'memory_banks',
                ]):
                    pg_engram.append(param)
                elif any(k in name for k in [
                    'gate', 'gamma', 'beta', 'mask_token',
                ]):
                    pg_gates.append(param)
                elif any(k in name for k in [
                    'bn.', 'gn.', '.bn', '.gn'
                ]) or name.endswith('.bn.weight') or \
                name.endswith('.bn.bias') or \
                name.endswith('.gn.weight') or \
                name.endswith('.gn.bias'):
                    pg_norm.append(param)
                elif name.endswith('.bias'):
                    pg_bias.append(param)
                else:
                    pg_weight.append(param)

            # --- SGD for all standard + TTT parameters ---
            optimizer = torch.optim.SGD(
                pg_norm, lr=lr,
                momentum=self.momentum, nesterov=True,
                weight_decay=0.0
            )
            optimizer.add_param_group({
                "params": pg_weight, "lr": lr,
                "momentum": self.momentum, "nesterov": True,
                "weight_decay": self.weight_decay,
            })
            optimizer.add_param_group({
                "params": pg_bias, "lr": lr,
                "momentum": self.momentum, "nesterov": True,
                "weight_decay": 0.0,
            })
            optimizer.add_param_group({
                "params": pg_ttt_lrs, "lr": lr * 0.1,
                "momentum": self.momentum, "nesterov": True,
                "weight_decay": 0.0,
            })
            optimizer.add_param_group({
                "params": pg_gates, "lr": lr * 0.1,
                "momentum": self.momentum, "nesterov": True,
                "weight_decay": 0.0,
            })
            self.optimizer = optimizer

            # --- Separate AdamW exclusively for Engram parameters ---
            # AdamW handles sparse, high-variance, competing gradients correctly
            # lr=1e-3 is AdamW's natural scale, independent of SGD's lr schedule
            self.engram_optimizer = torch.optim.AdamW(
                pg_engram,
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=0.0,  # no decay on prototypes
                eps=1e-8
            )

        return self.optimizer