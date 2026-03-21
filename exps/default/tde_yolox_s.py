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
        self.max_epoch = 120        # Extended for Contrastive Convergence
        self.warmup_epochs = 15     # Longer stabilization
        self.no_aug_epochs = 20     # Crucial 'pure' phase at the end
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

            # Group 1: CNN Parameters (SGD)
            pg0_cnn, pg1_cnn, pg2_cnn = [], [], []  
            
            # Group 2: Transformer Parameters (AdamW)
            pg_transformer = []

            for k, v in self.model.named_modules():
                # IDENTIFY TRANSFORMER & ENGRAM COMPONENTS
                if "global_mixer" in k or "memory_banks" in k or "uncertainty_gates" in k or "latent_projectors" in k or "ttt_projector" in k or "lightning_indexer" in k:
                    if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                        pg_transformer.append(v.weight)
                    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                        pg_transformer.append(v.bias)
                    # Catch the orthogonal Engram bank if defined as a raw parameter
                    if hasattr(v, "prototypes") and isinstance(v.prototypes, nn.Parameter):
                        pg_transformer.append(v.prototypes)
                    continue # Skip CNN grouping for these

                # STANDARD CNN GROUPING
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2_cnn.append(v.bias)  # biases
                if isinstance(v, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) or "bn" in k or "gn" in k:
                    pg0_cnn.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1_cnn.append(v.weight)  # apply decay

            # 1. Instantiate SGD for CNN
            optimizer_cnn = torch.optim.SGD(
                pg0_cnn, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer_cnn.add_param_group({"params": pg1_cnn, "weight_decay": self.weight_decay})
            optimizer_cnn.add_param_group({"params": pg2_cnn})
            
            # 2. Instantiate AdamW for Transformers
            # AdamW requires a lower learning rate to prevent exploding Softmax gradients
            adam_lr = min(1e-3, lr * 0.1) 
            optimizer_transformer = torch.optim.AdamW(
                pg_transformer, lr=adam_lr, weight_decay=0.05
            )

            # Store both in a dictionary
            self.optimizer = {
                "cnn": optimizer_cnn,
                "transformer": optimizer_transformer
            }
            # Cache the base LRs so the scheduler knows the peaks
            self.base_lrs = {"cnn": lr, "transformer": adam_lr}

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler
        
        # Check if we are using the Dual-Optimizer setup
        if hasattr(self, "base_lrs"):
            lr_cnn = self.base_lrs["cnn"]
            lr_trans = self.base_lrs["transformer"]
            warmup_trans = self.warmup_lr * 0.1 # Scale warmup for AdamW
            
            scheduler_cnn = LRScheduler(
                self.scheduler, lr_cnn, iters_per_epoch, self.max_epoch,
                warmup_epochs=self.warmup_epochs, warmup_lr_start=self.warmup_lr,
                no_aug_epochs=self.no_aug_epochs, min_lr_ratio=self.min_lr_ratio,
            )
            
            scheduler_trans = LRScheduler(
                self.scheduler, lr_trans, iters_per_epoch, self.max_epoch,
                warmup_epochs=self.warmup_epochs, warmup_lr_start=warmup_trans,
                no_aug_epochs=self.no_aug_epochs, min_lr_ratio=self.min_lr_ratio,
            )
            
            return {"cnn": scheduler_cnn, "transformer": scheduler_trans}
            
        else:
            # Fallback for standard YOLOX
            scheduler = LRScheduler(
                self.scheduler, lr, iters_per_epoch, self.max_epoch,
                warmup_epochs=self.warmup_epochs, warmup_lr_start=self.warmup_lr,
                no_aug_epochs=self.no_aug_epochs, min_lr_ratio=self.min_lr_ratio,
            )
            return scheduler