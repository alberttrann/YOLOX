#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#yolox/exp/default/tde_yolox_s.py:
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
        self.ttt_noise_std = 0.08  # Harder contrastive denoising task
        
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
            in_channels = [128, 256, 512, 1024] 
            
            # 1. THE ADAPTIVE BACKBONE (Phase 1)
            # Uses GroupNorm + Functional TTT + Learnable LRs
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                out_features=("dark2", "dark3", "dark4", "dark5"), 
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
                in_channels=in_channels,
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

            pg_no_decay = []  # Biases, Norms, Gates, TTT LRs, Tokens, Prototypes
            pg_decay = []     # Conv weights, Linear weights

            # 1. Map parameters by object ID to ensure we don't miss poorly-named modules
            no_decay_param_ids = set()

            for mn, m in self.model.named_modules():
                # Extract weights and biases from ALL standard Normalization layers
                # This perfectly catches the GroupNorm hiding at `projector.net.1`
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                    if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
                        no_decay_param_ids.add(id(m.weight))
                    if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                        no_decay_param_ids.add(id(m.bias))
                
                # Extract gamma and beta from custom Global Response Norm (GRN)
                if m.__class__.__name__ == "GRN":
                    if hasattr(m, "gamma") and isinstance(m.gamma, nn.Parameter):
                        no_decay_param_ids.add(id(m.gamma))
                    if hasattr(m, "beta") and isinstance(m.beta, nn.Parameter):
                        no_decay_param_ids.add(id(m.beta))

            # 2. Iterate through all named parameters
            for pn, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                
                # Biases should never be decayed
                if pn.endswith(".bias"):
                    no_decay_param_ids.add(id(p))
                
                # TDE-YOLOX Elastic Parameters (Must float freely without L2 penalty)
                # Catching: single ttt_lr, mask_token, C2f_Tribrid gates, and Engram prototypes
                if any(x in pn for x in ["ttt_lr", "mask_token", "gate", "prototypes"]):
                    no_decay_param_ids.add(id(p))

                # 3. Route to the correct optimizer group using the robust ID map
                if id(p) in no_decay_param_ids:
                    pg_no_decay.append(p)
                else:
                    pg_decay.append(p)

            # 4. Initialize Optimizer
            optimizer = torch.optim.SGD(
                pg_no_decay, lr=lr, momentum=self.momentum, nesterov=True
            )
            
            optimizer.add_param_group(
                {"params": pg_decay, "weight_decay": self.weight_decay}
            )

            self.optimizer = optimizer

        return self.optimizer