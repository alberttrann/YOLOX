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
        self.ttt_mask_ratio = 0.20 # Exact 20% relative contrast masking
        
        # --- DATASET OVERRIDES & PROTOCOL PATHS ---
        self.data_dir = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
        self.train_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_train_clean_coco.json"
        self.val_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_val_clean_coco.json"
        self.test_ann = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json"

        # --- NATIVE 16:9 RULE-OF-64 RESOLUTION ---
        self.input_size = (576, 1024)   # (Height, Width)
        self.test_size = (576, 1024)
        self.multiscale_range = 1

        # --- AUGMENTATION CURRICULUM ---
        self.mosaic_prob = 0.50
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.shear = 0.0
        self.translate = 0.1

        # --- TRAINING SCHEDULE ---
        self.max_epoch = 80        
        self.warmup_epochs = 10
        self.no_aug_epochs = 15
        self.ramp_epochs = 50
        self.prob_start = 0.10
        self.prob_end = 0.60

        self.min_lr_ratio = 0.05     
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 0.0005 
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1     
        self.ema = True            

        self.data_num_workers = 4
        self.batch_size = 16
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        """
        Quantized 16:9 Rule-of-64 multiscale resizing.
        Locks resolution to target (576, 1024) during the no_aug fine-tuning phase (final 15 epochs).
        """
        import random
        tensor = torch.LongTensor(2).cuda()
        if rank == 0:
            # Only apply multiscale jitter BEFORE the no_aug phase
            if self.multiscale_range > 0 and epoch < (self.max_epoch - self.no_aug_epochs):
                scale_choices = [(512, 896), (576, 1024), (640, 1152)]
                h, w = random.choice(scale_choices)
                tensor[0] = h
                tensor[1] = w
            else:
                # Lock to exact target resolution during final L1 box refinement
                tensor[0] = self.input_size[0]
                tensor[1] = self.input_size[1]

        if is_distributed:
            import torch.distributed as dist
            dist.barrier()
            dist.broadcast(tensor, 0)

        return (tensor[0].item(), tensor[1].item())

    def get_optimizer(self, batch_size):
        """
        FIX 1: Unfreezes all custom Parameters (gate, prototypes, ttt_lrs, mask_token).
        Replaces named_modules() with named_parameters().
        """
        if "optimizer" in self.__dict__:
            return self.optimizer

        lr = self.basic_lr_per_img * batch_size
        pg_decay, pg_no_decay = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Biases, 1D norm parameters, gates, and scalar meta-parameters get no weight decay
            if "bias" in name or p.ndim <= 1 or "norm" in name or "gn." in name or "gate" in name:
                pg_no_decay.append(p)
            else:
                pg_decay.append(p)

        optimizer_grouped_parameters = [
            {"params": pg_decay, "weight_decay": self.weight_decay, "lr": lr},
            {"params": pg_no_decay, "weight_decay": 0.0, "lr": lr},
        ]

        self.optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=lr,
            momentum=self.momentum,
            nesterov=True,
        )
        return self.optimizer

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="", 
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        target_ann = self.test_ann if testdev else self.val_ann
        
        return COCODataset(
            data_dir=self.data_dir,
            json_file=target_ann,
            name="", 
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        return torch.utils.data.DataLoader(
            valdataset,
            num_workers=0, # Prevents Windows shared memory crashes
            pin_memory=True,
            sampler=sampler,
            batch_size=batch_size,
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
            backbone = CSPDarknet(
                self.depth, 
                self.width, 
                depthwise=False, 
                act=self.act,
                ttt_lr=self.init_ttt_lr,           
                ttt_noise_std=self.ttt_noise_std,
                ttt_mask_ratio=self.ttt_mask_ratio
            )
            
            neck = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=[256, 512, 1024],
                act=self.act,
            )
            neck.backbone = backbone
            
            head = TDE_Head(
                self.num_classes, 
                self.width, 
                in_channels=[256, 512, 1024], 
                act=self.act
            )
            
            self.model = YOLOX(neck, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        
        # Bind Exp schedule parameters directly to the model
        self.model.max_epochs = self.max_epoch
        self.model.warmup_epochs = self.warmup_epochs
        self.model.ramp_epochs = self.ramp_epochs
        self.model.prob_start = self.prob_start
        self.model.prob_end = self.prob_end
        
        return self.model