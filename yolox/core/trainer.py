#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import gc
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    MlflowLogger,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
)


class Trainer:
    """
    TDE-YOLOX Meta-Learning Distributed Trainer (Surgical vNext).
    
    Surgical Fixes Applied:
      - Patch 5.1: Automatically remaps checkpoint keys dark2.X -> dark2.backbone_stage.X to 
                   load pre-trained COCO edge filters into the GroupNorm-isolated Stage 1.
      - Full Telemetry Tracking: Unpacks and logs afford_loss, mem_loss, proto_loss, and ttt_prob.
      - Caveat 14.3 Resolution: Single-point meta-training state synchronization in before_epoch.
      - Numerical Stability Guard: Zeroes gradients and skips batch if loss evaluates to NaN/Inf.
      - Host RAM Reclamation: gc.collect() and empty_cache() after each epoch prevent Windows OOM.
    """

    def __init__(self, exp: Exp, args):
        self.exp = exp
        self.args = args

        # Training related attributes
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # Data & precision attributes
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0.0

        # Telemetry & metric recording
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        self.outputs = {}

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error("Exception in training loop: {}", e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        # NaN Guard: Catch numerical divergence before it corrupts optimizer states
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error("Numerical instability: total_loss is NaN/Inf. Skipping gradient step.")
            self.optimizer.zero_grad()
            return

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()

        # Retain outputs for telemetry logging in after_iter
        self.outputs = outputs
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # 1. Hardware initialization
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # 2. Solver initialization
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # 3. Checkpoint resumption & Pre-trained weight loading (with Key Remapping)
        model = self.resume_train(model)

        # 4. Data pipeline initialization
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("Initializing prefetcher...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )

        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        # 5. External loggers setup
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args, self.exp, self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("Logger must be either 'tensorboard', 'mlflow', or 'wandb'")

        logger.info("Training initialization completed.")

    def after_train(self):
        logger.info(
            "Training completed. Best AP: {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    "start_ckpt": self.args.ckpt,
                    "exp_file": self.args.exp_file,
                    "best_ap": float(self.best_ap),
                }
                self.mlflow_logger.on_train_end(
                    self.args, file_name=self.file_name, metadata=metadata
                )

    def before_epoch(self):
        """
        CAVEAT 14.3 RESOLUTION:
        Consolidated Single-Point Meta-Training Epoch Synchronization.
        Executes exactly once per epoch with no duplicate state calls.
        """
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        # Unwrap DDP if necessary to reach the base YOLOX meta-learning wrapper
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, "set_meta_training_state"):
            model_ref.set_meta_training_state(self.epoch + 1, self.max_epoch)

        # Check Mosaic shutoff condition
        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("---> Closing Mosaic augmentation and enabling L1 loss.")
            self.train_loader.close_mosaic()
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

        self.prefetcher = DataPrefetcher(self.train_loader)

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

        # HOST RAM RECLAMATION (Prevents Windows DataLoader MemoryError)
        gc.collect()
        torch.cuda.empty_cache()

    def before_iter(self):
        pass

    def after_iter(self):
        """Telemetry logging: tracks afford_loss alongside standard and meta metrics."""
        tracked_keys = [
            "total_loss",
            "iou_loss",
            "l1_loss",
            "conf_loss",
            "cls_loss",
            "mem_loss",
            "proto_loss",
            "afford_loss",
            "ttt_prob",
        ]
        for k, v in self.outputs.items():
            if k in tracked_keys:
                if torch.is_tensor(v):
                    val = v.detach().cpu().item()
                else:
                    try:
                        val = float(v)
                    except (ValueError, TypeError):
                        continue
                self.meter.update(**{k: val})

        if (self.iter + 1) % self.exp.print_interval == 0:
            curr_lr = self.optimizer.param_groups[0]["lr"]
            left_iters = self.max_iter * self.max_epoch - self.progress_in_iter
            avg_time = self.meter["iter_time"].global_avg if "iter_time" in self.meter else 0.0
            eta_seconds = avg_time * left_iters
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: [{}/{}][{}/{}]".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )

            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.avg) for k, v in loss_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    eta_str,
                    loss_str,
                    curr_lr,
                )
            )

            if self.rank == 0 and hasattr(self, "tblogger"):
                self.tblogger.add_scalar("train/lr", curr_lr, self.progress_in_iter)
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar(f"train/{k}", v.latest, self.progress_in_iter)

            if hasattr(self.meter, "clear_meters"):
                self.meter.clear_meters()
            else:
                self.meter.clear()

        # Dynamic Rule-of-64 Multiscale Resolution Selection
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        """
        Robust checkpoint and pre-trained weight loader.
        
        PATCH 5.1 APPLIED:
        Remaps checkpoint keys dark2.X -> dark2.backbone_stage.X to recover
        pre-trained COCO convolution weights into GroupNorm Stage 1.
        """
        if self.args.resume:
            logger.info("Resuming training from checkpoint...")
            ckpt_file = (
                self.args.ckpt
                if self.args.ckpt is not None
                else os.path.join(self.file_name, "latest_ckpt.pth")
            )

            ckpt = torch.load(ckpt_file, map_location=self.device)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.get("best_ap", 0.0)
            self.start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt.get("start_epoch", 0)
            )
            logger.info(
                "Resumed checkpoint '{}' at epoch {}".format(ckpt_file, self.start_epoch)
            )
        else:
            if self.args.ckpt is not None:
                logger.info("Loading pre-trained checkpoint for fine-tuning...")
                ckpt_file = self.args.ckpt
                raw_ckpt = torch.load(ckpt_file, map_location=self.device)
                state_dict = raw_ckpt["model"] if "model" in raw_ckpt else raw_ckpt
                
                clean_state_dict = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }

                # PATCH 5.1 APPLIED: Remap dark2 convolution keys to match TTTAdaptiveStage namespace
                remapped_state_dict = {}
                for k, v in clean_state_dict.items():
                    if "dark2." in k and "backbone_stage" not in k and ".bn." not in k:
                        new_k = k.replace("dark2.", "dark2.backbone_stage.")
                        remapped_state_dict[new_k] = v
                    remapped_state_dict[k] = v

                model = load_ckpt(model, remapped_state_dict)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        # Enforce full adaptation during validation
        if hasattr(evalmodel, "set_meta_training_state"):
            evalmodel.set_meta_training_state(self.max_epoch, self.max_epoch)

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if hasattr(self, "tblogger"):
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/best_ap": round(self.best_ap, 3),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)
            logger.info("\n" + summary)

        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Saving weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )