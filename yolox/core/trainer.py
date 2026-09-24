#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.
# Integrated for TDE-YOLOX v3.1: Certified Multi-Engine Trainer & Synchronization Barrier

import datetime
import os
import time
from loguru import logger

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

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
    synchronize
)


class Trainer:
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

        # Data/dataloader related attributes
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0.0
        self.last_ap = None
        self.start_epoch = 0
        self.outputs = {}
        self._loaded_ema_state = None

        # Metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)
            temp_dir = os.path.join(self.file_name, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            os.environ["TEMP"] = temp_dir
            os.environ["TMP"] = temp_dir

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
            logger.error("Exception in training: ", e)
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
        
        # Asynchronous CUDA stream memory barrier guarantees data is committed to VRAM
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        is_amp = getattr(self.args, "fp16", False) or getattr(self.scaler, "_enabled", False)
        with torch.cuda.amp.autocast(enabled=is_amp):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # Multi-Engine Unscaling & Global Gradient Clipping
        if hasattr(self.optimizer, "optimizers"):
            for opt in self.optimizer.optimizers:
                self.scaler.unscale_(opt)
        else:
            self.scaler.unscale_(self.optimizer)

        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # All-or-Nothing Multi-Engine Step Execution
        scale_before = self.scaler.get_scale()
        if hasattr(self.optimizer, "optimizers"):
            for opt in self.optimizer.optimizers:
                self.scaler.step(opt)
        else:
            self.scaler.step(self.optimizer)

        self.scaler.update()
        scale_after = self.scaler.get_scale()

        # ModelEMA update strictly gated on successful optimizer step
        step_successful = (scale_after >= scale_before)
        if self.use_model_ema and step_successful:
            self.ema_model.update(self.model)
            # Post-EMA Riemannian Retraction onto unit hypersphere S^127
            with torch.no_grad():
                for m in self.ema_model.ema.modules():
                    if hasattr(m, "prototypes") and getattr(m.prototypes, "is_hypersphere", False):
                        m.prototypes.data.copy_(F.normalize(m.prototypes.data, p=2, dim=-1, eps=1e-5))

        # Update learning rates while respecting per-group lr_factor
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            factor = param_group.get("lr_factor", 1.0)
            param_group["lr"] = lr * factor

        iter_end_time = time.time()
        self.outputs = outputs
        
        metric_dict = {
            "iter_time": iter_end_time - iter_start_time,
            "data_time": data_end_time - iter_start_time,
            "lr": lr,
        }
        
        # Track all 10 active loss components
        active_metrics = [
            "total_loss", "iou_loss", "nwd_loss", "l1_loss", "conf_loss", 
            "cls_loss", "mem_loss", "proj_loss", "phase_loss", "idx_loss", 
            "rep_loss", "ttt_prob"
        ]
        for k, v in outputs.items():
            if k in active_metrics:
                if torch.is_tensor(v):
                    metric_dict[k] = v.detach().cpu().item()
                elif isinstance(v, (int, float)):
                    metric_dict[k] = float(v)
                
        self.meter.update(**metric_dict)

    def after_iter(self):
        if (self.iter + 1) % self.exp.print_interval == 0:
            curr_lr = self.optimizer.param_groups[0]['lr']
            left_iters = self.max_iter - self.iter - 1
            avg_time = self.meter["iter_time"].global_avg if "iter_time" in self.meter else 0
            eta_seconds = avg_time * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: [{}/{}][{}/{}]".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(["{}: {:.3f}".format(k, v.avg) for k, v in loss_meter.items()])

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str, gpu_mem_usage(), eta_str, loss_str, curr_lr
                )
            )

            if hasattr(self.meter, "clear_meters"):
                self.meter.clear_meters()
            else:
                self.meter.clear()

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar("train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({"train/lr": self.meter["lr"].latest})
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        model = self.resume_train(model)

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        # ModelEMA Initialization with Zero-Step Fresh Warmup Protection
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            if self._loaded_ema_state is not None:
                self.ema_model.ema.load_state_dict(self._loaded_ema_state)
                self.ema_model.updates = self.max_iter * self.start_epoch
                logger.info("Successfully restored ModelEMA shadow weights from checkpoint!")
            else:
                self.ema_model.updates = 0
                logger.info("No EMA state found; initialized fresh ModelEMA with zero-step warmup.")

        self.model = model
        self.evaluator = self.exp.get_evaluator(batch_size=self.args.batch_size, is_distributed=self.is_distributed)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(self.args, self.exp, self.evaluator.dataloader.dataset)
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)

        logger.info("Training start...")

    def after_train(self):
        logger.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
                self.mlflow_logger.on_train_end(self.args, file_name=self.file_name, metadata=metadata)

    def before_epoch(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "set_meta_training_state"):
            model.set_meta_training_state(self.epoch + 1, self.max_epoch)
            
        if self.use_model_ema and hasattr(self.ema_model.ema, "set_meta_training_state"):
            self.ema_model.ema.set_meta_training_state(self.epoch + 1, self.max_epoch)

        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
            
        # Persist latest checkpoint strictly after evaluation so self.last_ap is updated
        current_eval_ap = getattr(self, "last_ap", None)
        self.save_ckpt(ckpt_name="latest", ap=current_eval_ap)
        
        # Multi-GPU barrier prevents race conditions across epoch transitions
        synchronize()

    def before_iter(self):
        pass

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            ckpt_file = self.args.ckpt if self.args.ckpt else os.path.join(self.file_name, "latest_ckpt.pth")
            ckpt = torch.load(ckpt_file, map_location=self.device)
            
            # Gracefully handles parameter group restructuring
            clean_model_state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["model"].items()}
            model.load_state_dict(clean_model_state)
            
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                logger.info("Optimizer state successfully restored from checkpoint.")
            except Exception as e:
                logger.warning(
                    f"Optimizer parameter groups restructured: {e}. "
                    "Fresh momentum buffers initialized for updated parameter groups."
                )
            
            if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(self, "scaler"):
                try:
                    self.scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
                
            self.best_ap = ckpt.pop("best_ap", 0.0)
            raw_ema = ckpt.get("ema", None)
            self._loaded_ema_state = {k[7:] if k.startswith("module.") else k: v for k, v in raw_ema.items()} if raw_ema else None

            if self.args.start_epoch is not None:
                self.start_epoch = max(0, self.args.start_epoch - 1)
            elif "start_epoch" in ckpt:
                self.start_epoch = ckpt["start_epoch"]
            else:
                self.start_epoch = 0

            logger.info("loaded checkpoint '{}' (resuming from epoch {})".format(ckpt_file, self.start_epoch + 1))
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                raw_ckpt = torch.load(ckpt_file, map_location=self.device)
                
                if isinstance(raw_ckpt, dict):
                    ckpt_state = raw_ckpt.get("model", raw_ckpt)
                    raw_ema = raw_ckpt.get("ema", None)
                    self._loaded_ema_state = {k[7:] if k.startswith("module.") else k: v for k, v in raw_ema.items()} if raw_ema else None
                    self.best_ap = raw_ckpt.get("best_ap", 0.0)
                else:
                    ckpt_state = raw_ckpt

                model = load_ckpt(model, ckpt_state)
                
                # Fine-tuning from base weights always starts at Epoch 0!
                if self.args.start_epoch is not None:
                    self.start_epoch = max(0, self.args.start_epoch - 1)
                else:
                    self.start_epoch = 0
            else:
                self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model

        # Universal parallel model unwrapping for both live models and EMA models
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        self.last_ap = ap50_95

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            logger.info("Save weights to {}".format(self.file_name))
            
            raw_model_state = self.model.state_dict()
            clean_model_state = {k[7:] if k.startswith("module.") else k: v for k, v in raw_model_state.items()}
            
            if self.use_model_ema and self.ema_model.ema is not None:
                raw_ema_state = self.ema_model.ema.state_dict()
                clean_ema_state = {k[7:] if k.startswith("module.") else k: v for k, v in raw_ema_state.items()}
            else:
                clean_ema_state = None

            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "max_epoch": self.max_epoch,
                "model": clean_model_state,
                "ema": clean_ema_state,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler is not None else None,
                "best_ap": self.best_ap,
                "curr_ap": ap,
                "curr_iter": self.progress_in_iter,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )