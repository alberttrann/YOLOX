#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import gc
import math
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
    TDE-YOLOX Distributed Trainer (RAM-Fortified & Telemetry-Hardened vFinal).

    Core Refinements:
      - Fixed UnboundLocalError & Execution Flow in train_one_iter.
      - Fixed AttributeError in before_train (TelemetryRecorder initialized strictly after self.model).
      - Resumption State Fidelity: Preserves both raw model (matching optimizer states) and EMA weights.
      - Riemannian Tangent-Space Momentum Hook: Keeps prototype momentum vectors strictly on the manifold.
      - Progressive Phase Harmonization: Smooth Mosaic fade (Epochs 55-65) and linear L1 warmup (Epochs 60-65).
      - Dynamic EMA Decay: Annealed to 0.9950 across Epochs 65-70 to eliminate tracking lag.
      - RAM-Saving Severance: Immediate detached Python float conversion and explicit batch tensor disposal.
      - Pre-Validation Memory Purge: Clears prefetcher and collects garbage before validation to prevent OOM.
    """

    def __init__(self, exp: Exp, args):
        self.exp = exp
        self.args = args

        # Training related attributes
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16, init_scale=1024.0, growth_interval=2000)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # Data & precision attributes
        self.data_type = torch.float32  # AMP autocast handles convs; targets/coordinates stay FP32
        self.input_size = exp.input_size
        self.best_ap = 0.0

        # Telemetry & metric recording
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)
        self.outputs = {}
        self.successful_optimizer_steps = 0
        self.amp_skipped_steps = 0
        self.consecutive_amp_skips = 0
        self.max_consecutive_amp_skips = 12

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

    def _project_prototype_momentum(self, model_ref):
        """
        Riemannian Tangent-Space Momentum Projection Hook.
        Projects SGD velocity buffers of prototypes onto the spherical tangent plane:
        v_tangent = v - (v . P) * P
        Prevents radial momentum chatter and maintains stable geodesic optimization.
        """
        if not hasattr(model_ref, "head") or not hasattr(model_ref.head, "memory_banks"):
            return

        for bank in model_ref.head.memory_banks:
            p = bank.prototypes  # [K, M, D]
            state = self.optimizer.state.get(p, None)
            if state is not None and "momentum_buffer" in state:
                v = state["momentum_buffer"]  # [K, M, D]
                v_dot_p = (v * p.data).sum(dim=-1, keepdim=True)
                v.sub_(v_dot_p * p.data)

    def train_one_iter(self):
        iter_start_time = time.time()

        # 1. Fetch next batch from prefetcher
        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        if inps.ndim != 4 or inps.shape[1] != 4:
            raise RuntimeError('PG-SMA training expected 4-channel RGB+validity batch')
        valid_image_mask = inps[:, 3:4] > 0.5
        inps = inps[:, :3].contiguous()
        data_end_time = time.time()

        # 2. Forward pass under mixed precision
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, valid_image_mask=valid_image_mask)

        loss = outputs["total_loss"]

        # Semantic objective uses the same native feature and weight snapshot.
        # It has an independent optimizer, FP32 backward, gradient clipping and
        # nonfinite step gate; semantic failure cannot skip the native update.
        model_ref = self.model.module if self.is_distributed else self.model
        from .pgsma_step import semantic_optimizer_step
        if torch.isfinite(outputs["total_loss"]).all() and self.exp.pgsma_variant != 'native':
            with torch.cuda.amp.autocast(enabled=False):
                semantic_success, semantic_metrics = semantic_optimizer_step(
                    model_ref, self.semantic_optimizer,
                    seed=1000003 * (self.progress_in_iter + 1) + 17,
                    correction_mode=self.exp.pgsma_variant)
            if not semantic_success:
                logger.warning("PG-SMA independent optimizer skip: {}", semantic_metrics)
            for k,v in semantic_metrics.items():
                if isinstance(v,(float,int)):
                    outputs['pg_'+k] = outputs['total_loss'].new_tensor(v)

        # 3. Numerical Stability Guard with Component Diagnostics and VRAM Purge
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                "Numerical instability: total_loss is NaN/Inf! "
                "total: {:.3f}, iou: {}, conf: {}, cls: {}, mem: {}, proto: {}, rec: {}, afford: {}".format(
                    loss.item() if not torch.isnan(loss) else float('nan'),
                    outputs.get('iou_loss', 'N/A'),
                    outputs.get('conf_loss', 'N/A'),
                    outputs.get('cls_loss', 'N/A'),
                    outputs.get('mem_loss', 'N/A'),
                    outputs.get('proto_loss', 'N/A'),
                    outputs.get('rec_loss', 'N/A'),
                    outputs.get('afford_loss', 'N/A'),
                )
            )
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite forward loss; optimizer update forbidden")

        # Semantic gradients belong to the semantic optimizer, whose step already
        # completed above. Clear them before the native backward and telemetry.
        # Native gradient clipping must NEVER include semantic parameters.
        self.semantic_optimizer.zero_grad(set_to_none=True)

        # 4. Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # 5. Gradient Unscaling & Telemetry Hook (Pre-step gradient flow logging)
        self.scaler.unscale_(self.optimizer)
        if hasattr(self, "telemetry"):
            self.telemetry.record_gradient_flow(self.epoch + 1, self.progress_in_iter + 1)

        # 6. Distinguish recoverable AMP LOSS-SCALE overflow from persistent
        # FP32 backward failure. Never scrub or step with invalid gradients.
        # Parameter ownership is defined by the native optimizer's param groups,
        # not by model.parameters(): PG-SMA parameters are in the same module.
        native_params = [p for group in self.optimizer.param_groups
                         for p in group['params'] if p.requires_grad]
        native_ids = {id(p) for p in native_params}
        named_grads = [(n, p.grad) for n, p in self.model.named_parameters()
                       if id(p) in native_ids and p.grad is not None]
        invalid_names = [n for n, g in named_grads if not torch.isfinite(g).all().item()]
        if invalid_names and not self.amp_training:
            logger.error("FP32 backward invalid at epoch={} iter={}: {}",
                         self.epoch + 1, self.iter + 1, invalid_names[:25])
            raise FloatingPointError("Persistent FP32 backward failure")
        if not invalid_names:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                native_params, max_norm=10.0, error_if_nonfinite=True
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Total gradient norm non-finite despite finite individual gradients")

        snapshot = self.telemetry.snapshot_parameter_updates(self.progress_in_iter + 1) if hasattr(self, "telemetry") else None
        # GradScaler recorded found_inf during unscale_. The scaler handles the
        # optimizer skip and halves the loss scale, without modifying parameters.
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        new_scale = self.scaler.get_scale()
        did_step = (not self.amp_training) or (new_scale >= old_scale)
        if invalid_names and did_step:
            raise RuntimeError("Invalid gradients but scaler did not skip optimizer update")
        if not did_step:
            self.amp_skipped_steps += 1
            self.consecutive_amp_skips += 1
            logger.warning("AMP overflow/backoff epoch={} iter={} scale {:.0f}->{:.0f}, offenders={}, consecutive={}",
                           self.epoch + 1, self.iter + 1, old_scale, new_scale,
                           invalid_names[:12], self.consecutive_amp_skips)
            if self.consecutive_amp_skips >= self.max_consecutive_amp_skips:
                raise FloatingPointError("Repeated AMP overflows; run FP32/anomaly audit")
        else:
            self.consecutive_amp_skips = 0

        # 8-9. Projection and EMA ONLY after a successful optimizer update.
        if did_step:
            self.successful_optimizer_steps += 1
            model_ref = self.model.module if self.is_distributed else self.model
            if hasattr(model_ref, "project_prototypes_to_unit_sphere"):
                model_ref.project_prototypes_to_unit_sphere()
            self._project_prototype_momentum(model_ref)
            if self.use_model_ema:
                self.ema_model.update(self.model)
        else:
            pass  # recorded above with detailed AMP overflow diagnostics

        if hasattr(self, "telemetry"):
            self.telemetry.record_parameter_updates(
                self.epoch + 1, self.progress_in_iter + 1, snapshot, did_step
            )

        # 10. Learning Rate Schedule Update
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        base_lr = self.exp.basic_lr_per_img * self.args.batch_size
        semantic_factor = lr / max(base_lr, 1e-12)
        for param_group in self.semantic_optimizer.param_groups:
            param_group['lr'] = self.exp.semantic_lr * semantic_factor

        iter_end_time = time.time()

        # 11. Detached Scalar Telemetry (Instant memory severance)
        self.outputs = {
            k: (v.detach().cpu().item() if torch.is_tensor(v) else float(v))
            for k, v in outputs.items()
        }

        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
        )

        # 12. Offline Structured JSONL Telemetry Logging
        if hasattr(self, "telemetry"):
            self.telemetry.record_iteration_losses(
                self.epoch + 1, self.progress_in_iter + 1, lr, self.outputs
            )
            if "num_fg" in self.outputs:
                self.telemetry.record_simota_assignment(
                    self.epoch + 1, self.progress_in_iter + 1,
                    self.outputs["num_fg_total"], num_gt=self.outputs["num_gt_total"]
                )

        # Explicit RAM reclamation
        del inps, targets, valid_image_mask, outputs, loss

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
        if self.rank == 0:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("PARAMETER INVENTORY: registered={} trainable={} (disabled branches may still be registered)", total, trainable)
            for prefix in ("backbone.backbone", "backbone.spi_highway", "backbone.C3_", "head.memory_", "head.latent_", "affordance_head"):
                count = sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
                logger.info("PARAMETER BUCKET prefix={} count={}", prefix, count)

        # 2. Solver initialization. Native and semantic optimizers are disjoint.
        if not getattr(self.exp, 'allow_full_training', False):
            raise RuntimeError('vNext.3 full training is preflight-blocked. Run verification suite '
                               'and repair upstream YOLOX data collation before enabling it.')
        if self.is_distributed:
            raise NotImplementedError('PG-SMA semantic DDP gradient synchronization has not been '
                                      'verified. Use -d 1 for integration tests.')
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        self.semantic_optimizer = torch.optim.AdamW(model.semantic_parameters(),
            lr=float(self.exp.semantic_lr), weight_decay=0.0)

        # 3. Checkpoint resumption
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
            # Feature flags create intentionally unused parameters during ablations.
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False,
                        find_unused_parameters=True)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch
            # Restore separate EMA weights if resuming from checkpoint
            if hasattr(self, "_resume_ema_state_dict") and self._resume_ema_state_dict is not None:
                self.ema_model.ema.load_state_dict(self._resume_ema_state_dict)
                logger.info("Restored separate EMA model state from checkpoint.")

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        # 5. Telemetry Engine Initialization (Rank 0 only, zero terminal clutter)
        # Initialized strictly AFTER self.model and self.evaluator exist
        from yolox.utils.telemetry import TelemetryRecorder
        self.telemetry = TelemetryRecorder(
            save_dir=self.file_name,
            model=self.model,
            record_interval=self.exp.print_interval,
            rank=self.rank,
        )

        # 6. External loggers setup
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
        # Flush and close all file pointers
        if hasattr(self, "telemetry"):
            self.telemetry.close()

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
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        # Synchronize epoch state across model, head, and dark2
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, "set_meta_training_state"):
            model_ref.set_meta_training_state(self.epoch + 1, self.max_epoch)

        fade_start_epoch = self.max_epoch - self.exp.no_aug_epochs - 10  # Epoch 55
        fade_end_epoch = self.max_epoch - self.exp.no_aug_epochs        # Epoch 65

        # Progressive Mosaic Annealing (Epochs 55-65)
        if self.epoch + 1 >= fade_end_epoch or self.no_aug:
            if not self.no_aug:
                logger.info("---> Fully closing Mosaic augmentation and setting native resolution.")
                self.train_loader.close_mosaic()
                self.no_aug = True
                self.exp.eval_interval = 1
                self.save_ckpt(ckpt_name="last_mosaic_epoch")
        elif self.epoch + 1 >= fade_start_epoch:
            progress = (self.epoch + 1 - fade_start_epoch) / 10.0
            current_mosaic_prob = max(0.0, self.exp.mosaic_prob * (1.0 - progress))
            if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "mosaic_prob"):
                self.train_loader.dataset.mosaic_prob = current_mosaic_prob
            logger.info(f"---> Mosaic probability linearly annealed to {current_mosaic_prob:.3f}")

        # Progressive L1 Loss Ramp (Epochs 60-65)
        l1_start_epoch = self.max_epoch - self.exp.no_aug_epochs - 5  # Epoch 60
        if self.epoch + 1 >= l1_start_epoch:
            l1_weight = min(1.0, (self.epoch + 1 - l1_start_epoch) / 5.0)
            head_ref = self.model.module.head if self.is_distributed else self.model.head
            head_ref.use_l1 = True
            head_ref.l1_weight = l1_weight
            logger.info(f"---> Progressive L1 loss active with weight: {l1_weight:.2f}")

        # Dynamic EMA Momentum Annealing across post-Mosaic transition (Epochs 65-70)
        if self.use_model_ema:
            if self.epoch + 1 == fade_end_epoch:
                self.ema_model.decay = 0.9950
                logger.info("---> Annealing EMA decay to 0.9950 for rapid post-mosaic synchronization.")
            elif self.epoch + 1 == fade_end_epoch + 5:
                self.ema_model.decay = 0.9990
                logger.info("---> Restoring EMA decay to 0.9990 for final precision convergence.")

        # RAM REFINEMENT: Explicitly delete previous prefetcher and clean garbage before instantiating
        if hasattr(self, "prefetcher"):
            del self.prefetcher
        gc.collect()
        torch.cuda.empty_cache()

        self.prefetcher = DataPrefetcher(self.train_loader)

    def after_epoch(self):
        # Record epoch-level prototype manifold geometry
        if hasattr(self, "telemetry"):
            self.telemetry.record_prototype_geometry(self.epoch + 1)

        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            # RAM REFINEMENT: Pre-Validation Memory Purge
            if hasattr(self, "prefetcher"):
                del self.prefetcher
            gc.collect()
            torch.cuda.empty_cache()

            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

        # Post-epoch memory reclamation
        gc.collect()
        torch.cuda.empty_cache()

    def before_iter(self):
        pass

    def after_iter(self):
        tracked_keys = [
            "total_loss",
            "iou_loss",
            "l1_loss",
            "conf_loss",
            "cls_loss",
            "mem_loss",
            "proto_loss",
            "intra_loss",
            "rec_loss",
            "afford_loss",
            "ttt_prob",
            "det_loss",
            "aux_total",
            "aux_factor",
        ]
        for k, v in self.outputs.items():
            if k in tracked_keys:
                self.meter.update(**{k: v})

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

            # RAM REFINEMENT: Periodic cyclic garbage collection on Windows to flush worker IPC buffers
            gc.collect()

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
        Checkpoints store raw model weights matching optimizer momentum states,
        and store EMA weights under 'ema_model'.
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
            if not ckpt.get('semantic_optimizer'):
                raise RuntimeError('Cannot resume vNext.3 without semantic optimizer state')
            if ckpt.get('pgsma_variant') != self.exp.pgsma_variant:
                raise RuntimeError('Cannot resume with changed PG-SMA variant')
            self.semantic_optimizer.load_state_dict(ckpt['semantic_optimizer'])
            self.best_ap = ckpt.get("best_ap", 0.0)
            if ckpt.get("scaler") is not None:
                self.scaler.load_state_dict(ckpt["scaler"])
            self.successful_optimizer_steps = ckpt.get("successful_optimizer_steps", 0)
            self.start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt.get("start_epoch", 0)
            )
            # Retain EMA dictionary for restoration after ModelEMA is created
            self._resume_ema_state_dict = ckpt.get("ema_model", None)
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

                # Safe Key Remapping: maps dark2 convolution filters to TTTAdaptiveStage
                # Completely bypasses scalar num_batches_tracked to eliminate concatenation crashes
                remapped_state_dict = {}
                for k, v in clean_state_dict.items():
                    if "dark2." in k and "backbone_stage" not in k:
                        new_k = k.replace("dark2.", "dark2.backbone_stage.")
                        remapped_state_dict[new_k] = v
                    remapped_state_dict[k] = v

                # Report actual loaded elements (not simply missing custom keys).
                current = model.state_dict()
                compatible = {k for k, v in remapped_state_dict.items()
                              if k in current and tuple(v.shape) == tuple(current[k].shape)}
                matched = sum(current[k].numel() for k in compatible)
                all_elements = sum(v.numel() for v in current.values())
                logger.info("CHECKPOINT COVERAGE matched={}/{} elements ({:.2%}) keys={}/{}",
                            matched, all_elements, matched / max(1, all_elements), len(compatible), len(current))
                for critical in ("backbone.backbone.stem", "backbone.backbone.dark2.backbone_stage",
                                 "backbone.backbone.dark3", "backbone.backbone.dark4", "backbone.backbone.dark5",
                                 "backbone.C3_p4", "backbone.C3_p3", "backbone.C3_n3", "backbone.C3_n4",
                                 "head.stems", "head.reg_convs", "head.obj_preds"):
                    part = {k for k in current if k.startswith(critical)}
                    part_match = part & compatible
                    logger.info("CHECKPOINT COVERAGE {}: {}/{} keys", critical, len(part_match), len(part))
                model = load_ckpt(model, remapped_state_dict)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        # Enforce full adaptation during validation
        if hasattr(evalmodel, "set_meta_training_state"):
            evalmodel.set_meta_training_state(self.epoch + 1, self.max_epoch)

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
        """
        Saves raw model weights matching optimizer momentum states,
        and separately preserves EMA weights to eliminate state-dict desynchronization.
        """
        if self.rank == 0:
            raw_model = self.model.module if is_parallel(self.model) else self.model
            logger.info("Saving weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": raw_model.state_dict(),  # Raw model matching optimizer states!
                "ema_model": self.ema_model.ema.state_dict() if self.use_model_ema else None,
                "optimizer": self.optimizer.state_dict(),
                "semantic_optimizer": self.semantic_optimizer.state_dict(),
                "semantic_step": int(raw_model.semantic.semantic_step.item()),
                "pgsma_variant": self.exp.pgsma_variant,
                "best_ap": self.best_ap,
                "curr_ap": ap,
                "scaler": self.scaler.state_dict(),
                "successful_optimizer_steps": self.successful_optimizer_steps,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
