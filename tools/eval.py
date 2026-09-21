#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: Selective High-Fidelity Fusion & Zero-Shot Adverse Evaluation

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    get_local_rank,
    get_model_info,
    setup_logger
)
from yolox.utils.checkpoint import load_ckpt


def fuse_conv_bn(conv, bn):
    """
    Fuses standard Convolution and BatchNorm2d weights mathematically in FP32.
    W_fused = W * (gamma / sqrt(var + eps))
    B_fused = (B - mean) * (gamma / sqrt(var + eps)) + beta
    """
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(device=conv.weight.device, dtype=conv.weight.dtype)

    w_conv = conv.weight.float().view(conv.out_channels, -1)
    scale = bn.weight.float().div(torch.sqrt(bn.eps + bn.running_var.float()))
    w_bn = torch.diag(scale)
    fused_weight = torch.mm(w_bn, w_conv).view(fused_conv.weight.shape).to(conv.weight.dtype)
    fused_conv.weight.copy_(fused_weight)

    b_conv = conv.bias.float() if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
    b_bn = bn.bias.float() - bn.weight.float().mul(bn.running_mean.float()).div(torch.sqrt(bn.running_var.float() + bn.eps))
    
    fused_bias = (torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).to(conv.weight.dtype)
    fused_conv.bias.copy_(fused_bias)

    return fused_conv


def fuse_tde_model(model):
    """
    SELECTIVE HIGH-FIDELITY FUSION:
    Protects TTTAdaptiveStage and all BaseConvGN layers from fusion.
    Redirects child.forward -> child.fuseforward to eliminate redundant Python nn.Identity dispatches.
    """
    from yolox.models.ttt_modules import TTTAdaptiveStage
    from yolox.models.network_blocks import BaseConv, BaseConvGN

    def recursive_fuse(module):
        if isinstance(module, TTTAdaptiveStage):
            logger.info("  -> Protected: TTTAdaptiveStage (Dark2 remains unfused)")
            return module

        for name, child in module.named_children():
            if isinstance(child, BaseConv):
                if isinstance(child.bn, nn.BatchNorm2d):
                    fused_conv = fuse_conv_bn(child.conv, child.bn)
                    child.conv = fused_conv
                    child.bn = nn.Identity()
                    child.forward = child.fuseforward  # Direct C++ execution!
            elif isinstance(child, BaseConvGN):
                continue
            else:
                recursive_fuse(child)
        return module

    model.eval()
    logger.info("Executing Selective TDE-YOLOX Fusion...")
    return recursive_fuse(model)


def make_parser():
    parser = argparse.ArgumentParser("TDE-YOLOX v3.1 Adverse Weather Evaluation")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-d", "--devices", default=1, type=int, help="device count")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--conf", default=0.001, type=float, help="test conf threshold")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true", help="evaluate in FP16")
    parser.add_argument("--fuse", dest="fuse", default=True, action="store_true", help="fuse conv and bn")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="evaluate in TensorRT")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true")
    parser.add_argument("--test", dest="test", default=False, action="store_true")
    parser.add_argument("--speed", dest="speed", default=False, action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Deterministic CUDNN setting activated.")

    is_distributed = num_gpu > 1
    rank = get_local_rank()
    file_name = os.path.join(exp.output_dir, args.experiment_name if args.experiment_name else exp.exp_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info(f"Evaluation Args: {args}")

    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()

    # FORCE ZERO-SHOT INFERENCE ADAPTATION
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        logger.info("TDE-YOLOX: Enforcing Meta-Inference state (TTT probability = 1.0, Tau = 0.50)...")
        m.set_meta_training_state(exp.max_epoch, exp.max_epoch)

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    # Load checkpoint with strict ModelEMA priority
    ckpt_file = args.ckpt if args.ckpt else os.path.join(file_name, "best_ckpt.pth")
    logger.info(f"Loading checkpoint from: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=f"cuda:{rank}")

    if isinstance(ckpt, dict):
        if "ema" in ckpt and ckpt["ema"] is not None:
            logger.info("Loading ModelEMA shadow weights for evaluation...")
            ckpt_state = ckpt["ema"]
        elif "model" in ckpt and ckpt["model"] is not None:
            logger.info("Loading standard model weights for evaluation...")
            ckpt_state = ckpt["model"]
        else:
            ckpt_state = ckpt
    else:
        ckpt_state = ckpt

    while isinstance(ckpt_state, dict) and "model" in ckpt_state and isinstance(ckpt_state["model"], dict):
        ckpt_state = ckpt_state["model"]

    model = load_ckpt(model, ckpt_state)

    # Selective fusion executed BEFORE DDP wrapping
    if args.fuse:
        model = fuse_tde_model(model)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.trt:
        assert not args.fuse and not is_distributed and args.batch_size == 1
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model is not found! Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    logger.info("Executing TDE-YOLOX v3.1 Benchmark Evaluation...")
    eval_fp16 = args.fp16 and not args.trt
    *_, summary = evaluator.evaluate(model, is_distributed, eval_fp16, trt_file, decoder, exp.test_size)
    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    launch(
        main,
        num_gpu,
        1,
        0,
        backend="nccl",
        dist_url="auto",
        args=(exp, args, num_gpu),
    )