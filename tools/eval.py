#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    get_local_rank,
    get_model_info,
    setup_logger,
)


def fuse_conv_bn(conv, bn):
    """
    Mathematically folds Conv2d and BatchNorm2d weights into a single Conv2d layer.
    Pure PyTorch tensor implementation (zero NumPy syntax, zero AttributeError).
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
    ).requires_grad_(False).to(conv.weight.device)

    # 1. Weight Fusion: W_fused = W_conv * (gamma / sqrt(var + eps))
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    scale_std = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    w_bn = torch.diag(scale_std)
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    # 2. Bias Fusion: B_fused = (B_conv - mean) * (gamma / sqrt(var + eps)) + beta
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels).to(conv.weight.device)

    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    fused_conv.bias.copy_(fused_bias)

    return fused_conv


def fuse_tde_model(model):
    """
    High-Fidelity Selective Fusion for TDE-YOLOX.
    
    Protects:
      - TTTAdaptiveStage (dark2): GroupNorm parameters must remain accessible for instance TTT.
      - BaseConvGN / DWConvGN: GroupNorm depends on dynamic instance statistics and cannot be statically folded.
    Fuses:
      - All standard BaseConv (Conv2d + BatchNorm2d) across deep backbone and neck layers.
    """
    from yolox.models.ttt_modules import TTTAdaptiveStage
    from yolox.models.network_blocks import BaseConv, BaseConvGN

    def recursive_fuse(module):
        for name, child in module.named_children():
            # 1. Protect Meta-Adaptive Stage 1
            if isinstance(child, TTTAdaptiveStage):
                logger.info("  -> Protected: TTTAdaptiveStage (dark2)")
                continue

            # 2. Skip GroupNorm convolution primitives
            if isinstance(child, BaseConvGN):
                continue

            # 3. Fuse standard BatchNorm convolutions
            if isinstance(child, BaseConv) and isinstance(child.bn, nn.BatchNorm2d):
                fused_conv = fuse_conv_bn(child.conv, child.bn)
                child.conv = fused_conv
                child.bn = nn.Identity()
            else:
                recursive_fuse(child)
        return module

    model.eval()
    logger.info("Executing High-Fidelity TDE-YOLOX Selective Fusion...")
    return recursive_fuse(model)


def make_parser():
    parser = argparse.ArgumentParser("TDE-YOLOX Evaluation Harness")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name")

    parser.add_argument("--dist-backend", default="nccl", type=str, help="Distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="Distributed initialization URL")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="Number of GPU devices")
    parser.add_argument("--num_machines", default=1, type=int, help="Number of nodes")
    parser.add_argument("--machine_rank", default=0, type=int, help="Node rank")

    parser.add_argument("-f", "--exp_file", default=None, type=str, help="Experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="Path to checkpoint weights")
    parser.add_argument("--conf", default=None, type=float, help="Confidence threshold")
    parser.add_argument("--nms", default=None, type=float, help="NMS IoU threshold")
    parser.add_argument("--tsize", default=None, type=int, help="Evaluation image size override")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    parser.add_argument("--fp16", default=False, action="store_true", help="Adopt half-precision evaluation")
    parser.add_argument("--fuse", default=False, action="store_true", help="Perform selective layer fusion")
    parser.add_argument("--trt", default=False, action="store_true", help="Deploy TensorRT model")
    parser.add_argument("--legacy", default=False, action="store_true", help="Maintain legacy data transform")
    parser.add_argument("--test", default=False, action="store_true", help="Evaluate against test-dev split")
    parser.add_argument("--speed", default=False, action="store_true", help="Execute speed profiling only")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Command-line option overrides")
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Deterministic CUDA seeding is enabled.")

    is_distributed = num_gpu > 1
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()
    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Evaluation Arguments: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 1. Instantiate Model Architecture
    model = exp.get_model()

    # 2. Enforce Deterministic Full-Adaptation Mode for Zero-Shot OOD Testing
    model_ref = model.module if hasattr(model, "module") else model
    if hasattr(model_ref, "set_meta_training_state"):
        logger.info("TDE-YOLOX: Enforcing Test-Time Adaptation Mode (ttt_prob = 1.0)...")
        model_ref.set_meta_training_state(exp.max_epoch, exp.max_epoch)

    # 3. Trigger Structural Re-Parameterization Deployment Folding (RepDWConv -> 3x3 DWConv)
    if hasattr(model_ref, "backbone") and hasattr(model_ref.backbone, "switch_to_deploy"):
        logger.info("TDE-YOLOX: Folding RepDWConv branches into single 3x3 convs for deployment...")
        model_ref.backbone.switch_to_deploy()

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # 4. Setup COCO Evaluator with Per-Class Metrics Output
    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    # 5. Load Checkpoint Weights
    if not args.speed and not args.trt:
        ckpt_file = args.ckpt if args.ckpt is not None else os.path.join(file_name, "best_ckpt.pth")
        logger.info("Loading evaluation checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        clean_state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_state_dict)
        logger.info("Loaded checkpoint weights successfully.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # 6. Execute Selective Fusion
    if args.fuse:
        model = fuse_tde_model(model)

    if args.trt:
        assert not args.fuse and not is_distributed and args.batch_size == 1, (
            "TensorRT deployment requires non-distributed, batch_size=1 execution without fusion."
        )
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), f"TensorRT engine {trt_file} not found. Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # 7. Execute Evaluation
    logger.info("Starting TDE-YOLOX High-Fidelity OOD Evaluation...")
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )