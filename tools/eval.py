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

import torch
import torch.nn as nn
from loguru import logger

def fuse_conv_bn(conv, bn):
    """
    Fuses Convolution and BatchNorm2d weights mathematically.
    Formula: W_fused = W * (gamma / sqrt(var + eps))
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
    ).requires_grad_(False).to(conv.weight.device)

    # Prepare parameters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    
    # Calculate fused weights
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))
    
    # Calculate fused bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels).to(conv.weight.device)
        
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    
    # Corrected Bias Fusion Logic:
    # (b_conv * scale) + b_bn
    # Note: w_bn is diagonal scale factor.
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    fused_conv.bias.copy_(fused_bias)

    return fused_conv

def fuse_tde_model(model):
    """
    SUPREME SELECTIVE FUSION: 
    Fuses high-compute towers while protecting Meta-Learning parameters.
    """
    from yolox.models.ttt_modules import TTTAdaptiveStage
    from yolox.models.network_blocks import BaseConv, BaseConvGN

    def recursive_fuse(module):
        # 1. PROTECT TTT STAGE: Adaptation requires original GN layers
        if isinstance(module, TTTAdaptiveStage):
            logger.info("  -> Protected: TTTAdaptiveStage (Dark2)")
            return module

        # 2. SELECTIVE FUSION: Identify fuseable Conv-BN pairs
        for name, child in module.named_children():
            if isinstance(child, BaseConv):
                # Standard YOLOX BaseConv uses BN. We can fuse this.
                if isinstance(child.bn, nn.BatchNorm2d):
                    # Perform Mathematical Fusion
                    fused_conv = fuse_conv_bn(child.conv, child.bn)
                    # Replace Conv+BN with FusedConv
                    child.conv = fused_conv
                    # Remove BN by replacing with Identity
                    child.bn = nn.Identity()
                    # Update forward logic via the module's fuseforward method
            
            elif isinstance(child, BaseConvGN):
                # BaseConvGN uses GroupNorm. 
                # GN depends on current instance stats, so it CANNOT be fused statically.
                # skip to preserve mathematical accuracy.
                continue
                
            else:
                # Recursively descend into Bottlenecks, CSPLayers, PAFPN, and Head
                recursive_fuse(child)
        return module

    model.eval()
    logger.info("Executing High-Fidelity TDE-YOLOX Fusion...")
    fused_model = recursive_fuse(model)
    return fused_model


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    from yolox.utils import configure_nccl
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 1. Instantiate the Model
    model = exp.get_model()
    
    # --- TRIGGER FULL ADAPTATION MODE ---
    # unwrap the model to reach the TDE-YOLOX specific methods
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        # set current_epoch to max_epoch to ensure ttt_prob = 1.0 (End of ramp)
        # guarantees every image in the Adverse test set is adapted.
        logger.info("TDE-YOLOX: Setting Meta-Inference state (Probability 1.0)...")
        m.set_meta_training_state(exp.max_epoch, exp.max_epoch)
    
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # 2. Setup Evaluator with Per-Class Metrics
    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    # 3. Load Research Weights
    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])
    
    # 4. SELECTIVE HIGH-FIDELITY FUSION
    if args.fuse:
        # call custom fusion logic instead of standard YOLOX logic
        # protects the Adaptive Norms and Engram weights
        model = fuse_tde_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # 5. START OOD EVALUATION
    logger.info("Starting TDE-YOLOX OOD Forensic Evaluation...")
    # The TTT loop runs internally inside Dark2 if run_ttt is enabled
    # The Engram head restores identities in the Classification branch
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    
    # Log the full results including the Class AP/AR table 
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