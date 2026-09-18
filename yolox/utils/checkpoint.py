#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: High-Fidelity Pretrained Key Remapping & Zero-AttnRes Guard

import os
import shutil
from loguru import logger
import torch


def load_ckpt(model, ckpt):
    """
    TDE-YOLOX v3.1 High-Fidelity Pretrained Checkpoint Loader.
    Resolves Defect U1 (Dark2 Random Initialization) and Defect L3 (Stem GN Omission).
    
    Remapping Schema:
      - backbone.stem.conv.bn.* -> backbone.stem.conv.gn.*
      - backbone.dark2.*        -> backbone.dark2.backbone_stage.*
      - bn.*                    -> gn.* (in all Stage 1 layers)
      - Drops all running_mean and running_var buffers for GroupNorm
      - Strictly excludes Scale-AttnRes pseudo-queries to maintain w_l = 0.0
    """
    model_state_dict = model.state_dict()
    load_dict = {}

    for key_model, param_model in model_state_dict.items():
        # EXCLUSION RULE 1: Never overwrite Scale-AttnRes pseudo-queries (Kimi §5 Zero-Init Invariant)
        if "attnres" in key_model and ".w" in key_model:
            logger.info(f"  -> Preserved zero-initialization for Scale-AttnRes query: {key_model}")
            continue

        # EXCLUSION RULE 2: Keep new custom layers at their mathematically derived initializations
        if any(custom in key_model for custom in ["memory_banks", "anchor_projectors", "manifold_decoders", "post_engram_dw", "p5_dense_attn", "p4_bqsa", "p3_bqsa", "projector", "ttt_lrs"]):
            if key_model not in ckpt:
                continue

        # 1. Exact Key Match
        if key_model in ckpt:
            if param_model.shape == ckpt[key_model].shape:
                load_dict[key_model] = ckpt[key_model]
            else:
                logger.warning(f"Shape mismatch for {key_model}: {param_model.shape} vs {ckpt[key_model].shape}. Skipping.")
            continue

        # 2. Stage 0 Stem Remapping: Focus Layer (bn -> gn)
        key_stem_remapped = key_model.replace("conv.gn.", "conv.bn.")
        if key_stem_remapped in ckpt:
            if param_model.shape == ckpt[key_stem_remapped].shape:
                load_dict[key_model] = ckpt[key_stem_remapped]
                logger.info(f"  -> Remapped Stem: {key_stem_remapped} -> {key_model}")
                continue

        # 3. Stage 1 Remapping: dark2 -> dark2.backbone_stage (and bn -> gn)
        # Handle prefix nesting difference in TTTAdaptiveStage
        key_dark2_cand = key_model.replace("dark2.backbone_stage.", "dark2.")
        key_dark2_cand = key_dark2_cand.replace(".gn.", ".bn.")
        if key_dark2_cand in ckpt:
            if param_model.shape == ckpt[key_dark2_cand].shape:
                load_dict[key_model] = ckpt[key_dark2_cand]
                logger.info(f"  -> Remapped Stage 1: {key_dark2_cand} -> {key_model}")
                continue

        # Handle YOLOPAFPN wrapping prefix variations (backbone.backbone vs backbone)
        key_alt = key_model.replace("backbone.backbone.", "backbone.")
        key_alt_dark2 = key_alt.replace("dark2.backbone_stage.", "dark2.").replace(".gn.", ".bn.")
        if key_alt_dark2 in ckpt:
            if param_model.shape == ckpt[key_alt_dark2].shape:
                load_dict[key_model] = ckpt[key_alt_dark2]
                logger.info(f"  -> Remapped Nested Stage 1: {key_alt_dark2} -> {key_model}")
                continue

    # Load matched parameters cleanly
    missing, unexpected = model.load_state_dict(load_dict, strict=False)
    logger.info(f"Pretrained weights successfully loaded: {len(load_dict)} tensors transferred.")

    # ASSERTION SAFEGUARDS
    # Assert that Scale-AttnRes queries remain strictly zero-initialized
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "backbone"):
        neck = m.backbone
        if hasattr(neck, "attnres_n3"):
            assert torch.all(neck.attnres_n3.w == 0.0), "Invariant violation: attnres_n3.w was corrupted on load!"
            assert torch.all(neck.attnres_n4.w == 0.0), "Invariant violation: attnres_n4.w was corrupted on load!"
            assert torch.all(neck.attnres_n5.w == 0.0), "Invariant violation: attnres_n5.w was corrupted on load!"
            logger.info("Scale-AttnRes zero-initialization invariant verified.")

    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)