#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import functools
import os
import time
from collections import defaultdict, deque
import psutil

import numpy as np

import torch

__all__ = [
    "AverageMeter",
    "MeterBuffer",
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
    "mem_usage"
]


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        cuda_device = int(visible_devices[cuda_device])
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def mem_usage():
    """
    Compute the memory usage for the current machine (GB).
    """
    gb = 1 << 30
    mem = psutil.virtual_memory()
    return mem.used / gb


class AverageMeter:
    """
    TDE-YOLOX METER: 
    Stateless, Windowed, and CUDA-Safe.
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        # Call clear() to initialize all underscored attributes
        self.clear() 

    def clear(self):
        """
        YOLOX's internal call site in MeterBuffer.
        """
        self._deque = deque(maxlen=self.window_size)
        self._sum = 0.0
        self._count = 0
        self.current_val = 0.0

    def update(self, val):
        """
        Force-converts any input to a CPU scalar before storage.
        Prevents VRAM leaks and NumPy/CUDA collisions.
        """
        if val is None:
            return
            
        # 1. Forensic Extraction
        if torch.is_tensor(val):
            # detach() is critical to stop Meta-Learning graph growth
            clean_val = val.detach().cpu().item()
        elif hasattr(val, "item"): 
            clean_val = val.item()
        else:
            clean_val = float(val)

        # 2. Update stats
        self.current_val = clean_val
        self._deque.append(clean_val)
        self._sum += clean_val
        self._count += 1

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        """Windowed average."""
        if len(self._deque) == 0:
            return 0.0
        return np.mean(list(self._deque))

    @property
    def global_avg(self):
        """Running average across the whole epoch."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
    
    @property
    def latest(self):
        return self.current_val

    @property
    def total(self):
        return self._sum


class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    # In MeterBuffer class:
    def get_filtered_meter(self, filter_key="loss"):
            """
            Handles YOLOX versions where MeterBuffer is a dict subclass.
            """
            tde_metrics = ["mem_loss", "ttt_prob"]
            filtered_dict = {}
            
            source = getattr(self, "meters", self)
            
            for k, v in source.items():
                if filter_key in k or k in tde_metrics:
                    filtered_dict[k] = v
            return filtered_dict

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()
