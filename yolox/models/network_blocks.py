#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """Standard Conv2d -> Batchnorm -> Activation block"""
    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class BaseConvGN(nn.Module):
    """
    TDE-YOLOX EXCLUSIVE: Conv2d -> GroupNorm -> Activation block.
    
    Robustness Feature: Dynamic Group Calculation.
    Instead of hard-failing to LayerNorm (groups=1), this calculates the 
    largest valid group count that is closest to the desired target (16 or 8).
    This preserves the 'Group' semantics even with odd channel counts.
    """
    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", target_groups=16
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        
        # Robust Group Calculation
        if out_channels % target_groups == 0:
            actual_groups = target_groups
        else:
            # Find largest divisor <= target_groups (prefer 8, then 4, etc.)
            actual_groups = math.gcd(out_channels, target_groups)
            if actual_groups < 4: 
                # If we can't find a good group size, find ANY divisor > 1
                # This avoids LayerNorm collapse unless absolutely necessary
                for i in range(target_groups, 1, -1):
                    if out_channels % i == 0:
                        actual_groups = i
                        break
        
        self.gn = nn.GroupNorm(actual_groups, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))
        
    def fuseforward(self, x):
        # GroupNorm cannot be fused into Conv weights mathematically for inference speedup
        # because it depends on instance statistics, unlike BatchNorm which uses fixed running stats.
        # We perform standard forward to be safe.
        return self.forward(x)

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class DWConvGN(nn.Module):
    """
    TDE-YOLOX EXCLUSIVE: Depthwise Conv + Conv with GroupNorm.
    Essential for TTT-Stage 1 where both Depthwise and Pointwise layers 
    need to be adaptable without buffer dependency.
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        # Depthwise part: groups = in_channels
        self.dconv = BaseConvGN(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        # Pointwise part: standard convolution
        self.pconv = BaseConvGN(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        use_gn=False # TTT-Ready Flag
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Expert Selector: Dynamically choose Norm type based on stage requirement
        if use_gn:
            ConvBlock = BaseConvGN
            DWConvBlock = DWConvGN
        else:
            ConvBlock = BaseConv
            DWConvBlock = DWConv
            
        Conv = DWConvBlock if depthwise else ConvBlock
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu", use_gn=False
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        # Select the correct ConvBlock based on the flag
        ConvBlock = BaseConvGN if use_gn else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        use_gn=False # TTT-Ready Flag
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        
        # Propagate GN choice to all internal components
        if use_gn:
            ConvBlock = BaseConvGN
        else:
            ConvBlock = BaseConv
            
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, use_gn=use_gn
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        # Focus layer remains standard BN (Stage 0) as per TDE-YOLOX design
        # to ensure initial pixel statistics are handled by standard mechanism.
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)