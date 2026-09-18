#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Integrated for TDE-YOLOX v3.1: GroupNorm Instance Centering & Parameter Invariance

import math
import torch
import torch.nn as nn


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
        raise AttributeError(f"Unsupported act type: {name}")
    return module


class BaseConv(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> Activation block"""
    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
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
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class BaseConvGN(nn.Module):
    """
    Conv2d -> GroupNorm -> Activation block.
    Stateless instance-level normalization: eliminates frozen running-statistic DC shifts
    under adverse weather (fog, blinding snow, extreme dynamic range shifts).
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
        
        # Robust group count derivation: guarantees valid divisibility
        if out_channels % target_groups == 0:
            actual_groups = target_groups
        else:
            actual_groups = math.gcd(out_channels, target_groups)
            if actual_groups < 4:
                for i in range(target_groups, 1, -1):
                    if out_channels % i == 0:
                        actual_groups = i
                        break
                else:
                    actual_groups = 1  # Fallback to LayerNorm behavior if prime

        self.gn = nn.GroupNorm(actual_groups, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

    def fuseforward(self, x):
        # GroupNorm is instance-dependent and cannot be statically folded into conv weights
        return self.forward(x)


class DWConv(nn.Module):
    """Depthwise Conv + Pointwise Conv with BatchNorm"""
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
    """Depthwise Conv + Pointwise Conv with GroupNorm (TTT-Safe)"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConvGN(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConvGN(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck with switchable Normalization layer type"""
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        use_gn=False
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
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
    """Residual layer with `in_channels` inputs."""
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
    """Spatial pyramid pooling layer used in YOLOv3-SPP / YOLOX-SPP"""
    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
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
    """CSP Bottleneck with recursive GroupNorm propagation support"""
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        use_gn=False
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        ConvBlock = BaseConvGN if use_gn else BaseConv

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
    """
    Focus width and height information into channel space.
    Upgraded for TDE-YOLOX v3.1: Uses BaseConvGN to prevent frozen-BN saturation
    on un-normalized adverse weather pixel distributions.
    """
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        # Input has in_channels * 4 after space-to-depth slicing
        self.conv = BaseConvGN(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
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