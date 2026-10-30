#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

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
    """Standard Conv2d -> BatchNorm2d -> Activation block."""

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
    TDE-YOLOX Stateless Normalization Primitive: Conv2d -> GroupNorm -> Activation.
    
    Dynamically computes the optimal group count (targeting 16 groups) while 
    strictly guaranteeing out_channels % actual_groups == 0. Eliminates 
    BatchNorm running_mean/running_var dependency for stateless meta-adaptation.
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

        if out_channels % target_groups == 0:
            actual_groups = target_groups
        else:
            actual_groups = math.gcd(out_channels, target_groups)
            if actual_groups < 4:
                for i in range(target_groups, 0, -1):
                    if out_channels % i == 0:
                        actual_groups = i
                        break

        self.gn = nn.GroupNorm(actual_groups, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

    def fuseforward(self, x):
        return self.forward(x)


class DWConv(nn.Module):
    """Depthwise Conv + Conv using standard BatchNorm2d."""

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
    """Depthwise Conv + Conv using GroupNorm for stateless meta-adaptation."""

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
    """Standard residual bottleneck with selectable GroupNorm/BatchNorm support."""

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        use_gn=False,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        ConvBlock = BaseConvGN if use_gn else BaseConv
        DWConvBlock = DWConvGN if use_gn else DWConv
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
    """Spatial pyramid pooling layer used in YOLOv3-SPP."""

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
    """C3 in yolov5, CSP Bottleneck with optional GroupNorm propagation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        use_gn=False,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        ConvBlock = BaseConvGN if use_gn else BaseConv

        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                act=act,
                use_gn=use_gn,
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
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

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


# =========================================================================
# ADVANCED UAV/SCALE-PRESERVING BLOCKS (EMA & C2fA)
# =========================================================================

class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention (Ouyang et al., ICASSP 2023).
    
    PATCH 2.1 APPLIED: Zero-Initialized Identity Residual (X + gamma * EMA(X)).
    Eliminates the 0.50x initial feature magnitude shock in deep backbone stages (dark4, dark5).
    At Step 0, gamma = 0.0 guarantees exact mathematical identity passthrough of pre-trained COCO features.
    """

    def __init__(self, channels, factor=8):
        super().__init__()
        if channels % factor == 0:
            self.groups = factor
        else:
            self.groups = math.gcd(channels, factor)
            if self.groups < 2:
                self.groups = 1

        self.channels_per_group = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gn_groups = math.gcd(self.channels_per_group, 8)
        self.gn = nn.GroupNorm(gn_groups, self.channels_per_group)

        self.conv1x1 = nn.Conv2d(
            self.channels_per_group, self.channels_per_group, kernel_size=1, stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2d(
            self.channels_per_group, self.channels_per_group, kernel_size=3, stride=1, padding=1
        )

        # PATCH 2.1: Zero-initialized learnable residual gating parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, self.channels_per_group, h, w)

        # 1. 1D Directional Strip Encodings along H and W
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 2. Parallel 1x1 Semantic Attention and 3x3 Spatial Feature Extraction
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # 3. Cross-Spatial Global Interaction
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, self.channels_per_group, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, self.channels_per_group, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        ema_out = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        # PATCH 2.1 APPLIED: Non-destructive identity residual addition
        return x + self.gamma * ema_out


class C2fA(nn.Module):
    """
    YOLO-UTD C2fA Module (Sensors 2026).
    
    Integrates the EMA cross-spatial attention mechanism into standard C2f blocks.
    Replaces standard CSPLayer in deep stages (dark4 stride-16, dark5 stride-32)
    to maintain spatial centroids of distant and small objects prior to Neck aggregation.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, depthwise=False, act="silu", e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = BaseConv(c1, c2, 1, 1, act=act)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, 1, act=act)
        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                act=act,
                use_gn=False,
            )
            for _ in range(n)
        )
        self.ema = EMA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.ema(out)