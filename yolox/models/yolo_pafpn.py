#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


from .tribrid_neck import C2f_Tribrid

class YOLOPAFPN(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        # Passing TTT parameters to inner backbone
        ttt_lr=0.005,
        ttt_noise_std=0.05
    ):
        super().__init__()
        # Instantiate the TTT-Aware Backbone
        self.backbone = CSPDarknet(
            depth, width, out_features=in_features, 
            depthwise=depthwise, act=act,
            ttt_lr=ttt_lr, ttt_noise_std=ttt_noise_std
        )
        self.in_features = in_features
        self.in_channels = in_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Lateral convolutions
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.bu_conv2 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.bu_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        
        # Replace every CSPLayer with C2f_Tribrid
        # C3_p4: lateral_conv0(P5) [256] + P4 [256] = 512
        self.C3_p4 = C2f_Tribrid(
            int(2 * in_channels[1] * width), 
            int(in_channels[1] * width), 
            round(3 * depth), False, depthwise=depthwise, act=act
        )
        
        # C3_p3: reduce_conv1(p4) [128] + P3 [128] = 256
        self.C3_p3 = C2f_Tribrid(
            int(2 * in_channels[0] * width), 
            int(in_channels[0] * width), 
            round(3 * depth), False, depthwise=depthwise, act=act
        )
        
        # C3_n3: bu_conv2(p3) [128] + fpn_out1 [128] = 256
        self.C3_n3 = C2f_Tribrid(
            int(2 * in_channels[0] * width), 
            int(in_channels[1] * width), 
            round(3 * depth), False, depthwise=depthwise, act=act
        )
        
        # C3_n4: bu_conv1(n3) [256] + fpn_out0 [256] = 512
        self.C3_n4 = C2f_Tribrid(
            int(2 * in_channels[1] * width), 
            int(in_channels[2] * width), 
            round(3 * depth), False, depthwise=depthwise, act=act
        )
    
    @property
    def current_epoch(self):
        return getattr(self.backbone, "current_epoch", 0)

    @current_epoch.setter
    def current_epoch(self, val):
        if hasattr(self.backbone, "current_epoch"):
            self.backbone.current_epoch = val
        else:
            # Fallback for manual assignment
            setattr(self.backbone, "current_epoch", val)

    def forward(self, input, ttt_prob=0.0):
        """
        Args:
            inputs: input images.
            ttt_prob: Probability of running TTT (passed from YOLOX).
        """

        #  backbone
        out_features = self.backbone(input, ttt_prob=ttt_prob)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
