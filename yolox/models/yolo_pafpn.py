#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


from .tribrid_neck import C2f_Tribrid

class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_channels=[256, 512, 1024], act="silu"):
        super().__init__()
        # Backbone (Now includes Phase 1 TTTAdaptiveStage)
        self.backbone = CSPDarknet(depth, width, act=act)
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Replace every CSPLayer with C2f_Tribrid
        # Top-Down Path
        self.C3_p4 = C2f_Tribrid(int(in_channels[2]*width + in_channels[1]*width), int(in_channels[1]*width), round(3*depth))
        self.C3_p3 = C2f_Tribrid(int(in_channels[1]*width + in_channels[0]*width), int(in_channels[0]*width), round(3*depth))
        
        # Bottom-Up Path
        self.C3_n3 = C2f_Tribrid(int(in_channels[0]*width + in_channels[0]*width), int(in_channels[1]*width), round(3*depth))
        self.C3_n4 = C2f_Tribrid(int(in_channels[1]*width + in_channels[1]*width), int(in_channels[2]*width), round(3*depth))

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
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
