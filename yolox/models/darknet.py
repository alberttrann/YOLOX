import torch
from torch import nn
from .network_blocks import BaseConv, BaseConvGN, CSPLayer, Focus, DWConv, DWConvGN, SPPBottleneck, ResLayer
from .ttt_modules import TTTAdaptiveStage 

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        # ESSENTIAL RESEARCH PARAMETERS
        ttt_lr=0.05,       # Aggressive Adaptation Rate
        ttt_noise_std=0.08 # Hard Denoising Task
    ):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        
        # Stage-specific Conv for TTT Stage (GN-based)
        # This ensures the entry convolution of the stage is also adaptable
        ConvGN = DWConvGN if depthwise else BaseConvGN

        base_channels = int(wid_mul * 64)  # 64 for YOLOX-S
        base_depth = max(round(dep_mul * 3), 1)  # 3 for YOLOX-S

        # 0. THE STATIC STEM
        # Focus layer converts 3-channel image to base_channels.
        # keep this standard BN to normalize pixel inputs initially.
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 1. THE ADAPTIVE DARK2 (STAGE 1)
        # construct the inner backbone stage using GroupNorm components.
        dark2_base = nn.Sequential(
            # Entry Convolution: Transforms Stem features to Stage 1 dimension
            ConvGN(base_channels, base_channels * 2, 3, 2, act=act),
            
            # CSP Block: The main processing unit of Stage 1
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
                use_gn=True # Forces internal blocks to use BaseConvGN
            ),
        )
        
        # Meta-Engine Wrapping
        # encapsulate the GN-based Dark2 in the TTT logic
        self.dark2 = TTTAdaptiveStage(
            dark2_base, 
            in_channels=base_channels * 2,
            init_ttt_lr=ttt_lr,
            noise_std=ttt_noise_std
        )

        # 2. DARK3 (DEEP BACKBONE) - Standard BN
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # 3. DARK4 (DEEP BACKBONE) - Standard BN
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # 4. DARK5 (DEEP BACKBONE + SPP) - Standard BN
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x, ttt_prob=None, ttt_noise_scale=1.0): 
        outputs = {}
        x = self.stem(x)
        
        run_ttt = False
        if not self.training:
            if ttt_prob is None or ttt_prob > 0.0: run_ttt = True
        else:
            prob = ttt_prob if ttt_prob is not None else 0.0
            if prob > 0.0 and torch.rand(1).item() < prob: run_ttt = True
        
        # Pass the dynamic noise scale to the adaptive stage
        # We multiply the base noise_std by the schedule scale
        if run_ttt:
            # We temporarily override the noise_std just for this forward pass
            original_noise = self.dark2.noise_std
            self.dark2.noise_std = original_noise * ttt_noise_scale
            x = self.dark2(x, run_ttt=True)
            self.dark2.noise_std = original_noise # Restore
        else:
            x = self.dark2(x, run_ttt=False)
            
        outputs["dark2"] = x
        
        # Step 2-4: Standard Forward pass through Deep Backbone
        # These layers benefit from the 'cleaned' features from Dark2
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        
        return {k: v for k, v in outputs.items() if k in self.out_features}