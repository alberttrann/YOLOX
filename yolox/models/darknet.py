import torch
from torch import nn
from .network_blocks import BaseConv, BaseConvGN, CSPLayer, Focus, DWConv, DWConvGN, SPPBottleneck
from .ttt_modules import TTTAdaptiveStage 

class Darknet(nn.Module):
    # (Standard Darknet code remains intact per instructions)
    pass

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

    def forward(self, x, ttt_prob=None): 
        """
        Args:
            x: Input image tensor
            ttt_prob: Probability [0.0, 1.0] of running TTT loop. 
                      If None, defaults to behavior based on self.training.
        """
        outputs = {}
        
        # Step 0: Input Stem
        x = self.stem(x)
        outputs["stem"] = x
        
        # Step 1: ADAPTIVE FEATURE EXTRACTION (TTT)
        run_ttt = False
        
        if not self.training:
            # Inference Mode:
            # Default to True unless explicitly disabled (ttt_prob=0.0)
            if ttt_prob is None or ttt_prob > 0.0:
                run_ttt = True
        else:
            # Training Mode:
            # Use stochastic probability if provided, else default to 0.0 (off)
            prob = ttt_prob if ttt_prob is not None else 0.0
            if prob > 0.0 and torch.rand(1).item() < prob:
                run_ttt = True
        
        # Execute the Adaptive Stage
        # The TTTAdaptiveStage handles the inner optimization loop internally
        x = self.dark2(x, run_ttt=run_ttt)
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