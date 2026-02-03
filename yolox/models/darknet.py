import torch
from torch import nn
from .network_blocks import BaseConv, CSPLayer, Focus, DWConv, SPPBottleneck
from .ttt_modules import TTTAdaptiveStage 

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
        # FOR TDE-YOLOX
        ttt_lr=0.005,      # Meta-learning step size
        ttt_noise_std=0.05 # DINO-logic noise intensity
    ):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # 0. The Static Stem (Focus Layer)
        # Converts 3-channel pixels to base_channels features
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 1. The Adaptive Dark2 (Meta-TTT Stage)
        # the ONLY stage that learns to adapt itself
        dark2_base = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        
        self.dark2 = TTTAdaptiveStage(
            dark2_base, 
            in_channels=base_channels * 2,
            ttt_lr=ttt_lr,
            noise_std=ttt_noise_std # (Denoising)
        )

        # 2. Dark3 (Deep Stage)
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 3. Dark4 (Deep Stage)
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # 4. Dark5 (Deep Stage + SPP)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x, ttt_prob=0.0): 
        outputs = {}
        # Step 0: Stem
        x = self.stem(x)
        
        # Step 1: ADAPTIVE FEATURE EXTRACTION
        # Stochastic Meta-Learning Logic
        run_ttt = False
        
        if not self.training:
            # Inference: Always run if enabled in config (handled by ttt_prob=1.0 from YOLOX)
            run_ttt = True
        else:
            # Training: Roll the dice against the annealed probability
            if torch.rand(1).item() < ttt_prob:
                run_ttt = True
        
        # Execute Stage 1 with decision
        x = self.dark2(x, run_ttt=run_ttt)
        outputs["dark2"] = x
        
        # Step 2-4: Deep backbone processing
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        
        return {k: v for k, v in outputs.items() if k in self.out_features}