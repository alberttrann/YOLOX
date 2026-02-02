import torch
import torch.nn as nn
from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import SimAM, CoordinateAttention, MBConvConditioner, DeepSeekSparseAttention

class C2f_Tribrid(nn.Module):
    """
    LW-DETR C2f Block + Tribrid Global Context Branch.
    Branch A: Dense Local (Standard Bottlenecks)
    Branch B: Sparse Global (SimAM -> CA -> MBConv -> DSA)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = BaseConv(c1, c2, 1, 1)
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1)
        
        # Branch A: Local Dense Bottlenecks
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        
        # Branch B: Tribrid Global Context Mixer
        self.global_mixer = nn.Sequential(
            SimAM(),
            CoordinateAttention(self.c),
            MBConvConditioner(self.c),
            DeepSeekSparseAttention(self.c)
        )
        
        # D-YOLO Gated Fusion: Initialized at 0.01 to allow safe warm-up
        self.gate = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        # Split input: y[0] goes to local path, y[1] is base for global
        
        # Local Dense Path
        y.extend(m(y[-1]) for m in self.m)
        
        # Global Sparse Path (Applied to the final local feature)
        # This provides global reasoning on top of the deepest local features
        global_context = self.global_mixer(y[-1])
        
        # Gated Skip Connection
        y[-1] = y[-1] + torch.tanh(self.gate) * global_context
        
        return self.cv2(torch.cat(y, 1))