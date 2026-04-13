import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, Bottleneck
from .ttt_modules import SimAM, CoordinateAttention, MBConvConditioner, DeepSeekSparseAttention, InstanceConditionedRouter

class C2f_Tribrid(nn.Module):
    """mHC-Constrained Tribrid Neck Block"""
    def __init__(self, c1, c2, n=1, shortcut=False, depthwise=False, act="silu", e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = BaseConv(c1, c2, 1, 1, act=act) 
        self.cv2 = BaseConv((2 + n) * self.c, c2, 1, 1, act=act)
        
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, 1.0, depthwise, act) for _ in range(n)
        )
        
        self.global_mixer = nn.Sequential(
            SimAM(),
            CoordinateAttention(self.c),
            MBConvConditioner(self.c),
            DeepSeekSparseAttention(self.c) 
        )
        
        self.dynamic_router = InstanceConditionedRouter(self.c)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        global_context = self.global_mixer(y[-1])
        routing_logits = self.dynamic_router(y[-1])
        
        norm_weights = F.softmax(routing_logits, dim=-1) 
        w_local = norm_weights[:, 0].view(-1, 1, 1, 1)
        w_global = norm_weights[:, 1].view(-1, 1, 1, 1)
        
        # Convex Manifold Fusion
        y[-1] = (w_local * y[-1]) + (w_global * global_context)
        
        return self.cv2(torch.cat(y, 1))