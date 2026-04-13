import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import BaseConv
from .tribrid_neck import C2f_Tribrid

class BlockAttnRes(nn.Module):
    """
    Kimi 'Attention Residuals'.
    Full-Fidelity: Agnostic initialization to force historical feature exploration.
    """
    def __init__(self, current_channels, history_channels_list):
        super().__init__()
        self.C = current_channels
        
        self.history_projs = nn.ModuleList([
            nn.Conv2d(hc, current_channels, 1) for hc in history_channels_list
        ])
        
        self.query_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(current_channels, current_channels, 1)
        )
        
        self.key_proj = nn.Conv2d(current_channels, current_channels, 1)
        self.norm = nn.GroupNorm(8, current_channels)
        
        # --- THE EXPERT FIX: Initialization ---
        # Ensure query and key projections start near zero
        # This makes initial dot products ~0.0, leading to uniform attention weights.
        # e.g., Softmax([0,0,0]) = [0.33, 0.33, 0.33].
        nn.init.constant_(self.query_proj[1].weight, 0.0)
        nn.init.constant_(self.query_proj[1].bias, 0.0)

    def forward(self, current_feat, history_feats):
        B, C, H, W = current_feat.shape
        
        aligned_history = []
        for i, feat in enumerate(history_feats):
            feat_proj = self.history_projs[i](feat)
            
            if feat_proj.shape[-1] != W or feat_proj.shape[-2] != H:
                if feat_proj.shape[-1] > W:
                    feat_proj = F.adaptive_max_pool2d(feat_proj, (H, W))
                else:
                    feat_proj = F.interpolate(feat_proj, size=(H, W), mode='nearest')
            aligned_history.append(feat_proj)
            
        all_feats = aligned_history + [current_feat]
        
        V = torch.stack(all_feats, dim=1) 
        
        V_flat = V.view(B * len(all_feats), C, H, W)
        K_flat = self.key_proj(self.norm(V_flat))
        K_pool = F.adaptive_avg_pool2d(K_flat, 1).view(B, len(all_feats), C) 
        
        Q = self.query_proj(current_feat).view(B, C, 1)
        
        attn_logits = torch.bmm(K_pool, Q).squeeze(-1) / (C ** 0.5) 
        
        # REMOVED the +5.0 "Safety" bias. Let the network learn to weight the features!
        attn_weights = F.softmax(attn_logits, dim=-1) 
        
        attn_weights = attn_weights.view(B, len(all_feats), 1, 1, 1)
        out = torch.sum(V * attn_weights, dim=1)
        
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import BaseConv
from .tribrid_neck import C2f_Tribrid

class YOLOPAFPN(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2", "dark3", "dark4", "dark5"),
        in_channels=[128, 256, 512, 1024], # Raw backbone stage counts
        depthwise=False,
        act="silu",
        ttt_lr=0.02,
        ttt_noise_std=0.1
    ):
        super().__init__()
        # Instantiate the TTT-Aware Backbone
        self.backbone = CSPDarknet(
            depth, width, out_features=in_features, 
            depthwise=depthwise, act=act,
            ttt_lr=ttt_lr, ttt_noise_std=ttt_noise_std
        )
        self.in_features = in_features
        
        # --- CHANNEL CALCULATION (Width Multiplied) ---
        # YOLOX-S (width=0.5): 64, 128, 256, 512
        c2 = int(in_channels[0] * width) # 64  (TTT Adapted)
        c3 = int(in_channels[1] * width) # 128 (P3)
        c4 = int(in_channels[2] * width) # 256 (P4)
        c5 = int(in_channels[3] * width) # 512 (P5)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Lateral/Reduction Convolutions
        self.lateral_conv0 = BaseConv(c5, c4, 1, 1, act=act) # P5 -> P4 logic
        self.reduce_conv1  = BaseConv(c4, c3, 1, 1, act=act) # P4 -> P3 logic
        
        # Bottom-Up Convolutions (Downsampling)
        self.bu_conv2 = BaseConv(c3, c3, 3, 2, act=act) # P3 -> P4
        self.bu_conv1 = BaseConv(c4, c4, 3, 2, act=act) # P4 -> P5
        
        # --- PHASE 2: ATTNRES MODULES (Kimi-inspired Lookback) ---
        # P4 lookback: Querying (Dark2, Dark3) from current P4 features
        self.attn_res_p4 = BlockAttnRes(current_channels=c4, history_channels_list=[c2, c3])
        # P3 lookback: Querying (Dark2) from current P3 features
        self.attn_res_p3 = BlockAttnRes(current_channels=c3, history_channels_list=[c2])

        # --- THE MHC TRIBRID BLOCKS ---
        # C3_p4 (Top-Down): fpn_out0 [256] + P4_Attn [256] = 512 in
        self.C3_p4 = C2f_Tribrid(c4 + c4, c4, round(3 * depth), False, depthwise=depthwise, act=act)
        
        # C3_p3 (Top-Down): fpn_out1 [128] + P3_Attn [128] = 256 in
        self.C3_p3 = C2f_Tribrid(c3 + c3, c3, round(3 * depth), False, depthwise=depthwise, act=act)
        
        # --- FIXED: BOTTOM-UP CHANNEL MATH ---
        # C3_n3: bu_conv2 out [128] + reduce_conv1 out [128] = 256 in (Previously 384)
        self.C3_n3 = C2f_Tribrid(c3 + c3, c4, round(3 * depth), False, depthwise=depthwise, act=act)
        
        # C3_n4: bu_conv1 out [256] + lateral_conv0 out [256] = 512 in (Previously 768)
        self.C3_n4 = C2f_Tribrid(c4 + c4, c5, round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, input, ttt_prob=0.0):
        # 1. Extract Backbone
        out_features = self.backbone(input, ttt_prob=ttt_prob)
        features = [out_features[f] for f in self.in_features]
        [x_dark2, x_dark3, x_dark4, x_dark5] = features

        # 2. Top-Down Pathway
        fpn_out0 = self.lateral_conv0(x_dark5)  # 512 -> 256
        f_out0 = self.upsample(fpn_out0)        # 256
        
        # Lookback to S1/S2 via AttnRes
        x_dark4_attn = self.attn_res_p4(x_dark4, [x_dark2, x_dark3]) # 256
        f_out0 = torch.cat([f_out0, x_dark4_attn], 1)               # 512
        f_out0 = self.C3_p4(f_out0)                                 # 256

        fpn_out1 = self.reduce_conv1(f_out0)    # 256 -> 128
        f_out1 = self.upsample(fpn_out1)        # 128
        
        # Lookback to S1 via AttnRes
        x_dark3_attn = self.attn_res_p3(x_dark3, [x_dark2])         # 128
        f_out1 = torch.cat([f_out1, x_dark3_attn], 1)               # 256
        pan_out2 = self.C3_p3(f_out1)                               # 128 (P3_fused)

        # 3. Bottom-Up Pathway
        p_out1 = self.bu_conv2(pan_out2)            # 128 -> 128
        p_out1 = torch.cat([p_out1, fpn_out1], 1)   # 256
        pan_out1 = self.C3_n3(p_out1)               # 256 (P4_fused)

        p_out0 = self.bu_conv1(pan_out1)            # 256 -> 256
        p_out0 = torch.cat([p_out0, fpn_out0], 1)   # 512
        pan_out0 = self.C3_n4(p_out0)               # 512 (P5_fused)

        return (pan_out2, pan_out1, pan_out0)