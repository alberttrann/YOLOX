import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_blocks import BaseConv, DWConv
from .ttt_modules import EngramMemoryBank, UncertaintyEstimator

class TDE_Head(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu"):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = 128
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        
        # Engram Specifics
        self.memory_banks = nn.ModuleList()
        self.uncertainty_gates = nn.ModuleList()
        self.latent_projectors = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(int(in_channels[i]*width), int(256*width), 1, 1, act=act))
            
            # Branch 1: REGRESSION (Memory-Free)
            self.reg_convs.append(nn.Sequential(
                BaseConv(int(256*width), int(256*width), 3, 1, act=act),
                BaseConv(int(256*width), int(256*width), 3, 1, act=act)
            ))
            
            # Branch 2: CLASSIFICATION (Engram-Augmented)
            self.cls_convs.append(nn.Sequential(
                BaseConv(int(256*width), int(256*width), 3, 1, act=act),
                BaseConv(int(256*width), int(256*width), 3, 1, act=act)
            ))
            
            # Identity Restoration Components
            self.latent_projectors.append(nn.Linear(int(256*width), self.latent_dim))
            self.memory_banks.append(EngramMemoryBank(num_classes, self.latent_dim))
            self.uncertainty_gates.append(UncertaintyEstimator(self.latent_dim))
            
            # Final Predictors
            self.cls_preds.append(nn.Conv2d(int(256*width), num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(int(256*width), 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256*width), 1, 1, 1, 0))

    def forward(self, xin):
        outputs = []
        for k, (x, stem, cls_conv, reg_conv) in enumerate(zip(xin, self.stems, self.cls_convs, self.reg_convs)):
            x = stem(x)
            
            # 1. REGRESSION PATH (High Fidelity)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat) # Objectness derived from structural edges
            
            # 2. CLASSIFICATION PATH (Engram Augmented)
            cls_feat = cls_conv(x)
            B, C, H, W = cls_feat.shape
            
            # Flatten for memory operations
            cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            # Step A: Project to Latent Space
            latent_vec = self.latent_projectors[k](cls_feat_flat)
            
            # Step B: Check Uncertainty and Objectness
            uncertainty = self.uncertainty_gates[k](latent_vec) # [B, HW, 1]
            obj_mask = torch.sigmoid(obj_output.view(B, 1, -1).permute(0, 2, 1)) # [B, HW, 1]
            
            # Step C: Retrieve Clean Identity from Memory
            memory_feat = self.memory_banks[k](latent_vec, uncertainty, obj_mask)
            
            # Step D: Fusion (Restore Identity)
            # Re-project memory back to feature dimension
            memory_inflated = F.linear(memory_feat, self.latent_projectors[k].weight.t())
            restored_cls_feat = cls_feat_flat + memory_inflated
            
            # Reshape back to spatial
            restored_cls_feat = restored_cls_feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Final Class Scores
            cls_output = self.cls_preds[k](restored_cls_feat)
            
            # Concat for YOLOX Post-processing
            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)
            
        return outputs