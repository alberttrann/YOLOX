import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_head import YOLOXHead 
from .ttt_modules import EngramMemoryBank, UncertaintyEstimator

class TDE_Head(YOLOXHead):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
        
        self.latent_dim = 128
        self.memory_banks = nn.ModuleList()
        self.uncertainty_gates = nn.ModuleList()
        self.latent_projectors = nn.ModuleList()

        for i in range(len(in_channels)):
            feat_channels = int(256 * width)
            self.latent_projectors.append(nn.Linear(feat_channels, self.latent_dim))
            self.memory_banks.append(EngramMemoryBank(num_classes, self.latent_dim))
            self.uncertainty_gates.append(UncertaintyEstimator(self.latent_dim))

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            
            # BRANCH 1: REGRESSION (Memory-Free)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # BRANCH 2: CLASSIFICATION (Engram-Augmented)
            cls_feat = cls_conv(x)
            B, C, H, W = cls_feat.shape
            cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            latent_vec = self.latent_projectors[k](cls_feat_flat)
            uncertainty = self.uncertainty_gates[k](latent_vec)
            obj_mask = torch.sigmoid(obj_output.view(B, 1, -1).permute(0, 2, 1))
            
            memory_feat = self.memory_banks[k](latent_vec, uncertainty, obj_mask)
            memory_inflated = F.linear(memory_feat, self.latent_projectors[k].weight.t())
            restored_cls_feat = (cls_feat_flat + memory_inflated).reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            cls_output = self.cls_preds[k](restored_cls_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output_tmp = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output_tmp = reg_output_tmp.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output_tmp.clone())
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        if self.training:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype)
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs