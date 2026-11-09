"""Fully differentiable training, single-pass inference, no test-time adaptation.

A0/A1/A2/A3 differ only in classification-adapter/memory evidence.
Backbone, SPI, Tribrid, bbox, objectness, assignment and optimizer stay shared.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FourierDetector(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone=backbone
        self.head=head
        self.num_classes=head.num_classes
        self.current_epoch=1
        self.max_epochs=80

    @property
    def eval_ttt_enabled(self):
        return False

    @eval_ttt_enabled.setter
    def eval_ttt_enabled(self,value):
        if bool(value):
            raise RuntimeError('TTT is absent in FourierDetector; run native DG evaluation')

    def set_meta_training_state(self,epoch,max_epochs):
        self.current_epoch=int(epoch)
        self.max_epochs=int(max_epochs)
        self.head.set_epoch_state(epoch,max_epochs)

    @torch.no_grad()
    def project_prototypes_to_unit_sphere(self):
        if hasattr(self.head,'memory_banks'):
            for bank in self.head.memory_banks:
                bank.prototypes.copy_(F.normalize(bank.prototypes,dim=-1))

    def forward(self, x, targets=None):
        if x.ndim != 4:
            raise ValueError(f'Expected NCHW image, got {tuple(x.shape)}')
        if x.shape[-2] % 32 != 0 or x.shape[-1] % 32 != 0:
            raise ValueError('Use geometry-preserving, stride-32-aligned padding; '
                             'automatic input stretching is forbidden')
        pyramid=self.backbone(x,ttt_prob=0.0)
        if not self.training:
            return self.head(pyramid)
        if targets is None:
            raise ValueError('Training requires targets')
        loss, iou, conf, cls, l1, fg = self.head(pyramid,targets,x)
        zero=loss.detach()*0.0
        return {'total_loss':loss, 'det_loss':loss.detach(),
                'iou_loss':iou, 'conf_loss':conf, 'cls_loss':cls,
                'l1_loss':l1, 'num_fg':fg,
                'mem_loss':zero, 'proto_loss':zero, 'intra_loss':zero,
                'rec_loss':zero, 'afford_loss':zero,'ttt_prob':0.0}

    def visualize(self,x,targets,save_prefix='assign_vis_'):
        pyramid=self.backbone(x,ttt_prob=0.0)
        return self.head.visualize_assign_result(pyramid,targets,x,save_prefix)
