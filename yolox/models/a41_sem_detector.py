"""A4.1 with the original training-only tri-scale semantic decoder restored.

Never edits Fourier/Engram detection paths or executes the semantic decoder at
inference. Gradient scale applies ONLY to semantic gradients entering the neck.
"""
import torch
from .a4_detector import A4Detector
from .ttt_modules import WeakSemanticAffordanceHead
from .a41_sem_aux import SemanticSupervision


class A41SemanticDetector(A4Detector):
    def __init__(self,backbone,head,*,width,aux_mode,
                 sem_weight=0.25,backbone_grad_scale=0.05,
                 sigma_divisor=6.0,harmonic_weight=0.05):
        super().__init__(backbone,head)
        if not 0 <= sem_weight <= 1 or not 0 <= backbone_grad_scale <= 1:
            raise ValueError('Semantic objective weight and gradient scale must be in [0,1]')
        self.aux_mode=str(aux_mode)
        self.sem_weight=float(sem_weight)
        self.backbone_grad_scale=float(backbone_grad_scale)
        self.aux_sem_head=WeakSemanticAffordanceHead(
            in_channels_n3=int(256*width), in_channels_n4=int(512*width),
            in_channels_n5=int(1024*width),num_classes=head.num_classes,
            hidden_dim=64)
        self.sem_supervision=SemanticSupervision(
            head.num_classes,mode=aux_mode,sigma_divisor=sigma_divisor,
            harmonic_weight=harmonic_weight)
        self.last_semantic_losses={}

    def forward(self,x,targets=None):
        if not self.training:
            # Identical inference as A4.1: the training-only head is NOT called.
            return super().forward(x,targets)
        if targets is None:
            raise ValueError('Semantic auxiliary training needs YOLOX targets')
        if x.ndim!=4 or x.shape[-2]%32 or x.shape[-1]%32:
            raise ValueError('A4.1 requires stride-32-aligned NCHW input')
        pyramid=self.backbone(x,ttt_prob=0.0)
        loss,iou,conf,cls,l1,fg=self.head(pyramid,targets,x)
        memory=getattr(self.head,'aux_memory_loss',None)
        if memory is None:
            memory=loss*0
        # Identical forward activations, precisely scaled auxiliary gradients.
        scale=self.backbone_grad_scale
        sem_features=tuple(f.detach()+scale*(f-f.detach()) for f in pyramid)
        sem_logits=self.aux_sem_head(*sem_features)
        if sem_logits.shape[1]!=self.num_classes:
            raise RuntimeError('Auxiliary class count differs from detection head')
        if sem_logits.shape[-2:]!=pyramid[0].shape[-2:]:
            raise RuntimeError('Semantic head must operate at the N3/P3 geometry')
        sem_raw=self.sem_supervision(sem_logits,targets,x.shape[-2:])
        sem_weighted=self.sem_weight*sem_raw
        total=loss.float()+sem_weighted.float()
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite semantic/detection total loss')
        self.last_semantic_losses={
            'raw':sem_raw.detach(),'weighted':sem_weighted.detach(),
            **self.sem_supervision.last_stats}
        zero=loss.detach()*0
        return {
            'total_loss':total,'det_loss':(loss-memory).detach(),
            'iou_loss':iou,'conf_loss':conf,'cls_loss':cls,'l1_loss':l1,
            'num_fg':fg,'mem_loss':memory,'proto_loss':zero,
            'intra_loss':zero,'rec_loss':zero,
            'afford_loss':sem_raw.detach(),
            # Train-only aliases for rigorous per-loss gradient comparisons.
            'sem_aux_loss':sem_weighted,'sem_aux_raw':sem_raw.detach(),
            'ttt_prob':0.0,
        }
