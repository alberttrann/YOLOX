"""vNext.3 top-level model. Native loss and semantic loss are intentionally distinct."""
from torch import nn
from .yolo_pafpn import YOLOPAFPN
from .pgsma_head import PGSMAHead
from .pgsma import PGSMA


class PGSMAYOLOX(nn.Module):
    def __init__(self, backbone=None, head=None, num_classes=9, width=.5,
                 semantic_dim=64, semantic_modes=4, correction='direct'):
        super().__init__()
        self.backbone = backbone if backbone is not None else YOLOPAFPN(.33, width,
            pretrained_compatible_dark2=True, ttt_residual_adapter=False)
        self.head = head if head is not None else PGSMAHead(num_classes, width=width)
        self.head.pgsma_engine = PGSMA(num_classes, channels=int(width*256),
                                      dim=semantic_dim, modes=semantic_modes,
                                      correction=correction)
        self.num_classes = num_classes
        self.enable_semantic_training = True

    @property
    def semantic(self):
        return self.head.pgsma_engine

    def native_parameters(self):
        return (p for n,p in self.named_parameters() if 'head.pgsma_engine.' not in n)

    def semantic_parameters(self):
        return self.semantic.parameters()

    def forward(self, x, targets=None, valid_image_mask=None):
        # dark2 reconstruction TTT is explicitly disabled in both modes.
        # Existing dark2 wrapper remains only for checkpoint compatibility;
        # the new experiment creates it with residual_adapter=False.
        fpn = self.backbone(x, ttt_prob=0.0)
        if self.training:
            # valid_anchor_mask explicitly flows through matching, objectness
            # normalization, and semantic hard-negative selection.
            valid_anchor_mask = None
            if valid_image_mask is not None:
                from .pgsma_validity import valid_anchors_from_image_mask
                valid_anchor_mask=valid_anchors_from_image_mask(
                    valid_image_mask, fpn, strides=self.head.strides)
            return self.head(fpn, targets, x, valid_anchor_mask=valid_anchor_mask)
        return self.head(fpn)

    def semantic_losses(self, seed=17, correction_mode='pgsma', **kwargs):
        if correction_mode == 'memory':
            kwargs.setdefault('teacher_weight', 0.0)
        cache = self.head.pgsma_cache
        if cache is None or cache.get('match') is None:
            raise RuntimeError('Run a native training forward before semantic_losses')
        weights = [(m.weight, m.bias) for m in self.head.cls_preds]
        return self.semantic.semantic_losses(cache['features'], weights,
                     cache['native_cls'], cache['native_obj'],
                     cache['match'], seed=seed, correction_mode=correction_mode, **kwargs)

    def set_inference_mode(self, mode='pgsma'):
        if mode not in ('pgsma','native','memory'):
            raise ValueError(mode)
        self.head.inference_semantic_mode = mode

    def set_meta_training_state(self, epoch, max_epochs):
        self.head.set_epoch_state(epoch, max_epochs)
