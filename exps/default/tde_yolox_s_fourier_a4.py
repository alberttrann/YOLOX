"""A4: task-grounded, identity-preserving tdevNext.3+local Fourier model."""
from torch import nn
from tde_yolox_s_fourier import Exp as FourierExp
from tde_yolox_s import Exp as SourceExp
from yolox.models.a4_head import A4Head
from yolox.models.a4_backbone import prepare_a4_neck
from yolox.models.a4_detector import A4Detector


class Exp(FourierExp):
    def __init__(self):
        super().__init__()
        self.exp_name='tdevnext3_a4_fourier_grounded'
        self.fourier_variant='a4'
        self.a4_filter_basis='dct'
        self.a4_architecture='grounded_prototype_only_aux_and_identity_spi'
        self.enable_ttt=False
        self.enable_legacy_ttt=False
        self.enable_rec_aux=False
        self.enable_proto_aux=False
        self.enable_intra_aux=False
        self.enable_afford_aux=False
        self.enable_memory_aux=True  # foreground-matched prototype-only classification
        self.enable_memory=True
        self.enable_spi=True

    def get_model(self):
        if getattr(self,'model',None) is not None:
            return self.model
        self._preflight()
        source=SourceExp.get_model(self)
        neck=prepare_a4_neck(source.backbone)
        head=A4Head(self.num_classes,width=self.width,
                    in_channels=(256,512,1024),act=self.act,
                    loss_type='oiou',basis=self.a4_filter_basis)
        for layer in head.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.eps=1e-3;layer.momentum=0.03
        self.model=A4Detector(neck,head)
        self.model.max_epochs=int(self.max_epoch)
        return self.model
