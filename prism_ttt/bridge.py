"""YOLOX vNext.2 bridge. The native checkpoint and source remain unmodified.

Research-only instance-local monkeypatch ensures legacy dark2 TTT is OFF. The
single-process context is not reentrant/thread-safe; do not deploy as a server
or run DataParallel with this bridge. A permanent explicit native-only interface
is preferable before production integration.
"""
from __future__ import annotations
import torch
from torch import nn
try:
    from .legacy_router import Dark2AuditRouter
except ImportError:
    raise

class PRISMYOLOX(nn.Module):
    def __init__(self,native:nn.Module,sidecar:nn.Module,mode:str='off',native_amp:bool=False):
        super().__init__();self.native=native;self.sidecar=sidecar
        self.mode=mode;self.native_amp=bool(native_amp)
        dev=next(native.parameters()).device
        self.register_buffer('accepted_levels',torch.zeros(3,device=dev),persistent=False)
        self.register_buffer('images_seen',torch.zeros(1,device=dev),persistent=False)
        if not hasattr(native,'backbone') or not hasattr(native,'head'):
            raise TypeError('Expect vNext.2 model with backbone and head')

    @property
    def head(self):return self.native.head

    @property
    def backbone(self):return self.native.backbone

    def native_features(self,x):
        if self.native.training or self.native.backbone.training:
            raise RuntimeError('Native YOLOX must remain eval()')
        with Dark2AuditRouter(self.native) as router:
            router.set_mode('native')
            with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                                enabled=self.native_amp and x.device.type=='cuda'):
                f=self.native.backbone(x,ttt_prob=0.0)
            router.require_one_hit(int(x.shape[0]))
        if not isinstance(f,(tuple,list)) or len(f)!=3:
            raise RuntimeError('Expected P3/P4/P5 native pyramid')
        self.sidecar._check(f)
        return tuple(f)

    def head_predictions(self,pyramids,images=None,labels=None):
        device=pyramids[0].device
        with torch.autocast(device_type=device.type,dtype=torch.float16,
                            enabled=self.native_amp and device.type=='cuda'):
            if labels is None:return self.native.head(pyramids)
            return self.native.head(pyramids,labels,images)

    def forward(self,x,targets=None,*,mode=None,meta_train=False,require_calibration=True):
        mode=mode or self.mode
        if mode not in ('off','on','auto'):raise ValueError('Unsupported PRISM mode')
        # The detector's gradient to feature input is needed for meta training,
        # but never to its frozen native parameters.
        with torch.no_grad():
            features=self.native_features(x)
        if mode=='off':
            adapted=features;stats=None
        else:
            adapted,stats=self.sidecar(features,mode=mode,meta_train=meta_train,
                                   require_calibration=require_calibration)
        if not meta_train:
            with torch.no_grad():
                self.images_seen.add_(float(x.shape[0]))
                if stats is not None and stats['applied'] is not None:
                    self.accepted_levels.add_(stats['applied'].detach().float().sum(0))
        return self.head_predictions(adapted,images=x if targets is not None else None,labels=targets)
