"""Reversible instance-local dark2 bypass for vNext.2 audit only.

The current residual-TTT implementation enters via _forward_residual_ttt;
the older GN/functional_call snapshot enters via forward. Both are supported
as isolated *read-only audit* routes without overwriting any original source.

This is intentionally NOT a production deployment integration. Refuses an
unknown stage signature and checks that exactly one native stage invocation
occurred per audited head pass (model.eval(), no training subbatch split).
"""
from __future__ import annotations
from types import MethodType

import torch


class Dark2AuditRouter:
    def __init__(self, model, safe_ttt=None):
        if not hasattr(model,'backbone') or not hasattr(model.backbone,'backbone'):
            raise RuntimeError('Expected YOLOPAFPN -> CSPDarknet hierarchy')
        dark2 = model.backbone.backbone.dark2
        if not hasattr(dark2,'backbone_stage'):
            raise RuntimeError('Requires dark2.backbone_stage; this YOLOX layout is unsupported')
        self.stage=dark2
        self.safe=safe_ttt
        self.mode='native'
        self.z=None
        self.meta_train=False
        self.last_native=None
        self.last_stats=None
        self.hit=0
        self.old={}

    def set_mode(self, mode, *, z=None, meta_train=False):
        if mode not in ('native','fixed_z','episode'):
            raise ValueError(f'Unknown TTT audit mode {mode}')
        if mode!='native' and self.safe is None:
            raise RuntimeError('Missing SafeSpatialTTT stage for adapted mode')
        if mode=='fixed_z' and z is None:
            raise RuntimeError('fixed_z requires a differentiable compact state')
        self.mode=mode
        self.z=z
        self.meta_train=bool(meta_train)
        self.last_native=None
        self.last_stats=None
        self.hit=0

    def dispatch(self, stage, x_in, run_ttt=True, precomputed_feat=None):
        if precomputed_feat is not None:
            base_feat=precomputed_feat
        else:
            if x_in is None:raise RuntimeError('No native dark2 input or precomputed features')
            base_feat=stage.backbone_stage(x_in)
        if not isinstance(base_feat,torch.Tensor) or base_feat.ndim!=4:
            raise RuntimeError('Unsupported native dark2 output')
        self.hit+=1
        self.last_native=base_feat
        if self.mode=='native':return base_feat
        if self.mode=='fixed_z':
            if self.z.shape!=(base_feat.shape[0],self.safe.rank):
                raise RuntimeError('Compact state batch mismatch')
            with torch.autocast(device_type=base_feat.device.type, enabled=False):
                return self.safe.adapter(base_feat.float(), self.z).to(dtype=base_feat.dtype)
        output,self.last_stats=self.safe.run_episode(base_feat,meta_train=self.meta_train,enabled=True)
        return output

    def __enter__(self):
        stage=self.stage
        if hasattr(stage,'_forward_residual_ttt'):
            names=['_forward_residual_ttt']
        else:
            names=['forward']
            if hasattr(stage,'forward_from_features'):
                names.append('forward_from_features')
        router=self
        def replace_residual(obj,x_in,run_ttt=True,precomputed_feat=None):
            return router.dispatch(obj,x_in,run_ttt,precomputed_feat)
        def replace_forward(obj,x,run_ttt=True):
            return router.dispatch(obj,x,run_ttt)
        def replace_features(obj,native_features,run_ttt=True):
            return router.dispatch(obj,None,run_ttt,precomputed_feat=native_features)
        for name in names:
            self.old[name]=(name in stage.__dict__, stage.__dict__.get(name))
            func=(replace_residual if name=='_forward_residual_ttt' else
                  replace_features if name=='forward_from_features' else replace_forward)
            setattr(stage,name,MethodType(func,stage))
        return self

    def require_one_hit(self, batch):
        if self.hit!=1 or self.last_native is None or self.last_native.shape[0]!=batch:
            raise RuntimeError(f'Dark2 audit intercepted {self.hit} calls; expected exactly one full-batch native dark2 call. '
                               'Source may have different forwarding/subbatch semantics; refusing results.')

    def __exit__(self, *_):
        for name,(was_set,old_val) in self.old.items():
            if was_set:
                self.stage.__dict__[name]=old_val
            else:
                self.stage.__dict__.pop(name,None)
        self.old.clear()
