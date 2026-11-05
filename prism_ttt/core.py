"""PRISM-TTT v1: versioned multiscale, source-meta-trainable, episodic TTT.

No GT, pseudo-labels, mutable memory, or target labels are used by `propose`.
Call with torch.no_grad() (not inference_mode()) during evaluation. All three
pyramids have separate spatial bases and a 20-D per-image fast state.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION='prism-ttt-v1.0'
LEVELS=('P3','P4','P5')


def _finite(x: Tensor)->Tensor:
    return torch.isfinite(x).flatten(1).all(1)


def _rms(x:Tensor, dim:int=1)->Tensor:
    return x.square().mean(dim=dim,keepdim=True).add(1e-12).sqrt()


def _instance_norm(x:Tensor)->Tensor:
    # Per-pixel channel-vector normalization preserves local relations.
    return x / _rms(x,1).clamp_min(1e-3)


def relations(x:Tensor)->Tensor:
    """Local feature cosine affinities, spatially registered, no GT."""
    u=F.normalize(x,dim=1,eps=1e-5)
    h=(u[:,:,:,1:]*u[:,:,:,:-1]).sum(1,keepdim=True)
    v=(u[:,:,1:,:]*u[:,:,:-1,:]).sum(1,keepdim=True)
    return torch.stack((h.mean((2,3)),v.mean((2,3))),dim=-1).squeeze(1)


def robust_difference(a:Tensor,b:Tensor,delta:float=.25)->Tensor:
    d=a-b
    return (delta*delta*(torch.sqrt(1+(d/delta).square())-1)).flatten(1).mean(1)


class PyramidBasis(nn.Module):
    def __init__(self,channels:int,rank:int=8,budget:float=.06):
        super().__init__()
        if channels<8 or rank<1 or not 0<budget<=.25:raise ValueError('Invalid basis specification')
        self.channels=channels;self.rank=rank;self.budget=budget
        gn=math.gcd(channels,8)
        self.gn=nn.GroupNorm(gn,channels)
        self.dw_local=nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        self.dw_dilated=nn.Conv2d(channels,channels,3,padding=2,dilation=2,groups=channels,bias=False)
        self.to_basis=nn.Conv2d(channels,rank,1,bias=False)
        self.to_channels=nn.Conv2d(rank,channels,1,bias=False)
        self.shared=nn.Linear(8,rank,bias=False)
        self.local=nn.Linear(4,rank,bias=False)
        # Never zero the decoder: zero state already makes the bypass exact.
        nn.init.normal_(self.to_channels.weight,mean=0,std=.04)

    def forward(self,x:Tensor,zg:Tensor,zs:Tensor)->Tensor:
        if x.ndim!=4 or x.shape[1]!=self.channels or zg.shape!=(x.shape[0],8) or zs.shape!=(x.shape[0],4):
            raise ValueError('Pyramid/state shape mismatch')
        n=self.gn(x)
        spatial=F.silu(self.dw_local(n)+self.dw_dilated(n))
        b=self.to_basis(spatial)
        coeff=torch.tanh(self.shared(zg)+self.local(zs))[:,:,None,None]
        proposed=self.to_channels(b*coeff)
        # Smooth pointwise RMS bound (native data path is untouched).
        native_rms=_rms(x)
        cap=self.budget*(native_rms+.01)
        proposed_rms=_rms(proposed)
        bounded=proposed*cap/torch.sqrt(cap.square()+proposed_rms.square()+1e-20)
        return x+bounded


class ScoreGate(nn.Module):
    """Source-trained, pre-head reliability policy. Uncalibrated auto fails closed."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(6,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-1.5)
        self.register_buffer('thresholds',torch.tensor([.80,.80,.80],dtype=torch.float32))
        self.register_buffer('calibrated',torch.tensor(False))

    def forward(self,signals:Tensor)->Tensor:
        if signals.ndim!=3 or signals.shape[1:]!=(3,6):raise ValueError('Expected [B,3,6]')
        return torch.sigmoid(self.net(signals.float())).squeeze(-1)


@dataclass
class Proposal:
    state:Tensor
    grad:Tensor
    ssl_before:Tensor
    ssl_after:Tensor
    proposed:tuple[Tensor,Tensor,Tensor]
    signals:Tensor
    valid:Tensor


class PRISMTTT(nn.Module):
    """One-step task-aligned spatial TTT with optional source-calibrated gate."""
    def __init__(self,channels:Sequence[int]=(128,256,512),rank:int=8,
                 budget:float=.06, max_step:float=.18,signal_floor:float=1e-9,
                 inner_style_strength:float=.06, relation_weight:float=.35):
        super().__init__()
        if len(channels)!=3 or max_step<=0 or signal_floor<=0:raise ValueError('Invalid configuration')
        self.channels=tuple(int(c) for c in channels)
        self.bases=nn.ModuleList(PyramidBasis(c,rank,budget) for c in channels)
        self.max_step=float(max_step)
        self.signal_floor=float(signal_floor)
        self.inner_style_strength=float(inner_style_strength)
        self.relation_weight=float(relation_weight)
        self.ssl_weight_net=nn.Sequential(nn.Linear(9,16),nn.SiLU(),nn.Linear(16,3))
        self.step_net=nn.Sequential(nn.Linear(9,12),nn.SiLU(),nn.Linear(12,1))
        nn.init.zeros_(self.step_net[-1].weight)
        nn.init.zeros_(self.step_net[-1].bias)
        self.gate=ScoreGate()
        # Initial value only for diagnostic/bootstrap. Offline source calibration
        # is REQUIRED before releasing on/auto evaluation results.
        self.register_buffer('gradient_scale',torch.full((20,),1e-5))
        self.register_buffer('gradient_calibrated',torch.tensor(False))

    def _check(self,features):
        if len(features)!=3:raise ValueError('Expected three head-facing pyramids')
        b=features[0].shape[0]
        if not all(isinstance(x,Tensor) and x.ndim==4 and x.shape[0]==b and x.shape[1]==c for x,c in zip(features,self.channels)):
            raise ValueError('P3/P4/P5 channel or batch mismatch')
        if next(self.parameters()).dtype!=torch.float32:raise RuntimeError('Keep sidecar parameters FP32')
        return b

    def adapt(self,features:Sequence[Tensor],z:Tensor)->tuple[Tensor,Tensor,Tensor]:
        if z.ndim!=2 or z.shape!=(features[0].shape[0],20):raise ValueError('Expected [B,20] fast state')
        zg=z[:,:8]
        return tuple(self.bases[i](f,zg,z[:,8+i*4:12+i*4]) for i,f in enumerate(features))

    def stats(self,features:Sequence[Tensor])->Tensor:
        # Three finite, source-only descriptors per level (no labels).
        per=[]
        for f in features:
            x=f.detach().float()
            per.extend([x.abs().mean((1,2,3)), x.flatten(1).std(1,unbiased=False),
                       (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean((1,2,3))])
        return torch.stack(per,dim=1)

    def _perturb(self,x:Tensor,sign:int)->Tensor:
        # Deterministic per-sample, strictly geometry-preserving feature copies.
        # No global RNG, image IDs or batch-dependent perturbation choices.
        b,c,h,w=x.shape;dev=x.device
        chan=torch.arange(c,device=dev,dtype=x.dtype)[None,:,None,None]
        sy=torch.linspace(-1,1,h,device=dev,dtype=x.dtype)[None,None,:,None]
        sx=torch.linspace(-1,1,w,device=dev,dtype=x.dtype)[None,None,None,:]
        channel_wave=torch.sin((chan+1)*.37)
        spatial_wave=torch.cos(math.pi*sy)*torch.cos(math.pi*sx*.5)
        amp=self.inner_style_strength
        return x*(1+sign*amp*(.6*channel_wave+.4*spatial_wave))

    def ssl(self,features:Sequence[Tensor],z:Tensor):
        # Fixed nonlearned native targets, no self-teacher target collapse.
        raw_stat=self.stats(features)
        logits=self.ssl_weight_net(raw_stat)
        # Weight floor 0.15, sum exactly 1, no collapse to all-zero weights.
        weights=.15+.55*logits.softmax(dim=-1)
        losses=[];relerr=[]
        for i,native in enumerate(features):
            src=native.detach().float()
            base_rel=relations(src).detach()
            target=_instance_norm(src).detach()
            a,b=(self._perturb(src,+1),self._perturb(src,-1))
            aa=self.bases[i](a,z[:,:8],z[:,8+i*4:12+i*4])
            bb=self.bases[i](b,z[:,:8],z[:,8+i*4:12+i*4])
            consistency=.5*(robust_difference(_instance_norm(aa),target)+robust_difference(_instance_norm(bb),target))
            re=.5*(robust_difference(relations(aa),base_rel)+robust_difference(relations(bb),base_rel))
            losses.append(consistency+self.relation_weight*re)
            relerr.append(re)
        L=torch.stack(losses,dim=1)
        return (weights*L).sum(1),L,torch.stack(relerr,dim=1),weights

    def source_scale_update(self,grad:Tensor,stats:Tensor)->Tensor:
        gain=torch.sigmoid(self.step_net(stats))
        # Not a unit-length normalized step; tiny gradients remain tiny.
        z=-self.max_step*gain*torch.tanh(grad/self.gradient_scale.clamp_min(1e-12))
        magnitude=z.norm(dim=1,keepdim=True)
        z=z*(self.max_step/magnitude.clamp_min(self.max_step)).clamp(max=1)
        return z

    def propose(self,features:Sequence[Tensor],*,meta_train:bool=False)->Proposal:
        if torch.is_inference_mode_enabled():raise RuntimeError('Use no_grad, not inference_mode, for episodic TTT')
        b=self._check(features)
        if b==0:raise ValueError('Empty batch')
        device=features[0].device
        with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
            fs=tuple(f.float() if meta_train else f.detach().float() for f in features)
            # Subset-finite handling: bad images get zeros, good images retain signal.
            finite=torch.stack([_finite(f) for f in fs],1).all(1)
            fs=tuple(torch.where(finite[:,None,None,None],f,torch.zeros_like(f)) for f in fs)
            z0=torch.zeros((b,20),device=device,dtype=torch.float32,requires_grad=True)
            before,per_level,relerr,w=self.ssl(fs,z0)
            if meta_train and not bool(torch.isfinite(before).all()):
                raise FloatingPointError('Nonfinite meta-training SSL')
            g,=torch.autograd.grad(before.sum(),z0,create_graph=meta_train)
            gfinite=_finite(g);norm=g.detach().norm(dim=1)
            valid=finite&gfinite&(norm>=self.signal_floor)
            grad=torch.where(valid[:,None],g,torch.zeros_like(g))
            state=self.source_scale_update(grad,self.stats(fs))
            state=torch.where(valid[:,None],state,torch.zeros_like(state))
            # Only the proposed samples carry correction; exact bypass of invalids.
            proposed=self.adapt(fs,state)
            after,per_after,rel_after,_=self.ssl(fs,state)
            changed=[];sig=[]
            for i,(native,new) in enumerate(zip(features,proposed)):
                native_f=native.detach().float()
                # Actual post-cast effect, not only unquantized FP32 proposal.
                new_cast=new.to(native.dtype).float()
                orig_safe=torch.where(finite[:,None,None,None],native_f,torch.zeros_like(native_f))
                rel_change=((new_cast-orig_safe).square().flatten(1).mean(1).sqrt()
                            /orig_safe.square().flatten(1).mean(1).add(1e-8).sqrt())
                improvement=(per_level[:,i]-per_after[:,i]).detach()
                sig.append(torch.stack((norm.detach().log1p(),improvement,
                  relerr[:,i].detach(),rel_after[:,i].detach(),rel_change.detach(),
                  per_level[:,i].detach()),dim=1))
                changed.append(new.to(native.dtype))
            signals=torch.stack(sig,dim=1)
            valid=valid&torch.isfinite(signals).flatten(1).all(1)
            safe=tuple(torch.where(valid[:,None,None,None],p,old) for p,old in zip(changed,features))
            return Proposal(state,grad,before,after,safe,signals,valid)

    def apply(self,features:Sequence[Tensor],proposal:Proposal,mode:str='auto',
              *,require_calibration:bool=True):
        if mode not in ('off','on','auto'):raise ValueError('mode off/on/auto')
        if mode=='off':return tuple(features),None
        if require_calibration and not bool(self.gradient_calibrated.item()):
            raise RuntimeError('Source gradient-scale calibration is required')
        if mode=='auto' and not bool(self.gate.calibrated.item()):
            # Fail-closed: never represent untrained gate as calibrated.
            return tuple(features),torch.zeros_like(proposal.signals[:,:,0])
        scores=self.gate(proposal.signals.detach())
        if mode=='on':r=proposal.valid[:,None].float().expand(-1,3)
        else:
            r=(scores>=self.gate.thresholds[None,:]).float()*proposal.valid[:,None].float()
        combined=tuple(torch.where((r[:,i,None,None,None]>0),
                  native+(prop-native)*r[:,i,None,None,None], native)
                       for i,(native,prop) in enumerate(zip(features,proposal.proposed)))
        return combined,r

    def forward(self,features:Sequence[Tensor],*,mode:str='auto',meta_train:bool=False,
                require_calibration:bool=True):
        if mode=='off':return tuple(features),{'mode':'off','native':True}
        proposal=self.propose(features,meta_train=meta_train)
        out,gates=self.apply(features,proposal,mode,require_calibration=require_calibration)
        stats={'mode':mode,'valid':proposal.valid.detach(),'g_norm':proposal.grad.detach().norm(dim=1),
               'state_norm':proposal.state.detach().norm(dim=1),'ssl_before':proposal.ssl_before.detach(),
               'ssl_after':proposal.ssl_after.detach(),'applied':gates.detach() if gates is not None else None}
        return out,stats

    @torch.no_grad()
    def source_calibrate_gradients(self,gradients:Tensor, *,quantile:float=.5):
        if gradients.ndim!=2 or gradients.shape[1]!=20 or gradients.shape[0]<16 or not 0<quantile<=1:
            raise ValueError('Need >=16 independent source-image gradients [N,20]')
        vals=gradients.abs().float()
        if not bool(torch.isfinite(vals).all()):raise ValueError('Nonfinite source gradient')
        estimate=torch.quantile(vals,quantile,dim=0).clamp_min(1e-8)
        self.gradient_scale.copy_(estimate.to(device=self.gradient_scale.device))
        self.gradient_calibrated.fill_(True)

    def verify_configuration(self,channels:Sequence[int],version:str=VERSION):
        if version!=VERSION or tuple(channels)!=self.channels:raise RuntimeError('PRISM version/channel mismatch')
