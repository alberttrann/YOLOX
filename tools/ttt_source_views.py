"""Deterministic, geometry-preserving clean-source photometric episodes.

No ACDC-specific settings, labels, or weather synthesis. Inputs are the actual
CHW/BNCHW float detector input. Source annotations are NEVER transformed.
The CUDA-side operation is deterministic given its recorded pure-Python spec.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AppearanceSpec:
    brightness: float
    contrast: float
    gamma: float
    channel_gains: tuple[float, float, float]
    blur_fraction: float
    strength: float
    seed: int
    index: int


def sample_appearance(seed: int, index: int, *, strength: float=1.0) -> AppearanceSpec:
    """Reproducible *source-only* nuisance changes; no global RNG mutation.

    Different from stochastic loader augmentations, this spec is purely a
    function of seed+image ID; pre/post training holdout uses identical pixels.
    """
    if not 0 <= strength <= 1.0 or not math.isfinite(strength):
        raise ValueError('strength must be finite in [0,1]')
    rng = random.Random((int(seed)*0x9E3779B1 + int(index)*0x85EBCA77) & ((1<<64)-1))
    amp = float(strength)
    # Clip-preserving ranges deliberately modest so class/box semantics hold.
    brightness = 1.0 + amp*rng.uniform(-0.18,0.18)
    contrast = 1.0 + amp*rng.uniform(-0.18,0.18)
    gamma = 1.0 + amp*rng.uniform(-0.15,0.15)
    gains=tuple(1.0 + amp*rng.uniform(-0.12,0.12) for _ in range(3))
    # A mild anti-aliasing / defocus component; opt-out on most episodes.
    blur_frac=amp*rng.uniform(0.08,0.22) if rng.random() < 0.25 else 0.0
    return AppearanceSpec(brightness,contrast,gamma,gains,blur_frac,amp,int(seed),int(index))


def _trailing_letterbox_pad(images: torch.Tensor)->torch.Tensor:
    """Only preserve trailing FULL rows/columns exactly equal to YOLOX pad=114.

    It never masks arbitrary interior pixels, objects, or image boundaries.
    If no full trailing border exists, the mask is entirely false. This is a
    conservative layout guard, not a general-purpose segmentation algorithm.
    """
    matching=(images.float()-114.0).abs().le(1e-4).all(dim=1)  # [B,H,W]
    rows=matching.all(dim=-1)
    cols=matching.all(dim=-2)
    bottom=rows.flip(-1).to(dtype=torch.int32).cumprod(-1).flip(-1).bool()
    right=cols.flip(-1).to(dtype=torch.int32).cumprod(-1).flip(-1).bool()
    mask=bottom[:,:,None] | right[:,None,:]
    return mask[:,None,:,:]


def apply_appearance(images:torch.Tensor,spec:AppearanceSpec)->torch.Tensor:
    """Input/output NCHW of identical shape, coords, device, dtype; no labels.

    For source images in 0..255, work in 0..1 then restore 0..255. Only
    sample-local channel/color/blur operations. No spatial resampling/cropping.
    Fail closed on nonstandard normalization instead of silently corrupting.
    """
    if images.ndim!=4 or images.shape[1]!=3 or not torch.is_floating_point(images):
        raise ValueError('Requires floating NCHW three-channel detector input')
    if not bool(torch.isfinite(images).all()):
        raise FloatingPointError('Nonfinite source input')
    lo,hi=float(images.min()),float(images.max())
    if lo < -1e-3 or hi > 255.01:
        raise ValueError('Expected raw source pixel domain 0..255 or 0..1; unsupported normalization')
    # Note: this pilot is for the established YOLOX source pipeline 0..255.
    # An all-dark image is ambiguous; refuse normalized shape guessing.
    if hi < 2.0:
        raise ValueError('Source image looks normalized (<=1); adapt the preprocessing protocol explicitly')
    pad_mask=_trailing_letterbox_pad(images)
    inp=images.float()/255.0
    out=inp*spec.brightness
    mean=out.mean(dim=(-2,-1),keepdim=True)
    out=(out-mean)*spec.contrast+mean
    gains=out.new_tensor(spec.channel_gains).reshape(1,3,1,1)
    out=out*gains
    out=out.clamp(0.0,1.0).pow(spec.gamma)
    if spec.blur_fraction>0:
        # Replicate padding preserves image geometry, avoiding zero boundaries.
        blurred=F.avg_pool2d(F.pad(out,(1,1,1,1),mode='replicate'),3,stride=1)
        out=out.lerp(blurred,spec.blur_fraction)
    out=(out*255.0).clamp(0.0,255.0).to(images.dtype)
    # Do not shift artificial YOLOX letterbox context into a faux weather cue.
    out=torch.where(pad_mask,images,out)
    if out.shape!=images.shape or not bool(torch.isfinite(out).all()):
        raise RuntimeError('Appearance perturbation broke shape/numerics')
    return out


def spec_dict(spec:AppearanceSpec)->dict:
    d=asdict(spec)
    d['channel_gains']=list(spec.channel_gains)
    return d


def image_fingerprint(images:torch.Tensor)->str:
    return hashlib.sha256(images.detach().to('cpu',dtype=torch.float32).contiguous().numpy().tobytes()).hexdigest()
