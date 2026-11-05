"""Isolated, auditable PyTorch *research prototype* for vNext.2 Safe Spatial TTT.

Not a drop-in YOLOX patch and not a validated DG model. It specifically tests:
  - masking before ALL spatial processing / normalization (no direct hidden-value leak),
  - compact per-image fast state with local-context spatial correction,
  - first-principles per-image gradients (no batch-coupled fast state),
  - true detection->inner-objective meta-gradient when create_graph=True,
  - explicit native identity fallback and per-site feature perturbation budget,
  - no persistent parameter/BN/EMA changes at inference.

Requires PyTorch 2.x. This module is intentionally standalone and CUDA agnostic.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def checkerboard_mask(x: torch.Tensor, phase: int = 0) -> torch.Tensor:
    """Source-independent 1/4 pattern: avoids target-informed masking leakage.

    Returns [B,1,H,W], one masked pixel in each 2x2 grid. H,W >=2.
    The deterministic phase MUST be fixed at inference; training may cover all
    phases but must audit that discrepancy separately.
    """
    if x.ndim != 4:
        raise ValueError("expected [B,C,H,W]")
    b, _, h, w = x.shape
    if h < 2 or w < 2:
        return torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
    yy = torch.arange(h, device=x.device).reshape(h, 1)
    xx = torch.arange(w, device=x.device).reshape(1, w)
    yy_phase = int(phase) // 2
    xx_phase = int(phase) % 2
    mask = (((yy % 2) == yy_phase) & ((xx % 2) == xx_phase)).to(x.dtype)
    return mask.reshape(1, 1, h, w).expand(b, 1, -1, -1)


class SpatialBasisAdapter(nn.Module):
    """Cheap local/multiradius basis conditioned by compact fast state z.

    Native feature identity path is untouched. A z=0 state is EXACT identity.
    A per-position RMS bound limits the magnitude of any correction, but cannot
    prove semantic non-regression under unlabeled unseen conditions.
    """
    def __init__(self, channels: int, rank: int = 8, relative_budget: float = 0.10,
                 basis_gain: float = 1.0):
        super().__init__()
        if channels < 2 or rank < 1 or not (0 < relative_budget <= 1):
            raise ValueError("invalid channels/rank/relative_budget")
        self.channels, self.rank = channels, rank
        self.relative_budget = float(relative_budget)
        if basis_gain <= 0 or not math.isfinite(basis_gain):
            raise ValueError("basis_gain must be positive finite")
        self.basis_gain = float(basis_gain)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dwd2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2,
                               groups=channels, bias=False)
        self.channel_to_basis = nn.Conv2d(channels, rank, 1, bias=False)
        self.basis_to_channel = nn.Conv2d(rank, channels, 1, bias=False)
        self.gate = nn.Conv2d(channels, 1, 1)
        nn.init.constant_(self.gate.bias, -2.0)
        # This weight must NOT be zero: otherwise d feature/d z vanishes and
        # the initial meta-gradient cannot train the SSL direction.
        nn.init.normal_(self.basis_to_channel.weight, std=0.03)

    def forward(self, features: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.shape != (features.shape[0], self.rank):
            raise ValueError("per-image z must have shape [B,rank]")
        norm = F.group_norm(features, math.gcd(self.channels, 8), eps=1e-5)
        local = F.silu(self.dw3(norm) + self.dwd2(norm))
        basis = self.channel_to_basis(local)
        gated_basis = basis * z.tanh().reshape(-1, self.rank, 1, 1)
        proposed = self.basis_to_channel(gated_basis) * self.basis_gain
        proposed = proposed * torch.sigmoid(self.gate(norm))
        # Spatially local pointwise feature-energy trust region. No single
        # spatial cell can exceed relative_budget times local feature RMS
        # plus a bounded numerical floor.
        native_rms = features.square().mean(1, keepdim=True).add(1e-8).sqrt()
        delta_rms = proposed.square().mean(1, keepdim=True).add(1e-12).sqrt()
        per_site_limit = self.relative_budget * (native_rms + 0.01)
        # Strict <= bound; no clamp of native feature path.
        gain = torch.clamp(per_site_limit / delta_rms, max=1.0)
        return features + proposed * gain


class NeighborPredictor(nn.Module):
    """Spatial masked feature predictor. Never pointwise-only."""
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, masked_input: torch.Tensor) -> torch.Tensor:
        return self.layers(masked_input)


class SafeSpatialTTT(nn.Module):
    """Episode-local, compact-state TTT with differentiable alignment path.

    The online/inner objective needs no labels and uses only the current image
    features. Supervised labels enter the OUTER training loss through a separate
    detector; for meta-training, differentiating through g_z is mandatory.

    Every operation in this module is sample-local. The module contains no
    BatchNorm, mutable test-time state, or cross-image statistic.
    """
    def __init__(self, channels: int, rank: int = 8,
                 relative_budget: float = 0.10, max_state_step: float = 0.2,
                 gradient_floor: float = 1e-4, step_policy: str = "legacy_floor",
                 min_signal_norm: float = 1e-8, basis_gain: float = 1.0):
        super().__init__()
        if not (max_state_step > 0 and gradient_floor > 0):
            raise ValueError("step/floor must be positive")
        self.channels, self.rank = channels, rank
        self.adapter = SpatialBasisAdapter(channels, rank, relative_budget, basis_gain=basis_gain)
        self.predictor = NeighborPredictor(channels)
        self.mask_token = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.step_logit = nn.Parameter(torch.tensor(-0.5))
        self.max_state_step = float(max_state_step)
        self.gradient_floor = float(gradient_floor)
        if step_policy not in ("legacy_floor", "normalized"):
            raise ValueError("Unknown fast-state step policy")
        if min_signal_norm <= 0:
            raise ValueError("min_signal_norm must be positive")
        self.step_policy = step_policy
        self.min_signal_norm = float(min_signal_norm)

    def compute_step(self, grad_z: torch.Tensor) -> torch.Tensor:
        """Bounded per-image step, with an explicit zero-signal abstention policy.

        The old absolute 1e-4 floor suppressed legitimate ~1e-6 gradients.
        Normalized mode is scale-independent above a conservative, source-only
        signal threshold, but can amplify misaligned gradients. Therefore it
        is a research ablation, NOT automatically a safe inference default.
        """
        norm=grad_z.norm(dim=1,keepdim=True)
        strength=self.max_state_step*torch.sigmoid(self.step_logit)
        if self.step_policy == 'legacy_floor':
            return -strength*grad_z/(norm+self.gradient_floor)
        valid=(norm>=self.min_signal_norm).to(dtype=grad_z.dtype)
        return -strength*grad_z/norm.clamp_min(self.min_signal_norm)*valid

    def masked_view(self, native: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask BEFORE basis convolutions and GroupNorm, preventing direct leak."""
        return native * (1.0 - mask) + self.mask_token * mask

    def ssl_loss(self, native: torch.Tensor, z: torch.Tensor,
                 mask: torch.Tensor | None = None, *, phase: int = 0):
        if mask is None:
            mask = checkerboard_mask(native, phase)
        # The target is the stop-gradient native representation of THIS image;
        # on adverse images, it is NOT a privileged clean-domain target.
        target = native.detach()
        if mask.sum().item() == 0:
            return z.sum(1) * 0.0
        corrupted = self.masked_view(target, mask)
        # Critically: no unmasked native spatial processing before masking.
        # This avoids the global-normalization/neighbor leakage in prior probe.
        predicted = self.predictor(self.adapter(corrupted, z))
        errors = (predicted - target).square()
        return (errors * mask).flatten(1).sum(1) / (
            mask.flatten(1).sum(1) * self.channels).clamp_min(1.0)

    def run_episode(self, native_features: torch.Tensor, *, meta_train: bool,
                    enabled: bool = True, phase: int = 0):
        """Return (features, diagnostics); no gradient outside at evaluation.

        `meta_train=True`: retains exact detector->g_ssl->predictor mixed
        derivative. `False`: frozen-EMA compatible in torch.no_grad().
        Never call inside torch.inference_mode() when enabled.
        """
        if not enabled:
            return native_features, {"native": True}
        if torch.is_inference_mode_enabled():
            raise RuntimeError("TTT requires no_grad(), NOT inference_mode()")
        if native_features.ndim != 4 or native_features.shape[1] != self.channels:
            raise ValueError("features must be [B,C,H,W] with configured C")
        if native_features.shape[0] == 0:
            return native_features, {"empty_batch": True}
        if next(self.parameters()).dtype != torch.float32:
            raise RuntimeError("Prototype expects float32 parameters; autocast the native model, not TTT weights")
        dev = native_features.device
        with torch.enable_grad(), torch.autocast(device_type=dev.type, enabled=False):
            source = native_features.detach().float()
            z0 = torch.zeros(source.shape[0], self.rank, device=dev,
                             dtype=torch.float32, requires_grad=True)
            mask = checkerboard_mask(source, phase)
            if mask.sum().item() == 0:
                return native_features, {"no_valid_mask_sites": True}
            losses = self.ssl_loss(source, z0, mask)
            if not torch.isfinite(losses).all():
                if meta_train:
                    raise FloatingPointError("Nonfinite inner SSL loss")
                return native_features, {"nonfinite_inner_loss": True}
            # Sum of per-image losses, NOT batch mean (preserves per-image step).
            grad_z, = torch.autograd.grad(losses.sum(), z0,
                                          create_graph=meta_train)
            if not torch.isfinite(grad_z).all():
                if meta_train:
                    raise FloatingPointError("Nonfinite inner fast-state gradients")
                return native_features, {"nonfinite_inner_gradient": True}
            # Per-example bounded update; no multiplication by batch size.
            norm = grad_z.norm(dim=1, keepdim=True)
            step = self.compute_step(grad_z)
            z1 = z0 + step
            # The task path should retain native gradient in meta-training;
            # the SSL state/target are computed from a detached native copy.
            adapted = self.adapter(native_features.float(), z1)
            if not torch.isfinite(adapted).all():
                if meta_train:
                    raise FloatingPointError("Nonfinite adapted features")
                return native_features, {"nonfinite_adapted": True}
            if not meta_train:
                adapted = adapted.detach()
            # The detector consumes the ORIGINAL dtype; especially under
            # FP16 autocast, small FP32 corrections may quantize to zero.
            adapted_cast = adapted.to(dtype=native_features.dtype)
            stats: dict[str, Any] = {
                "ssl_initial": losses.detach(),
                "gradient_norm": norm.detach(),
                "fast_state_norm": z1.norm(dim=1).detach(),
                "step_policy": self.step_policy,
                "low_signal_abstention": (norm[:,0] < self.min_signal_norm).detach() if self.step_policy == "normalized" else torch.zeros_like(norm[:,0], dtype=torch.bool),
                "relative_feature_change_precast": ((adapted.detach() - native_features.detach().float())
                    .square().flatten(1).mean(1).sqrt()
                    / native_features.detach().float().square().flatten(1).mean(1).add(1e-8).sqrt()),
                "postcast_changed_fraction": ((adapted_cast.detach() != native_features.detach())
                    .float().flatten(1).mean(1)),
                "relative_feature_change_postcast": ((adapted_cast.detach().float() - native_features.detach().float())
                    .square().flatten(1).mean(1).sqrt()
                    / native_features.detach().float().square().flatten(1).mean(1).add(1e-8).sqrt()),
            }
        return adapted_cast, stats


def alignment_surrogate(g_ssl: torch.Tensor, g_detector: torch.Tensor,
                        eps: float = 1e-20, min_gradient_norm: float = 1e-10) -> torch.Tensor:
    """Optional lower-memory surrogate for alignment; detector gradient fixed.

    g_ssl MUST have create_graph=True. g_detector should come from a fixed
    supervised target assignment for a matched native/adapted example. Stop
    gradient through g_detector to avoid differentiating detector's Hessian.
    """
    if g_ssl.shape != g_detector.shape:
        raise ValueError("gradient shapes mismatch")
    target = g_detector.detach()
    dot = (g_ssl * target).sum(1)
    n_ssl = g_ssl.norm(dim=1)
    n_det = target.norm(dim=1)
    denom = (n_ssl * n_det).clamp_min(eps)
    # The original TTT gradients can be O(1e-5). Do not accidentally ignore
    # them through an overly large default sqrt(eps) cutoff. Record/validate
    # tiny-gradient behavior on real model before choosing a production guard.
    valid = (n_ssl > min_gradient_norm) & (n_det > min_gradient_norm)
    return torch.where(valid, -dot / denom, torch.zeros_like(dot)).mean()
