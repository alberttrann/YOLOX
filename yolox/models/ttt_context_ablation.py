"""Opt-in vNext.2 spatial-gradient-route TTT experiment.

This module never modifies existing vNext.2 files or checkpoint keys.  A new
experiment installs an instance-local forward override on the existing dark2
TTTAdaptiveStage.  All modes use the same weights and native CSPDarknet.

CAUTION: changing the reconstruction objective requires re-training; untrained
inference against an old checkpoint is only an exploratory counterfactual.
"""
import math
from types import MethodType

import torch
import torch.nn.functional as F

_MODES = {'token_mse', 'token_balanced', 'context_mse', 'context_balanced'}


def contextual_corruption(observed, mask, mask_token, *, use_context):
    """Mask original features, optionally use *only visible* local neighbors.

    No trainable parameters; same 3x3 kernel for all classes/conditions/scales.
    A site lacking visible neighbors retains the learned mask token.
    The input feature tensor is NCHW and mask is [N,1,H,W].
    """
    if observed.ndim != 4 or mask.shape != (observed.shape[0], 1, *observed.shape[-2:]):
        raise ValueError('Expected NCHW observations and N1HW binary mask')
    visible = 1.0 - mask
    if not use_context:
        return observed * visible + mask_token * mask, visible
    sums = F.avg_pool2d(observed * visible, 3, stride=1, padding=1)
    counts = F.avg_pool2d(visible, 3, stride=1, padding=1)
    context = sums / counts.clamp_min(1e-6)
    context = torch.where(counts > 0, context, mask_token)
    return observed * visible + context * mask, visible


def per_image_region_loss(prediction, target, mask, *, balanced):
    """Returns per-image reconstruction losses (not averaged over batch).

    Original mode uses a full CxHxW mean. Balanced mode gives equal weight to
    per-location masked and unmasked MSE, independent of the mask coverage.
    """
    error = (prediction.float() - target.float()).square()
    if not balanced:
        return error.flatten(1).mean(1)
    masked = mask.expand_as(error)
    visible = 1.0 - masked
    lm = (error * masked).flatten(1).sum(1) / masked.flatten(1).sum(1).clamp_min(1.0)
    lv = (error * visible).flatten(1).sum(1) / visible.flatten(1).sum(1).clamp_min(1.0)
    return (lm + lv) * 0.5


def _contextual_forward(self, x_in, run_ttt=True, precomputed_feat=None):
    """First-order TTT with per-image contextual reconstruction gradients.

    Native dark2 computed once; inference fast weights are ephemeral; training
    preserves native detection gradients and existing projector-only outer loss.
    """
    base_feat = (self.backbone_stage(x_in) if precomputed_feat is None
                 else precomputed_feat)
    if not run_ttt:
        return base_feat
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Contextual TTT requires torch.no_grad(), not torch.inference_mode()')
    mode = getattr(self, 'ttt_context_mode', None)
    if mode not in _MODES:
        raise RuntimeError(f'Unsupported TTT context mode: {mode!r}')
    balanced = mode.endswith('balanced')
    use_context = mode.startswith('context')
    training_meta = bool(self.training)
    with torch.enable_grad(), torch.autocast(device_type=base_feat.device.type, enabled=False):
        feat = base_feat.float()
        source = feat.detach()
        B = source.shape[0]
        if B == 0:
            return base_feat
        with torch.no_grad():
            target = self._get_noise_clipped_target(source)
            mask = self._get_relative_contrast_mask(target, self.mask_ratio).float()
            normalized = F.group_norm(source, 8, eps=1e-5)
            noise = (torch.randn_like(source) * self.noise_std
                     if training_meta else torch.zeros_like(source))
        if training_meta:
            # Non-leaf, per-image expanded views preserve outer meta-gradients.
            fast_s = self.residual_scale.expand(B, -1, -1, -1)
            fast_b = self.residual_bias.expand(B, -1, -1, -1)
        else:
            # Frozen EMA params cannot be differentiated; create transient leaves.
            fast_s = self.residual_scale.detach().float().expand(B, -1, -1, -1).clone().requires_grad_(True)
            fast_b = self.residual_bias.detach().float().expand(B, -1, -1, -1).clone().requires_grad_(True)

        def residual(s, b, norm):
            return self.residual_gain * (s.tanh() * norm + b.tanh())

        restored = source + residual(fast_s, fast_b, normalized)
        token = self.projector.mask_token.float() if training_meta else self.projector.mask_token.detach().float()
        corrupted, _ = contextual_corruption(restored + noise, mask, token, use_context=use_context)
        predicted = self.projector(corrupted)
        per_image_loss = per_image_region_loss(predicted, target, mask, balanced=balanced)
        g_s, g_b = torch.autograd.grad(per_image_loss.sum(), (fast_s, fast_b), create_graph=False)
        if not torch.isfinite(g_s).all() or not torch.isfinite(g_b).all():
            raise FloatingPointError('Non-finite contextual TTT fast-weight gradients')
        grad_norm = torch.sqrt(g_s.detach().square().flatten(1).sum(1)
                               + g_b.detach().square().flatten(1).sum(1) + 1e-12)
        clip = (self.residual_grad_clip / grad_norm).clamp(max=1.0).view(B, 1, 1, 1)
        progress = min(1.0, max(0.0, self.current_epoch / max(1, self.max_epochs)))
        lr_max = self.min_meta_lr + (self.base_max_lr - self.min_meta_lr) * (
            0.5 * (1.0 + math.cos(math.pi * progress)))
        lr_s = self.min_meta_lr + (lr_max-self.min_meta_lr) * torch.sigmoid(
            self.ttt_lrs['residual_scale'])
        lr_b = self.min_meta_lr + (lr_max-self.min_meta_lr) * torch.sigmoid(
            self.ttt_lrs['residual_bias'])
        new_s = fast_s - lr_s * (g_s.detach() * clip)
        new_b = fast_b - lr_b * (g_b.detach() * clip)
        out = feat + residual(new_s, new_b, F.group_norm(feat, 8, eps=1e-5))
        if training_meta:
            # Preserve the original first-order projector-only reconstruction
            # gradient boundary. The native detector receives task gradients.
            outer_input, _ = contextual_corruption(
                out.detach() + noise, mask, token, use_context=use_context)
            outer_pred = self.projector(outer_input)
            self.current_rec_loss = per_image_region_loss(
                outer_pred, target, mask, balanced=balanced).mean()
    if not training_meta:
        out = out.detach()
    return out.to(dtype=base_feat.dtype)


def install_contextual_ttt(model, mode='context_mse'):
    """Patch only this model instance; preserves all parameters/state_dict keys.

    Install immediately after Exp.get_model(), prior to creating optimizer/EMA.
    The bound method survives deepcopy as a binding to the cloned stage.
    """
    if mode not in _MODES:
        raise ValueError(f'Unknown mode {mode!r}; choose {sorted(_MODES)}')
    stage = model.backbone.backbone.dark2
    if not getattr(stage, 'residual_adapter', False):
        raise RuntimeError('Requires pretrained-compatible residual TTT in dark2')
    if not hasattr(stage, '_forward_residual_ttt_per_image_eval'):
        raise RuntimeError('Apply the existing vNext.2 per-image TTT correctness fix first')
    if hasattr(stage, 'ttt_context_mode'):
        if stage.ttt_context_mode != mode:
            raise RuntimeError('TTT context mode cannot change on the same model instance')
        return model
    stage.ttt_context_mode = mode
    stage._forward_residual_ttt = MethodType(_contextual_forward, stage)
    return model
