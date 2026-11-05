"""Source-label-only gradient audit for compact latent TTT.

`task_losses(features)` must return a dict of SCALAR losses computed using
*fixed native assignments / teacher targets*. Never let YOLOX SimOTA change
assignment between native and adapted probes. Can be used on synthetic tasks
until integrated into a real YOLOX training head.

No parameter is optimized or mutated. Function calls may contain stochastic
layers; evaluate/freeze BN/dropout yourself before using this in YOLOX.
"""
from __future__ import annotations

import torch
from torch import nn


def _stats(g_ssl: torch.Tensor, g_task: torch.Tensor):
    # Per-image compact-state gradient, with no batch mixing.
    a = g_ssl.detach().float()
    b = g_task.detach().float()
    dot = (a*b).sum(dim=1)
    g1 = a.norm(dim=1)
    g2 = b.norm(dim=1)
    cos = dot / (g1*g2).clamp_min(1e-12)
    return {'ssl_norm':g1.cpu().tolist(), 'task_norm':g2.cpu().tolist(),
            'dot':dot.cpu().tolist(), 'cosine':cos.cpu().tolist()}


def audit_compact_gradient(ttt: nn.Module, native_features: torch.Tensor,
                           task_losses, *, phase=0):
    """Measure SSL vs each supervised detection-component direction.

    task_losses: callable adapted_features -> {name:scalar_loss}
    Native and adapted evaluation must share frozen target assignments.
    Use ONLY source-clean labeled training data for this probe; there are no
    adverse-condition labels in the inference interface.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Cannot audit gradients inside inference_mode')
    with torch.enable_grad():
        native=native_features.detach().float()
        batch=native.shape[0]
        z0=torch.zeros(batch, ttt.rank, device=native.device,
                       dtype=torch.float32,requires_grad=True)
        loss_ssl=ttt.ssl_loss(native,z0,phase=phase)
        g_ssl,=torch.autograd.grad(loss_ssl.sum(),z0,retain_graph=True)
        base=ttt.adapter(native,z0)
        losses_base=task_losses(base)
        if not isinstance(losses_base,dict) or not losses_base:
            raise TypeError('task_losses() must return nonempty dict of scalar losses')
        metric={}
        for name, loss in losses_base.items():
            if not isinstance(loss,torch.Tensor) or loss.ndim!=0:
                raise ValueError(f'{name} must be scalar')
            if not loss.requires_grad:
                raise ValueError(f'{name} is detached from fast state')
            g_task,=torch.autograd.grad(loss,z0,retain_graph=True)
            metric[name] = _stats(g_ssl,g_task)
        with torch.no_grad():
            state_norm=g_ssl.norm(dim=1,keepdim=True)
            strength=ttt.max_state_step*torch.sigmoid(ttt.step_logit.detach())
            strength_value = float(strength.detach())
            step=-strength*g_ssl/(state_norm+ttt.gradient_floor)
            adapted=ttt.adapter(native,step)
            losses_after=task_losses(adapted)
            for name,loss in losses_base.items():
                metric[name]['loss_before']=float(loss.detach())
                metric[name]['loss_after']=float(losses_after[name].detach())
                metric[name]['loss_delta']=metric[name]['loss_after']-metric[name]['loss_before']
                metric[name]['linear_predicted_delta']=sum(
                    d*(-strength_value)/(n+ttt.gradient_floor)
                    for d,n in zip(metric[name]['dot'],state_norm.squeeze(1).detach().cpu().tolist()))
        return {'by_task':metric,'step_norm':step.norm(dim=1).detach().cpu().tolist(),
                'ssl_loss':loss_ssl.detach().cpu().tolist(),
                'note':'Matched assigned labels are required for actual YOLOX task losses; no optimizer step.'}
