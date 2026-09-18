#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2024-2026 Astera Institute & Moonshot AI.
# Tailored for TDE-YOLOX v3.1: Polar Decomposition on Manifolds & Combined Multi-Optimizer

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz Polar Matrix Decomposition (5th-Order Polynomial).
    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315.
    Includes the Tall-Matrix Transpose-Gram Trick for 576 x 128 manifolds.
    """
    assert len(G.shape) == 2, f"Muon requires 2D matrices, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.float()
    X /= (X.norm() + eps)

    # Transpose-Gram Trick: Operates on 128x128 full-rank Gram matrix instead of 576x576 singular matrix
    is_tall = X.size(0) > X.size(1)
    if is_tall:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if is_tall:
        X = X.T

    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon (Momentum Orthogonalized by Newton-Schulz) Optimizer.
    Applied strictly to 2D matrices (Prototypes, Linear Projections, Manifold Decoders).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                state["step"] += 1

                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                buf.mul_(momentum).add_(g)
                m = g.add(buf, alpha=momentum) if nesterov else buf

                orig_shape = p.shape
                if len(orig_shape) == 4 and orig_shape[2:] == (1, 1):
                    m_2d = m.view(orig_shape[0], orig_shape[1])
                elif len(orig_shape) == 2:
                    m_2d = m
                else:
                    raise ValueError(f"Muon only supports 2D matrices or 1x1 convs. Got shape {orig_shape}.")

                # Polar decomposition step
                ortho_update = zeropower_via_newtonschulz5(m_2d, steps=ns_steps)

                if len(orig_shape) == 4:
                    ortho_update = ortho_update.view(orig_shape)

                p.add_(ortho_update, alpha=-lr)

                # Post-Step Riemannian Retraction onto Unit Hypersphere S^{d-1}
                if getattr(p, "is_hypersphere", False) or (len(orig_shape) == 2 and orig_shape[1] == 128 and orig_shape[0] % 9 == 0):
                    p.data.copy_(F.normalize(p.data, p=2, dim=-1, eps=1e-5))

        return loss


class CombinedOptimizer:
    """
    Composite Multi-Engine Optimizer Wrapper.
    Unifies Muon, Momentum SGD, and Meta-SGD under standard PyTorch Optimizer API.
    Guarantees seamless compatibility with torch.cuda.amp.GradScaler.
    """
    def __init__(self, optimizers):
        self.optimizers = optimizers
        # Aggregate all parameter groups for AMP GradScaler discovery
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            opt.step(closure=closure)
        return loss

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        for opt, s in zip(self.optimizers, state_dict):
            opt.load_state_dict(s)