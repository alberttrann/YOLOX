"""Prototype-Grounded Semantic Manifold Alignment for YOLOX classification towers.

No test-time state mutation or autograd. Training losses are *semantic-only*;
caller MUST use a separate optimizer and detached native features, and validate
full native optimizer-step parity. Exact identity at initialized predictor output.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


def _norm(x, dim=1):
    # Unlike normalize(x,eps), zero input is explicitly invalid and will be
    # suppressed in the correction gate instead of posing as a sphere point.
    magnitude = torch.linalg.vector_norm(x.float(), dim=dim, keepdim=True)
    return x.float() / magnitude.clamp(min=1e-6), magnitude > 1e-6


class LocalPredictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.mix = nn.Conv2d(d, d, 1)
        self.out = nn.Conv2d(d, d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        return self.out(F.silu(self.mix(F.silu(self.dw(z)))))


class PGSMA(nn.Module):
    """Class-conditional semantic score differences, not a replacement detector.

    Inputs at the AFTER-classification-tower interface are `[B,C,H,W]`.
    `forward_levels` returns corrected *raw* logits, unchanged objectness and
    regression being the responsibility of the caller. All reductions and
    prototype similarities use FP32; no `[N,K,M,D]` broadcast is allocated.
    """
    def __init__(self, classes, channels=128, dim=64, modes=4, levels=3,
                 logit_budget=2.0, correction='direct'):
        super().__init__()
        if classes < 1 or dim < 2 or modes < 1 or not (0 < logit_budget < 20):
            raise ValueError('Invalid semantic dimensions or logit budget')
        if correction not in ('direct', 'retraction'):
            raise ValueError('correction must be direct or retraction')
        self.classes = int(classes)
        self.dim = int(dim)
        self.modes = int(modes)
        self.levels = int(levels)
        self.budget = float(logit_budget)
        self.correction = correction
        self.projectors = nn.ModuleList([nn.Conv2d(channels, dim, 1) for _ in range(levels)])
        self.teacher_projectors = copy.deepcopy(self.projectors)
        for p in self.teacher_projectors.parameters(): p.requires_grad_(False)
        self.predictors = nn.ModuleList([LocalPredictor(dim) for _ in range(levels)])
        # Fixed uniform component measures. Learned occupancy is a separate ablation.
        self.prototypes = nn.Parameter(F.normalize(torch.randn(classes, modes, dim), dim=-1))
        self.log_concentration = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
        self.log_gain = nn.Parameter(torch.zeros(levels))
        # Gate sees native cls, obj, prototype logits and movement evidence.
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(classes * 3 + 1, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
            for _ in range(levels)
        ])
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)  # sigmoid(0) = 0.5, NOT closed
        # Separate, trainable scalar for a *genuine* memory-only ablation.
        self.memory_gain = nn.Parameter(torch.zeros(levels))
        self.register_buffer('semantic_step', torch.tensor(0, dtype=torch.long))

    @property
    def concentration(self):
        return F.softplus(self.log_concentration.float()).clamp(min=1e-3, max=30.0)

    @torch.no_grad()
    def update_teacher(self, decay=0.99):
        for ema, src in zip(self.teacher_projectors.parameters(), self.projectors.parameters()):
            ema.lerp_(src, 1.0 - float(decay))
        for ema, src in zip(self.teacher_projectors.buffers(), self.projectors.buffers()):
            ema.copy_(src)

    def _prototype_scores(self, z, frozen=False):
        p = self.prototypes.detach() if frozen else self.prototypes
        p = F.normalize(p.float(), dim=-1).reshape(-1, self.dim, 1, 1)
        # conv2d [B,K*M,H,W], no NxKxMxD expansion
        sims = F.conv2d(z.float(), p) * (self.concentration.detach() if frozen else self.concentration)
        b, _, h, w = sims.shape
        scores = sims.reshape(b, self.classes, self.modes, h, w)
        return torch.logsumexp(scores, dim=2) - math.log(self.modes)

    def _project(self, i, h, frozen=False):
        if frozen:
            # Parameters are held constant in the candidate loss; preserve
            # downstream derivatives through the predictor output.
            z = F.conv2d(h.float(), self.projectors[i].weight.detach().float(),
                         self.projectors[i].bias.detach().float())
        else:
            z = self.projectors[i](h.float())
        return _norm(z)

    def _candidate(self, i, h, cls_logit, obj_logit, mode='pgsma',
                   freeze_reference=True, train_gate=True):
        z, valid = self._project(i, h, frozen=freeze_reference)
        s = self._prototype_scores(z, frozen=freeze_reference)
        if mode == 'memory':
            # Meaningful baseline, distinct from disabling z -> r movement.
            # Separate calibration is learned on source labels.
            delta = self.budget * torch.tanh(self.memory_gain[i] * s / self.budget)
            r = z
        else:
            raw = z + self.predictors[i](z)
            r, valid_r = _norm(raw)
            valid = valid & valid_r
            if self.correction == 'retraction':
                tangent = r - (z * r).sum(dim=1, keepdim=True) * z
                aligned, valid_a = _norm(z + tangent)
                valid = valid & valid_a
            else:
                aligned = r
            sc = self._prototype_scores(aligned, frozen=freeze_reference)
            gain = F.softplus(self.log_gain[i].float())  # positive nonzero at init
            delta = self.budget * torch.tanh(gain * (sc - s) / self.budget)
        evidence = torch.cat([cls_logit.detach().float(),
                              obj_logit.detach().float(),
                              s.detach(), delta.detach().abs()], dim=1)
        gate_in = evidence if train_gate else evidence.detach()
        gate = torch.sigmoid(self.gates[i](gate_in))
        # For ill-defined embedding input, force zero update.
        delta = delta * valid.to(delta.dtype)
        return {'delta': delta, 'gate': gate, 'z': z, 'r': r,
                'score': s, 'valid': valid}

    def forward_levels(self, features, native_logits, objectness, mode='pgsma',
                       accept=True, collect=False):
        if mode not in ('pgsma', 'native', 'memory'):
            raise ValueError(mode)
        if not (len(features) == len(native_logits) == len(objectness) == self.levels):
            raise ValueError('Wrong number of pyramid levels')
        results = []; diagnostics = []
        for i, (h, cls, obj) in enumerate(zip(features, native_logits, objectness)):
            if mode == 'native':
                results.append(cls)
                continue
            # This is an inference function; caller must ensure semantic gradients
            # cannot update native parameters in training.
            d = self._candidate(i, h.detach(), cls.detach(), obj.detach(), mode)
            adjustment = d['delta'] * (d['gate'] if accept else 1.0)
            results.append(cls + adjustment.to(dtype=cls.dtype))
            if collect: diagnostics.append({k: v.detach() for k, v in d.items()})
        return (results, diagnostics) if collect else results

    @staticmethod
    def _flatten(level_tensors):
        return torch.cat([t.float().flatten(2).transpose(1, 2) for t in level_tensors], dim=1)

    @staticmethod
    def _sample_losses(logits, fg_ids, fg_labels, fg_quality, gt_uid,
                       bg_ids, hard_class=False):
        """Instance-balanced positives plus separately mean-normalized negatives."""
        nclass = logits.shape[-1]
        flat = logits.reshape(-1, nclass)
        zero = flat.sum() * 0.0
        pos = zero
        if fg_ids.numel():
            pred = flat[fg_ids]
            label = F.one_hot(fg_labels.long(), nclass).float()
            if not hard_class:
                label = label * fg_quality[:, None].float()
            per_pos = F.binary_cross_entropy_with_logits(pred, label, reduction='none').sum(-1)
            _, inverse, counts = torch.unique(gt_uid, sorted=True, return_inverse=True,
                                              return_counts=True)
            pos = (per_pos / counts[inverse]).sum() / max(int(counts.numel()), 1)
        neg = zero
        if bg_ids.numel():
            neg = F.softplus(flat[bg_ids]).sum(-1).mean()
        return pos, neg

    @staticmethod
    def _perturb(features, seed):
        """Source-feature perturbation with independent RNG and no spatial erasure.

        Mild channel-gain and local smoothing; does not certify weather realism.
        The strength envelope is a *pilot hypothesis*, not an OOD-tuned schedule.
        """
        result = []
        for i, feat in enumerate(features):
            g = torch.Generator(device=feat.device)
            g.manual_seed(int(seed) + i * 100003)
            b, c, _, _ = feat.shape
            gain_noise = torch.randn((b, c, 1, 1), device=feat.device,
                                     generator=g, dtype=torch.float32)
            gain = (1 + 0.08 * gain_noise).clamp(min=0.70, max=1.30)
            rho = torch.rand((b, 1, 1, 1), device=feat.device, generator=g)
            rho = 0.20 * rho
            blurred = F.avg_pool2d(feat.float(), 3, stride=1, padding=1)
            result.append((feat.float() * (1 - rho) + blurred * rho) * gain)
        return result

    def semantic_losses(self, features, native_classifier_weights,
                        native_logits, objectness, match, seed=17,
                        prototype_weight=1.0, teacher_weight=0.1,
                        candidate_weight=1.0, gate_weight=1.0,
                        negative_weight=1.0, correction_mode='pgsma'):
        """Independent semantic loss stream. No gradients into native parameters.

        match: dict of `fg_ids, fg_labels, fg_quality, gt_uid, bg_ids` where ids
        refer to flattened `[B,concat(P3,P4,P5),...]` feature layout.
        Native classifier weights are `(weight,bias)` pairs and are detached.
        The head must create match from *native* assignments only.
        """
        features = [f.detach() for f in features]
        clss = [x.detach() for x in native_logits]
        objs = [x.detach() for x in objectness]
        fg = match['fg_ids'].long(); bg = match['bg_ids'].long()
        lab = match['fg_labels'].long(); quality = match['fg_quality'].float().detach()
        uid = match['gt_uid'].long()
        clean_scores = []
        for i, feat in enumerate(features):
            z, _ = self._project(i, feat)
            clean_scores.append(self._prototype_scores(z))
        scored = self._flatten(clean_scores)
        proto_p, proto_n = self._sample_losses(scored, fg, lab, quality, uid, bg, True)
        loss_proto = proto_p + float(negative_weight) * proto_n

        perturb = self._perturb(features, seed)
        native_pert_logits = []
        candidate_deltas = []
        gates = []
        predicted = []
        clean_teacher = []
        for i, feat in enumerate(perturb):
            w, bias = native_classifier_weights[i]
            native_pert_logits.append(F.conv2d(feat, w.detach().float(),
                                               bias.detach().float() if bias is not None else None))
            with torch.no_grad():
                teacher_z, _ = _norm(self.teacher_projectors[i](features[i].float()))
            clean_teacher.append(teacher_z)
            d = self._candidate(i, feat, native_pert_logits[-1], objs[i],
                                freeze_reference=True, mode=correction_mode)
            candidate_deltas.append(d['delta'])
            gates.append(d['gate'])
            predicted.append(d['r'])

        pert_native = self._flatten(native_pert_logits)
        delta = self._flatten(candidate_deltas)
        gated = self._flatten(gates) * delta.detach()
        candidate = pert_native.detach() + delta
        final = pert_native.detach() + gated
        cp, cn = self._sample_losses(candidate, fg, lab, quality, uid, bg, False)
        gp, gn = self._sample_losses(final, fg, lab, quality, uid, bg, False)
        loss_candidate = cp + float(negative_weight) * cn
        loss_gate = gp + float(negative_weight) * gn

        if fg.numel():
            pred_flat = self._flatten(predicted)
            target_flat = self._flatten(clean_teacher).detach()
            discrepancy = 1 - F.cosine_similarity(pred_flat.reshape(-1, self.dim)[fg],
                                                   target_flat.reshape(-1, self.dim)[fg], dim=-1)
            _, inv, counts = torch.unique(uid, sorted=True, return_inverse=True, return_counts=True)
            loss_jepa = (discrepancy / counts[inv]).sum() / counts.numel()
        else:
            loss_jepa = sum(t.sum() * 0 for t in predicted)

        return {
            'prototype': loss_proto * float(prototype_weight),
            'teacher': loss_jepa * float(teacher_weight),
            'candidate': loss_candidate * float(candidate_weight),
            'gate': loss_gate * float(gate_weight),
            'candidate_improvement': (self._sample_losses(pert_native, fg, lab, quality, uid, bg, False)[0] - cp).detach(),
        }
