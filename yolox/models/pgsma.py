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
        # Semantic-v2 gate: SIGNED class changes are essential evidence.
        # Its altered input dimensions deliberately reject legacy PG-SMA gate
        # checkpoints rather than silently reinterpret learned weights.
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(classes * 4 + 1, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
            for _ in range(levels)
        ])
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, -3.0)  # Conservative prior, not zero-gradient.
        # Separate, trainable scalar for a *genuine* memory-only ablation.
        self.memory_gain = nn.Parameter(torch.zeros(levels))
        self.register_buffer('semantic_step', torch.tensor(0, dtype=torch.long))
        # Opt-in source-only objective. Default preserves semantic-v2 behavior.
        self.candidate_objective = 'v2'

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
        if mode == 'memory':
            s = self._prototype_scores(z, frozen=freeze_reference)
            # Meaningful baseline, distinct from disabling z -> r movement.
            delta = self.budget * torch.tanh(self.memory_gain[i] * s / self.budget)
            r = z
        else:
            # Numerical identity invariant: score the SAME normalised baseline
            # as the movement path uses. Otherwise F.normalize(F.normalize(z))
            # can differ from z by ~1e-7 at zero predictor output.
            baseline, valid_base = _norm(z)
            valid = valid & valid_base
            s = self._prototype_scores(baseline, frozen=freeze_reference)
            movement = self.predictors[i](z)
            raw = z + movement
            r, valid_r = _norm(raw)
            valid = valid & valid_r
            if self.correction == 'retraction':
                # Use actual predictor motion for tangent calculation. When
                # movement is exactly zero, tangent is exactly zero too.
                tangent = movement - (baseline * movement).sum(dim=1, keepdim=True) * baseline
                aligned, valid_a = _norm(z + tangent)
                valid = valid & valid_a
            else:
                aligned = r
            sc = self._prototype_scores(aligned, frozen=freeze_reference)
            gain = F.softplus(self.log_gain[i].float())
            delta = self.budget * torch.tanh(gain * (sc - s) / self.budget)
        evidence = torch.cat([cls_logit.detach().float(),
                              obj_logit.detach().float(),
                              s.detach(), delta.detach(), delta.detach().abs()], dim=1)
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

    @staticmethod
    def _balanced_per_gt(values, gt_uid):
        """Differentiable per-instance average; every GT has equal influence."""
        if not values.numel():
            return values.new_zeros((0,)), gt_uid.new_zeros((0,))
        names, inv, counts = torch.unique(gt_uid, sorted=True, return_inverse=True,
                                          return_counts=True)
        sums = values.new_zeros(len(names)).scatter_add(0, inv, values)
        return sums / counts.to(values.dtype), names

    @classmethod
    def _relative_margin_cost(cls, base, corrected, fg, labels, quality, gt_uid):
        """Paired *source-label-only* classification directional probe.

        IoU is a sample weight, not the target confidence. This tests whether
        IoU-soft BCE and background suppression have overdominated true-class
        margin learning. The full v2 BCE and scale guards remain active.
        Training only; no labels/matching needed in semantic inference.
        """
        nclass = base.shape[-1]
        if not fg.numel():
            return corrected.sum()*0.0
        pred = corrected.reshape(-1,nclass)[fg]
        ref = base.reshape(-1,nclass)[fg].detach()
        idx = torch.arange(len(fg),device=fg.device)
        delta_true = pred[idx,labels] - ref[idx,labels]
        # A small positive score movement is encouraged only when there is a
        # labeled source positive. Keep native IoU-soft BCE guard in place.
        cls_cost = 0.1 * F.softplus((0.025 - delta_true)/0.1)
        if nclass > 1:
            is_true = F.one_hot(labels,nclass).bool()
            competitor_ref = torch.logsumexp(ref.masked_fill(is_true,float('-inf')),dim=-1)
            competitor_pred = torch.logsumexp(pred.masked_fill(is_true,float('-inf')),dim=-1)
            margin_gain = delta_true-(competitor_pred-competitor_ref)
            cls_cost = cls_cost + 0.1 * F.softplus((0.05-margin_gain)/0.1)
        # Avoid discarding low-IoU/micro positives while giving well-localized
        # candidates modestly more influence.
        weighted = cls_cost*(0.4+0.6*quality.detach().clamp(0,1))
        per_gt,_ = cls._balanced_per_gt(weighted,gt_uid)
        return per_gt.mean()


    @classmethod
    def _relative_bounded_margin_cost(cls, base, corrected, fg, labels, quality, gt_uid):
        """Directional margin with an exact zero-gradient region beyond each target.

        At zero intervention, slopes match the G4 softplus probe (approximately
        0.562 and 0.622). When the target movement has been achieved, that
        component stops incentivizing larger logits. This is NOT a guarantee
        against background or tiny-object regression.
        """
        nclass = base.shape[-1]
        if not fg.numel():
            return corrected.sum() * 0.0
        pred = corrected.reshape(-1, nclass)[fg]
        ref = base.reshape(-1, nclass)[fg].detach()
        idx = torch.arange(len(fg), device=fg.device)
        dt = pred[idx, labels] - ref[idx, labels]
        true_target = 0.025
        true_initial_slope = 1.0 / (1.0 + math.exp(-true_target / 0.1))
        cost = (true_initial_slope / (2.0 * true_target)) * F.relu(true_target - dt).square()
        if nclass > 1:
            is_true = F.one_hot(labels.long(), nclass).bool()
            c_ref = torch.logsumexp(ref.masked_fill(is_true, float('-inf')), dim=-1)
            c_pred = torch.logsumexp(pred.masked_fill(is_true, float('-inf')), dim=-1)
            dm = dt - (c_pred - c_ref)
            margin_target = 0.05
            margin_initial_slope = 1.0 / (1.0 + math.exp(-margin_target / 0.1))
            cost = cost + (margin_initial_slope / (2.0 * margin_target)) * F.relu(margin_target - dm).square()
        weighted = cost * (0.4 + 0.6 * quality.detach().clamp(0.0, 1.0))
        per_gt, _ = cls._balanced_per_gt(weighted, gt_uid)
        return per_gt.mean()

    @staticmethod
    def _relative_background_regret(base, corrected, bg):
        """Penalty only when semantic logits worsen sampled native background BCE.

        Samples are native-objectness-mined GT-halo negatives. This tests a
        source-supervised optimization hypothesis; it is not an all-background
        or out-of-domain nonregression certificate.
        """
        if not bg.numel():
            return corrected.sum() * 0.0
        nclass = base.shape[-1]
        native = F.softplus(base.reshape(-1, nclass)[bg]).sum(-1).detach()
        proposed = F.softplus(corrected.reshape(-1, nclass)[bg]).sum(-1)
        regret = proposed - native
        return F.relu(regret.mean()) + 0.25 * F.relu(regret).mean()

    def _view_losses(self, features, native_logits, objectness, match,
                     negative_weight=0.25, fg_guard_weight=2.0,
                     bin_guard_weight=1.0, class_guard_weight=0.5,
                     gate_fg_weight=2.0, gate_bg_weight=0.25):
        """Source-only paired candidate utility and a selective gate objective.

        GT never enters inference. The foreground guard supervises only the
        isolated semantic optimizer, and does not alter native YOLOX losses.
        """
        candidates = [self._candidate(i, f, cl, ob, freeze_reference=True)
                      for i, (f, cl, ob) in enumerate(zip(features, native_logits, objectness))]
        base = self._flatten(native_logits).detach()
        delta = self._flatten([d['delta'] for d in candidates])
        gate = self._flatten([d['gate'] for d in candidates])
        corrected = base + delta
        fg = match['fg_ids'].long(); lab = match['fg_labels'].long()
        uid = match['gt_uid'].long(); bg = match['bg_ids'].long()
        quality = match['fg_quality'].float().detach()
        nclass = base.shape[-1]

        # Existing paired source labels supervise detection utility, not an
        # auxiliary proxy. Do not average hundreds of easy BG examples against
        # a handful of small-object GTs.
        cp, cn = self._sample_losses(corrected, fg, lab, quality, uid, bg, False)
        bp, _ = self._sample_losses(base, fg, lab, quality, uid, bg, False)
        zero = corrected.sum() * 0.0
        if self.candidate_objective not in ('v2', 'margin_probe', 'bounded_probe', 'margin_bg_probe', 'bounded_bg_probe'):
            raise ValueError('Unknown candidate objective: '+str(self.candidate_objective))
        if self.candidate_objective in ('margin_probe', 'margin_bg_probe'):
            rank_cost = self._relative_margin_cost(base, corrected, fg, lab, quality, uid)
        elif self.candidate_objective in ('bounded_probe', 'bounded_bg_probe'):
            rank_cost = self._relative_bounded_margin_cost(base, corrected, fg, lab, quality, uid)
        else:
            rank_cost = zero
        bg_guard = (self._relative_background_regret(base, corrected, bg)
                    if self.candidate_objective in ('margin_bg_probe', 'bounded_bg_probe') else zero)
        fg_guard = zero; bin_guard = zero; class_guard = zero
        gate_fg_loss = zero; gate_bg_loss = zero
        if fg.numel():
            true_labels = F.one_hot(lab, nclass).float() * quality[:, None]
            native_per = F.binary_cross_entropy_with_logits(base.reshape(-1,nclass)[fg],
                                                             true_labels, reduction='none').sum(-1).detach()
            corr_per = F.binary_cross_entropy_with_logits(corrected.reshape(-1,nclass)[fg],
                                                           true_labels, reduction='none').sum(-1)
            native_gt, gt_names = self._balanced_per_gt(native_per, uid)
            corr_gt, _ = self._balanced_per_gt(corr_per, uid)
            regressions = F.relu(corr_gt-native_gt)
            fg_guard = regressions.mean()

            # Quality-weighted guard: high-IoU true positives deserve stronger
            # target-class preservation. Low-IoU anchors retain calibration
            # freedom; don't impose an unnecessarily rigid invariant.
            class_movement = delta.reshape(-1,nclass)[fg,lab]
            class_gt, _ = self._balanced_per_gt(quality * F.relu(-class_movement), uid)
            class_guard = class_gt.mean()

            if 'fg_sizes' in match:
                sizes_gt, _ = self._balanced_per_gt(match['fg_sizes'].float().detach(), uid)
                which = torch.bucketize(sizes_gt, sizes_gt.new_tensor([4.,8.,16.,32.]))
                groups = [regressions[which==k].mean() for k in range(5) if (which==k).any()]
                if groups:bin_guard = torch.stack(groups).max()

            # Gate needs ground-truth utility counterfactuals, not more
            # foreground/background classification BCE driving it to 1.
            # Both lower BCE AND nonnegative GT-class movement are required
            # to label an anchor an acceptable correction during training.
            useful = ((native_per-corr_per.detach())>0.0) & (
                class_movement.detach() >= -0.1*(1.0-quality))
            gfg = gate.reshape(-1,1)[fg,0]
            per_fg_gate = F.binary_cross_entropy(gfg.clamp(1e-6,1-1e-6),
                                                 useful.float(), reduction='none')
            grouped, _ = self._balanced_per_gt(per_fg_gate, uid)
            gate_fg_loss = grouped.mean()
        if bg.numel():
            native_bg = F.softplus(base.reshape(-1,nclass)[bg]).sum(-1).detach()
            corr_bg = F.softplus(corrected.reshape(-1,nclass)[bg]).sum(-1).detach()
            useful_bg = ((native_bg-corr_bg)>0.0).float()
            gbg = gate.reshape(-1,1)[bg,0]
            gate_bg_loss = F.binary_cross_entropy(gbg.clamp(1e-6,1-1e-6),
                                                  useful_bg, reduction='mean')
        loss_candidate = (cp + float(negative_weight)*cn
                          + float(fg_guard_weight)*fg_guard
                          + float(bin_guard_weight)*bin_guard
                          + float(class_guard_weight)*class_guard
                          + rank_cost
                          + 2.0*bg_guard
                          + 0.01*delta.square().mean())
        loss_gate = float(gate_fg_weight)*gate_fg_loss + float(gate_bg_weight)*gate_bg_loss
        return {'candidate':loss_candidate, 'gate':loss_gate,
                'rank_probe_loss':rank_cost.detach(),
                'bg_guard':bg_guard.detach(),
                'fg_guard':fg_guard.detach(), 'bin_guard':bin_guard.detach(),
                'class_guard':class_guard.detach(),
                'gate_fg_loss':gate_fg_loss.detach(), 'gate_bg_loss':gate_bg_loss.detach(),
                'candidate_fg_improvement':(bp-cp).detach(),
                'candidate_bg_loss':cn.detach(), 'predicted':[c['r'] for c in candidates]}

    def semantic_losses(self, features, native_classifier_weights,
                        native_logits, objectness, match, seed=17,
                        prototype_weight=1.0, teacher_weight=0.1,
                        candidate_weight=1.0, gate_weight=1.0,
                        negative_weight=0.25, correction_mode='pgsma'):
        """Paired clean + perturbed source supervision; no native gradients.

        This is an experimental semantic-v2 objective. It is intentionally NOT
        checkpoint-compatible with old gates. Requires a matched new pilot.
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
        for i, feat in enumerate(perturb):
            w, bias = native_classifier_weights[i]
            native_pert_logits.append(F.conv2d(feat, w.detach().float(),
                                               bias.detach().float() if bias is not None else None))
        clean = self._view_losses(features, clss, objs, match, negative_weight)
        pert = self._view_losses(perturb, native_pert_logits, objs, match, negative_weight)
        if fg.numel():
            teacher_z = []
            for i, f in enumerate(features):
                with torch.no_grad():
                    t, _ = _norm(self.teacher_projectors[i](f.float()))
                teacher_z.append(t)
            pred_flat = self._flatten(pert['predicted'])
            target_flat = self._flatten(teacher_z).detach()
            discrepancy = 1 - F.cosine_similarity(pred_flat.reshape(-1,self.dim)[fg],
                                                   target_flat.reshape(-1,self.dim)[fg],dim=-1)
            per_gt, _ = self._balanced_per_gt(discrepancy, uid)
            loss_jepa = per_gt.mean()
        else:
            loss_jepa = sum(t.sum()*0 for t in pert['predicted'])
        return {
            'prototype':loss_proto*float(prototype_weight),
            'teacher':loss_jepa*float(teacher_weight),
            'candidate':0.5*(clean['candidate']+pert['candidate'])*float(candidate_weight),
            'gate':0.5*(clean['gate']+pert['gate'])*float(gate_weight),
            'candidate_improvement':0.5*(clean['candidate_fg_improvement']+
                                         pert['candidate_fg_improvement']).detach(),
            'rank_probe_loss':0.5*(clean['rank_probe_loss']+pert['rank_probe_loss']),
            'clean_bg_guard':clean['bg_guard'], 'pert_bg_guard':pert['bg_guard'],
            'clean_fg_guard':clean['fg_guard'], 'pert_fg_guard':pert['fg_guard'],
            'clean_bin_guard':clean['bin_guard'], 'pert_bin_guard':pert['bin_guard'],
            'clean_class_guard':clean['class_guard'], 'pert_class_guard':pert['class_guard'],
            'clean_gate_fg_loss':clean['gate_fg_loss'], 'pert_gate_fg_loss':pert['gate_fg_loss'],
            'clean_gate_bg_loss':clean['gate_bg_loss'], 'pert_gate_bg_loss':pert['gate_bg_loss'],
        }
