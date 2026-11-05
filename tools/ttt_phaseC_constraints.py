"""Source-only TTT outer loss with consistent detector-unit safeguards.

Designed for the strict fixed-native-match reconstructions returned by
fixed_detection_targets.reconstruct_tde_loss. No access to test-time labels;
used ONLY during supervised source meta-training. Constraints are surrogates,
not guarantees of post-NMS AP/AR non-regression or unseen-domain safety.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

SCALES = ('small','medium','large')


@dataclass(frozen=True)
class ConstraintConfig:
    background_weight: float = 0.5
    scale_weight: float = 0.25
    clean_total_weight: float = 1.0
    background_margin: float = 0.0
    scale_margin: float = 0.0
    clean_total_margin: float = 0.0

    def validate(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be nonnegative finite')
        return self


def detector_unit_objective(native, adapted, *, clean_view: bool,
                            config: ConstraintConfig = ConstraintConfig()):
    """Return differentiable outer objective and detached diagnostic floats.

    Invariant: returned component and total deltas are in the very same units
    used by the YOLOX-TDE detection head (sum over anchors / max(1,N_fg)).
    Scale deltas are deliberately normalized per positive OF THAT SCALE; do
    not conflate them with the total loss. This method does not assume any
    'safe' margin from previous ACDC results. Margin is explicit, in the
    corresponding units, and defaults to 0.0 for measurement transparency.

    A per-image clean safeguard applies only to clean_view. The shifted view
    still receives native-target comparison penalties for background+scales.
    """
    config.validate()
    if native['counts'] != adapted['counts']:
        raise RuntimeError('Different native vs adapted fixed assignment counts')
    if native['num_foreground_normalizer'] != adapted['num_foreground_normalizer']:
        raise RuntimeError('Different foreground normalizers')
    for k in ('total',):
        if not torch.isfinite(native[k]).all() or not torch.isfinite(adapted[k]).all():
            raise FloatingPointError('Non-finite detection loss')
    native_total = native['total'].detach()
    task_delta = adapted['total'] - native_total
    bg0 = native['components']['background_objectness'].detach()
    bg1 = adapted['components']['background_objectness']
    bg_delta = bg1 - bg0  # NO multiplication by (N_fg / N_bg)!
    bg_penalty = F.relu(bg_delta - config.background_margin)

    scale_deltas = {}
    scale_penalties = []
    fg_denominator = float(native['num_foreground_normalizer'])
    for s in SCALES:
        matched = int(native['counts'][s])
        if matched <= 0:
            continue
        b0 = sum(native['scales'][s].values()).detach()
        b1 = sum(adapted['scales'][s].values())
        d = (b1 - b0) * (fg_denominator / matched)
        scale_deltas[s] = d
        scale_penalties.append(F.relu(d - config.scale_margin))
    scale_penalty = torch.stack(scale_penalties).mean() if scale_penalties else task_delta * 0.0
    clean_penalty = (F.relu(task_delta - config.clean_total_margin)
                     if clean_view else task_delta * 0.0)

    # Use delta instead of adapted['total']; same derivative, clearer units.
    primary = task_delta
    obj = (primary
           + config.background_weight * bg_penalty
           + config.scale_weight * scale_penalty
           + config.clean_total_weight * clean_penalty)
    if not torch.isfinite(obj).all():
        raise FloatingPointError('Non-finite constrained outer objective')
    def val(t):return float(t.detach())
    stats = {
        'native_detection_loss':val(native_total),
        'adapted_detection_loss':val(adapted['total']),
        'adapted_minus_native':val(task_delta),
        'primary':val(primary),
        'background_delta_detector_units':val(bg_delta),
        'background_margin_detector_units':config.background_margin,
        'background_violation':val(bg_penalty),
        'background_guard_active':bool(val(bg_delta)>config.background_margin),
        'background_extra_gradient_multiplier_when_active':config.background_weight,
        'background_anchor_count':int(native['counts']['background']),
        'foreground_match_count':int(native['counts']['foreground']),
        'scale_delta_per_matched_positive':{s:val(d) for s,d in scale_deltas.items()},
        'scale_violations':{s:val(F.relu(d-config.scale_margin)) for s,d in scale_deltas.items()},
        'scale_coverage':{s:int(native['counts'][s]) for s in SCALES},
        'scale_penalty_mean':val(scale_penalty),
        'clean_total_penalty':val(clean_penalty),
        'clean_total_guard_active':bool(clean_view and val(task_delta)>config.clean_total_margin),
        'objective':val(obj),
    }
    return obj, stats
