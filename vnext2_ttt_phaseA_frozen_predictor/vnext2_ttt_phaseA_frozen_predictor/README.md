# vNext.2 Safe TTT — Phase A frozen-spatial-basis predictor learning

## Why this stage exists

The user's 16-factorial attribution at source-clean fixed matching, 0.5× step, 8 paired images produced **+0.000164017** mean trained-minus-untrained detection loss. Shapley allocations: predictor **48.1%**, adapter **32.6%**, mask token **12.4%**, step logit **6.9%**. These are interaction-sharing allocations over hybrid parameter states, *not* isolated causal effects or estimates of actual AP/AR.

Our immediate experimental question: **if the representation basis and step control are held fixed, can exact detection->SSL mixed-derivative learning improve the predictor's held-out adaptation effect?**

This package makes NO modifications to original YOLOX, TTT modules, optimizer/trainer, EMA, or checkpoints. It starts with the same seeded *initial* SafeSpatialTTT sidecar, but freezes all parameters except `predictor.*`.

## Installation

Copy only the two new files from this package into the matching YOLOX repository directories:

- `tools/pilot_safe_ttt_predictor_only.py`
- `tests/test_safe_ttt_predictor_only.py`

Already required (from the previous safe-TTT audit bundle): `tools/pilot_safe_ttt_meta.py`, `tools/safe_ttt_model.py`, `tools/audit_safe_ttt_yolox.py`, `tools/fixed_detection_targets.py`, `tools/native_dark2_router.py`.

From YOLOX root:

```powershell
python tests/test_safe_ttt_predictor_only.py
```

The portable CPU tests validate frozen-state immutability under an exact differentiable update, nonzero predictor gradients, fail-closed parameter selection and paired holdout reporting. **The full CUDA checkpoint pilot must be run locally.**

## Prespecified source-only pilot

Use source-clean TRAIN images for meta-training, with a disjoint clean-source TRAIN group as an *internal diagnostic holdout*. Do not use ACDC labels to train or select TTT.

```powershell
python tools/pilot_safe_ttt_predictor_only.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py `
  --weights ema `
  --train-indices 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 `
  --holdout-indices 500 501 502 503 504 505 506 507 `
  --steps 16 --lr 0.0001 `
  --seed 20261104 --objective-mode detector_total `
  --step-policy normalized --rank 8 --basis-gain 1 `
  --output audits/safe_ttt_predictor_only_phaseA.json `
  --save-sidecar audits/safe_ttt_predictor_only_phaseA.pth
```

This is a **16-update mini-pilot**, not one epoch, and not a guarantee of improvement. It freezes the native YOLOX parameters and all the sidecar weights except `predictor.*`. It takes one gradient step per source-clean labeled example, using the same fixed native SimOTA targets and real second-order detection->SSL path as the existing pilot. The code verifies all frozen tensor values remain **bitwise unchanged** after every optimizer step and that the native model's hash is unchanged.

Both the initial and learned predictor are evaluated on exactly the same eight disjoint source examples. Holdout improvements use thresholds `max(1e-5, 1e-5 * |native_loss|)` and report per-image paired differences. No optimizer choices are made using these holdout results inside the script.

For a second independent initialization, run a **separate** experiment with `--seed 20261105` and unique output filenames. Do not fit the seed, step strength, masking, LR or training indices to the eight earlier examples; that would consume the holdout through iterative tuning.

### What to look for

- `predictor_parameter_delta_l2 > 0`; exact detection-to-SSL gradients were effective.
- `frozen_state_changed_keys == []`; spatial basis, mask token and step controller were untouched.
- `native_model_mutations == []`; entire pretrained detector and EMA remain frozen.
- `heldout.mean_trained_minus_initial < 0` and a reasonable share of genuinely improved heldout source episodes, across **multiple seeds**.
- Source holdout per-scale box, classification and positive-objectness deltas should be inspected, not just the aggregate.

If held-out changes are tiny and inconsistent, do not promote this pilot. Repeatability of a deterministic forward pass is necessary but does **not** prove meaningful detection improvement at `1e-5`–`1e-4` loss scales. An effect should survive independent seeds and larger disjoint source examples, and eventually detection AP/AR validation.

### Critical limits

1. This script performs source-clean *supervised meta-training* of the SSL predictor. Ground-truth labels are never available to the online TTT inner update, but they **are** intentionally used for training and the offline holdout. The latter is not an unbiased final test set.
2. The native fixed-target loss is a controlled gradient diagnostic. It does not capture reassignment after adaptation or post-NMS AP/AR behavior. As originally implemented, scale-based non-regression is a **surrogate** and is not a proof of AP/AR safety.
3. The tested SSL objective is still masked self-feature prediction on the same image. If frozen-basis predictor learning fails, the problem may be the SSL objective or the distribution of training episodes. Further projector capacity alone is not justified.
4. The current source-clean episodic loader does not explicitly construct diverse paired appearance shifts. Success on that loader would *not* yet prove clean-to-adverse adaptation. A source-only controlled corruption/paired-view curriculum should be the next training-data research step, with a final untouched adverse test set.
5. A single pilot with 16 examples is not a generalization study and should not trigger 80-epoch training. It is a targeted structural test after the 16-way attribution.
6. No vNext.3 integration. No original file overwrites.
