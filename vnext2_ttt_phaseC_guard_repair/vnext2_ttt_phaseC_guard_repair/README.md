# vNext.2 Safe TTT — Phase C detector-unit constraint repair

## Why Phase C

The Phase B 32-step source-only diagnostic found:

- A beneficial *training-induced* shift of -6.06e-5 in shifted-view mean total fixed-target detector loss, but descriptive 95% image-bootstrap interval included zero.
- Remaining shifted background loss versus **native** +5.42e-5, and clean-view background +6.53e-6.
- Phase B trained predictor **improved** medium-scale loss versus its own initialization (clean -2.42e-5, shifted -3.34e-5), but worsened large-scale mean loss (clean +8.19e-5, shifted +1.02e-4), with only 8/16 holdout images containing large matched positives.
- Old extra background guard had an effective gradient factor `nonregression_weight * background_weight * N_fg/N_bg` (~0.000375 against original background-loss gradient) because background was changed to per-negative-anchor units, while the primary detector loss is per-foreground-normalized.
- The previous report's background violation threshold of `2e-5` **per background anchor** was equivalent to roughly `0.0027` in actual detection-loss units, so it missed the small measured background deterioration.

This patch corrects the objective units and makes the clean/shifted gradient mixture unambiguous. It DOES NOT prove that Phase C is a better detector or guarantee test-time safety.

## What changes (and what does not)

NEW, opt-in source-only files:

- `tools/ttt_phaseC_constraints.py`: direct foreground-normalized detector-unit background hinge; explicit per-matched-positive scale hinge; clean-view total non-regression hinge; detailed violation diagnostics. Defaults: `background_weight=0.5`, `scale_weight=0.25`, `clean_total_weight=1.0`, margins=0. These are **pilot values** to inspect, not calibrated/safe hyperparameters.
- `tools/pilot_safe_ttt_phaseC.py`: imports your previous installed sidecar, native dark2 router, strict fixed-target TDE head, clean appearance transform, and Phase A predictor-freezing logic; accumulates exact second-order meta-gradients across distinct clean training images; steps only the SSL predictor. `clean_weight=1` means clean and shifted contribute 50/50 total (not the Phase B 0.5:1 ratio). No GT enters test-time adaptation.
- `tests/test_ttt_phaseC_constraints.py`: portable unit tests for gradient coefficients, exact units, positive/negative hinge, explicit scale normalization, and invariant refusal.
- `tests/test_phaseC_snapshot_integration.py`: OPTIONAL full older-snapshot YOLOX/TDE forward+meta-gradient integration test; requires previous snapshot fixture. Not necessary on the user's current checkout, and does not replace actual CUDA testing.

Unchanged: original `tde_yolox_s.py`, native checkpoint, trainer, actual test-time TTT objective, frozen adapter, mask token, learned step logit, and downstream detector head. This is *only* a training-objective/optimization pilot.

## Installation

From `D:\YOLOX-3rd`, after extracting the archive to a similarly named folder:

```powershell
Copy-Item .\vnext2_ttt_phaseC_guard_repair\tools\*.py .\tools\ -Force
Copy-Item .\vnext2_ttt_phaseC_guard_repair\tests\*.py .\tests\ -Force
python tests/test_ttt_phaseC_constraints.py
python tools/pilot_safe_ttt_phaseC.py --help
```

Prerequisites already installed from prior successful phases:
`tools/audit_safe_ttt_yolox.py`, `tools/fixed_detection_targets.py`, `tools/native_dark2_router.py`, `tools/pilot_safe_ttt_meta.py`, `tools/pilot_safe_ttt_predictor_only.py`, `tools/pilot_safe_ttt_paired_views.py`, `tools/ttt_source_views.py`, `tools/safe_ttt_model.py`.

The existing Phase B files are never overwritten by this package.

## First: real-checkpoint smoke test (2 optimizer steps, 4 source episodes)

```powershell
python tools/pilot_safe_ttt_phaseC.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py --weights ema `
  --train-start 300 --train-count 4 `
  --holdout-start 500 --holdout-count 4 `
  --steps 2 --episodes-per-step 2 `
  --lr 0.00005 --clean-weight 1 `
  --background-weight 0.5 --scale-weight 0.25 --clean-total-weight 1 `
  --strength 0.85 --teacher-weight 0 `
  --rank 8 --basis-gain 1 --step-policy normalized `
  --output audits/safe_ttt_phaseC_smoke.json `
  --save-sidecar audits/safe_ttt_phaseC_smoke.pth
```

The script refuses to overwrite either output. The sidecar is saved *before* the final holdout evaluation (same reliability pattern as Phase B).

Pass the mechanical smoke gate only if: strict native head parity, finite nonzero meta-gradients, zero changes in frozen parameters and native checkpoint, and output report with both clean and shifted holdout sections. Positive heldout improvement is **NOT** required after only two updates.

## Next, ONLY IF smoke passes: 32 episodes / 8 optimizer updates

Keep images and holdout identities identical to the earlier Phase B comparison; however, this is a **multi-change optimization pilot**, not a causal isolated guard ablation. New guard weights, equal view weighting, accumulation, and LR all differ from Phase B.

```powershell
python tools/pilot_safe_ttt_phaseC.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py --weights ema `
  --train-start 300 --train-count 32 `
  --holdout-start 500 --holdout-count 16 `
  --steps 8 --episodes-per-step 4 `
  --lr 0.00005 --clean-weight 1 `
  --background-weight 0.5 --scale-weight 0.25 --clean-total-weight 1 `
  --strength 0.85 --teacher-weight 0 `
  --rank 8 --basis-gain 1 --step-policy normalized `
  --output audits/safe_ttt_phaseC_32episode.json `
  --save-sidecar audits/safe_ttt_phaseC_32episode.pth
```

**Do not** train for an epoch or evaluate on ACDC to select loss weights. You can inspect losses on source labels, but cannot use labels to control inference adaptation.

## Interpretation and safety

`summary.clean` and `summary.shifted` report trained versus initialized fixed-target detection loss (paired); `background_violations_detector_units` now checks the actual aggregate detector background component against the explicit margin in those same units (zero by default). It is a stringent *sign check*, not a statistically calibrated significance threshold. Inspect magnitude and numerical repeatability as well as violation count.

`steps[].episodes[]` records each image's guard activation, raw background delta in detector units, and all supported scale deltas per matched positive. Large-object coverage matters: a scale absent from a scene is **not a pass**. Some training changes can be positive on one task and negative on another.

A per-image hinge is **not** a formal constraint satisfaction guarantee in expectation or on unseen scenes. Gradient regularizers can conflict, especially with tiny absolute loss changes. A negative aggregate surrogate is also not AP/AR improvement. Real acceptance requires (i) numerical confidence, (ii) a disjoint source holdout with adequate large-object coverage, (iii) multiple independent seeds and source view types, (iv) eventual clean AP/AR non-regression and held-out adverse evaluation.

## Validation performed here

- `python tests/test_ttt_phaseC_constraints.py`: five portable CPU tests passed.
- Optional full older-source YOLOX/TDE snapshot test: two episode accumulation + one predictor step passed with nonzero meta-gradient; frozen model and sidecar parameter checks passed.
- NOT run: user's Windows current-source CUDA checkpoint, real BDD source sample experiment, production inference or ACDC. Nothing in this archive changes user files automatically.
