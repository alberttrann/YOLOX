# vNext.2 Safe TTT — Phase B paired-view meta-training (NO YOLOX epochs)

This is an **opt-in research sidecar**, not a patch to the vNext.2 detector and not a production TTT replacement. No original YOLOX model, trainer, checkpoint, or evaluator code is modified. It builds directly on the tools you already installed in the previous safe-TTT bundles.

## 1. Why this exists

Evidence from the real epoch-2 detector:

- Original masked reconstruction had a weak masked-region gradient path.
- Diagnostic local context amplified masked gradients, but improved SSL reconstruction did not improve AP/AR.
- New safe TTT passed mixed-derivative, EMA and native-identity audits, yet the two-step jointly optimized sidecar harmed the holdout on the fixed-target surrogate.
- Frozen-basis predictor-only Phase A had 16 finite/nonzero meta-gradients, but mean adapted-minus-native holdout detection loss remained positive (~+6.19e-5); only 3/8 holdout examples improved relative to the initial predictor.

**Research hypothesis for Phase B:** meta-train the predictor from *two geometric-identical source views* with direct supervision on clean and photometrically shifted detection consequences. This can teach the unlabeled online SSL gradient to be less damaging across appearance variations, while explicitly protecting the original clean model. This hypothesis is NOT yet demonstrated on real BDD/ACDC data.

## 2. Files and installation

Copy only these NEW files into the corresponding folders in `D:\YOLOX-3rd`:

- `tools/ttt_source_views.py` — pure seeded appearance specs; range and geometry checks; preserve YOLOX's trailing 114 letterbox pad
- `tools/pilot_safe_ttt_paired_views.py` — dual-view predictor-only exact meta-gradient training plus clean+shifted holdout
- `tools/summarize_safe_ttt_phaseB.py` — detailed source-only report review
- `tests/test_safe_ttt_phaseB_source_views.py` — portable CPU property tests
- `tests/test_safe_ttt_phaseB_snapshot.py` — optional CPU snapshot-model regression, see below

**Prerequisites** (from your previously installed bundles):
`tools/safe_ttt_model.py`, `tools/native_dark2_router.py`, `tools/fixed_detection_targets.py`, `tools/audit_safe_ttt_yolox.py`, `tools/pilot_safe_ttt_meta.py`, and `tools/pilot_safe_ttt_predictor_only.py`. The latter should be the Phase A reporting hotfix version that worked for your last 16-step run. Your original per-image vNext.2 TTT source fix must remain in place.

Installation (PowerShell, from repo root after extraction):

```powershell
Copy-Item .\vnext2_ttt_phaseB_paired_views\tools\*.py .\tools\ -Force
Copy-Item .\vnext2_ttt_phaseB_paired_views\tests\*.py .\tests\ -Force
python tests/test_safe_ttt_phaseB_source_views.py
```

The snapshot integration test is optional: it is designed for the supplied old source dump and its test fixtures, **not** a hard requirement for running against your current Git checkout. It was executed on CPU during package construction.

## 3. Exactly what is trained

For each original clean-source input image `x` and its detector labels `y`:

1. Read the existing **clean TRAIN** source image with your current COCO/YOLOX preprocessing. Do not read an ACDC image or label.
2. Construct deterministic photometric `x_shift` (brightness, contrast, gamma, per-channel gains, occasional mild blur), with **the same height, width and GT boxes**. Preserve trailing constant YOLOX letterbox padding.
3. **Clean branch:** freeze native SimOTA assignments on `x`, adapt the **compact fast state** from the unlabeled clean input using the current `safe.run_episode(meta_train=True)`, and evaluate the resulting detection loss on `y` (weight 0.5 by default).
4. **Shifted branch:** independently freeze native SimOTA assignments on `x_shift`, adapt the compact state from the *unlabeled shifted input only*, and evaluate the detection loss on the *same labels* (weight 1 by default).
5. Both branches use the same one-sided per-GT-scale and background non-regression regularizer against their corresponding *native* predictions. A missing scale is excluded, not counted as safe.
6. Backpropagate the **two exact second-order meta-gradients sequentially** (not two giant simultaneous graphs). Take **one AdamW optimizer step of the SSL predictor only**.

Frozen: YOLOX weights+buffers, spatial correction basis, mask token, step-logit. `create_graph=True` is used only for the compact state SSL gradient during meta-training. Native head remains in *loss mode* while all its BN/child layers stay in `eval()`.

The SSL inner objective is unchanged from the previous safe TTT model. It does **not** use labels, clean-reference images, or teacher predictions. At inference it uses a single unlabeled image and episodic per-image fast state only.

**Optional clean-reference consistency** `--teacher-weight` uses **GT-associated anchors only** to compare adapted shifted boxes/soft cls/obj probabilities with native clean predictions. It is an *outer* source-training-only regularizer, not part of online SSL. **Default 0.0** because an epoch-2 teacher may be weak. Start with 0.0 and do not enable just to make reconstruction look better.

## 4. First: the inexpensive two-step CUDA smoke test

Before attempting a 32-step pilot, check current-source compatibility with two complete episodes and four holdouts:

```powershell
python tools/pilot_safe_ttt_paired_views.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py --weights ema `
  --train-start 300 --train-count 4 `
  --holdout-start 500 --holdout-count 4 `
  --steps 2 --lr 0.0001 `
  --strength 0.85 --clean-weight 0.5 `
  --objective-mode detector_total --nonregression-weight 0.5 `
  --teacher-weight 0 --step-policy normalized --basis-gain 1 --rank 8 `
  --output audits/safe_ttt_phaseB_smoke.json `
  --save-sidecar audits/safe_ttt_phaseB_smoke.pth
```

Mandatory smoke gates: strictly matching clean-source geometry, finite second-order gradients on both views, exact native target parity, zero mutation in frozen native/sidecar parameters, and identical source image fingerprints on pre/post holdout evaluation. An improvement in 2 steps is **not** an acceptance requirement.

## 5. Then: 32 *source episodes*, not 32 epochs

**Only if smoke tests pass**, use an isolated fresh-seed initialization (by default, the exact same seed as Phase A) and separate filenames:

```powershell
python tools/pilot_safe_ttt_paired_views.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py --weights ema `
  --train-start 300 --train-count 32 `
  --holdout-start 500 --holdout-count 16 `
  --steps 32 --lr 0.0001 `
  --strength 0.85 --clean-weight 0.5 `
  --objective-mode detector_total --nonregression-weight 0.5 `
  --teacher-weight 0 --step-policy normalized --basis-gain 1 --rank 8 `
  --output audits/safe_ttt_phaseB_32step.json `
  --save-sidecar audits/safe_ttt_phaseB_32step.pth
```

This is **32 predictor optimizer updates** over distinct source-training images. It does NOT retrain YOLOX, and the original checkpoint is not overwritten.

The script writes its sidecar **before the final evaluation**, so a report-formatting failure does not destroy training work. To regenerate the report from that exact sidecar (no further optimization) with the exact same arguments, add:

```powershell
  --evaluate-only-sidecar audits/safe_ttt_phaseB_32step.pth
```

Use a different `--output` filename on recovery because the script refuses to overwrite completed reports.

## 6. Read the result, not just the training loss

```powershell
python tools/summarize_safe_ttt_phaseB.py `
  --report audits/safe_ttt_phaseB_32step.json
```

Two independent holdout views are measured before and after training, with exactly the same clean source image content, same appearance transform and seed, same native model, same native SimOTA targets and identical evaluator settings.

Look at:

- `summary.clean`: clean loss before/after adaptation, paired training benefit/harm and strict scale/background non-regression passes
- `summary.shifted`: the same on photometrically altered source images; an improvement here is not sufficient if clean performance worsens
- `scales`: **per-matched-positive** box/class/objectness deltas for small, medium and large objects with `count`; missing scales are NOT success cases
- `components`: actual box, class, positive-objectness and background-objectness loss differences
- `relative_feature_change_postcast`, `changed_feature_fraction`: does update reach the detector?
- `loss_parity`: exact head-math validation for every view; any incompatibility must abort instead of silently rematching
- `trainable_families`/`frozen_families` and hashes in the checkpoint: source label use, sidecar provenance and no frozen parameter mutation

The printed `ADAPT_DET` at different train indices is **not a learning curve**; compare the fixed holdout before/after instead.

## 7. Interpretation and stop/go decisions

1. **Clean or small-object harm**: reject current candidate despite a shifted average loss improvement. Adjust objective composition or appearance severity *using source data only* and rerun controlled short pilots.
2. **No useful finite-step effect** across clean and shifted, despite meta-gradients: do not train 80 epochs. The next hypothesis should target the SSL objective (masked own-feature prediction may be poor for restoration) or low-dimensional adapter basis; a higher learning rate alone is not evidence-based.
3. **Both clean and shifted improve on disjoint source samples across seeds, with no systematic scale/component regressions**: extend from 16 to 32–128 fixed source holdouts; then check clean benchmark AP/AR using `eval_safe_ttt_sidecar.py` without adverse tuning.
4. **Only then** freeze architecture, tune exclusively with clean training/validation, and evaluate on previously unseen adverse conditions. Previously inspected ACDC conditions are NOT an untouched final test set.

The inference evaluator from the earlier package can read these Phase B sidecars because it uses `safe_ttt`, `configuration.clean_train_only=True`, original checkpoint path/hash, rank, normalized step policy and state_dict compatibility. No new inference architecture is introduced.

## 8. Explicit limitations

- These are **source-TRAIN fixed-native-target surrogates**, not clean validation AP/AR and not novel adverse DG metrics.
- 16 source holdouts are far from sufficient for population-level confidence; report per-image effects and repeatability, and use >=32 disjoint samples plus multiple seeds before any positive claim.
- Transformations are modest hand-designed appearance perturbations; they may not cover fog, snow, night or real sensor physics and must not be described as such.
- The optional teacher may be wrong; its loss is not a clean ground-truth restoration target.
- No learned reliability/abstention gate exists, and the existing feature correction bound is NOT a semantic safety guarantee.
- Meta-training with `create_graph=True` at full detector resolution is a GPU-memory risk; sequential backprop bounds peak graph memory but does not make it cost-free.
- Tests were run in CPU with the reconstructed supplied TDE source snapshot. **The user checkpoint, real source-clean data and Windows CUDA/AMP runtime were not available in this environment.**
