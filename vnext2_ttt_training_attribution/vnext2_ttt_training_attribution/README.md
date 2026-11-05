# vNext.2 Safe TTT — Parameter-family attribution on the two-step pilot

**No training, no model writes, no online target labels.** Performs a 2⁴ factorial weight-family swap between the exact seeded initial sidecar and your two-step-trained sidecar on the *same* clean BDD source TRAIN images. This tests **what changed in the pilot** before any new architectural or training run. It does **not** show unbiased adverse-domain generalization.

## Why this exists

From the user's eight-image paired sweep at a 0.5× step:

- Untrained mean detection-loss delta: **−1.149e−4**; trained: **+4.908e−5**; trained minus initial: **+1.640e−4**.
- All eight *paired* differences are positive: the two-step pilot worsened the measured finite-step result at that strength.
- But the average gradient cosine changed **−0.088 → −0.042** (slightly less negative). A difference in finite-step loss cannot be attributed to SSL-gradient **direction** alone.
- Strict scale/background surrogate passes remain low: **2/8 untrained**, **1/8 trained**.
- The earlier pilot changed **adapter**, **predictor**, **mask_token**, and **step_logit** simultaneously. This is the primary attribution confound.

The probe rebuilds 16 sidecars: one for each choice of which parameter families use trained weights and which retain their seeded initialization. **The downstream YOLOX model is unchanged.** At each image, native SimOTA targets are captured once and replayed for every candidate.

## Install (copy only two NEW Python files)

The existing YOLOX `tools/` directory must already include `audit_safe_ttt_yolox.py`, `pilot_safe_ttt_meta.py`, `safe_ttt_model.py`, `fixed_detection_targets.py`, and `native_dark2_router.py` from the earlier audit package.

```powershell
Copy-Item .\vnext2_ttt_training_attribution\tools\audit_ttt_training_attribution.py .\tools\ -Force
Copy-Item .\vnext2_ttt_training_attribution\tests\test_ttt_training_attribution.py .\tests\ -Force
python tests/test_ttt_training_attribution.py
```

The unit tests verify group assignment, strict state blending, unknown-key rejection, 16-way matrix completeness, and Shapley decomposition, but do **not** execute a real GPU YOLOX model.

## Exact reproducibility-checked run

Pass BOTH earlier sweep JSONs. This is particularly important because the original `pilot_safe_ttt_meta.py` **did not save the initialization seed in the sidecar configuration**. The audit validates reconstructed initial and final results against the original sweeps and fails if they differ by more than `--reference-tolerance`.

```powershell
python tools/audit_ttt_training_attribution.py `
  --checkpoint YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --model-exp exps/default/tde_yolox_s.py `
  --weights ema `
  --trained-sidecar audits/safe_ttt_meta_pilot_sidecar.pth `
  --indices 10 37 53 79 101 137 181 223 `
  --strength 0.5 --seed 20261104 `
  --reference-initial audits/safe_ttt_step_sweep_untrained.json `
  --reference-trained audits/safe_ttt_step_sweep_pilot.json `
  --output audits/safe_ttt_training_attribution.json
```

The tool refuses to overwrite an existing JSON, mismatched native checkpoint/state hash, non-clean source data, unknown sidecar keys, overlapping training indices, nonfinite gradients, failed zero-state parity, failed head-loss parity and unacceptable reproduction of the two prior sweeps.

The tool checks **repeatability of the native forward** and of the **initial/full-trained adapted passes** on the same input. Tiny loss improvements smaller than observed repeatability noise should not be used for model selection. No single absolute epsilon can establish significance; assess error magnitude, variation across images and independent seeds.

## Output interpretation

`parameter_changes` reports changed tensors, maximum absolute state changes, and absolute L2 changes for each family. Beware of huge `relative_l2` when a parameter begins at zero (e.g., mask token).

`images[*].groups`: all 16 combinations, including `initial` (all initial weights) and `adapter+predictor+mask_token+step_logit` (all trained weights). Each records source fixed-target detection-loss delta, SSL/detector cosine, original compact gradient norms, step magnitude, feature change, Taylor-predicted delta, box/cls/objectness loss changes, GT-scale changes, and strict surrogate flags.

`attribution.shapley_mean_loss_effects`: an **exact decomposition** of the trained-minus-initial *mean finite-step loss difference* across these four weight families. Positive means that family's marginal contribution across the 16 hybrids increases the measured loss delta. Shapley *shares* interaction effects among groups; it **does not prove a parameter family intrinsically causes poor generalization**, because hybrids may be off the trained distribution.

### Decision interpretation

| Observation | What to investigate |
|---|---|
| Adapter dominates harmful Shapley effect | Frozen basis first; train only predictor and recheck gradient alignment |
| Predictor dominates | SSL objective/meta-learning data or gradient formulation is failing; hold spatial basis fixed |
| Step-logit dominates | Step calibration, not necessarily direction; source-calibrate update strength/abstention |
| Mask token dominates | Masked feature-prediction bias; test source-conditioned masking/target leakage assumptions |
| Large cross-family interactions | Do not interpret one-group swap in isolation; jointly designed training may be necessary |
| Observed repeatability error approaches total loss effect | Improve numerical resolution/reliability before any optimization claim |

If training only changes behavior at 1e−5–1e−4, avoid another entire YOLOX epoch. Next controlled pilot should freeze the native detector, freeze the adapter and step strength, train the SSL predictor on **diverse disjoint clean-source episodes**, evaluate on a separate disjoint source holdout, then (only if supported) unlock adapter/step modules.

## Validation status

- Tested CPU-only PyTorch helper tests: **passed**.
- `py_compile`: **passed**.
- Actual checkpoint and full CUDA source audit: **not run** in this container. The entire tool is read-only and designed to fail closed if source semantics differ.
- No vNext.3 integration, no changes to `ttt_modules.py`, `darknet.py`, `trainer.py`, EMA or trained model.
