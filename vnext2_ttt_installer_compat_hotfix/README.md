# vNext.2 TTT: syntax-aware installer compatibility hotfix

## Why the old installer rejected the source

The prior installer relied on this exact text appearing in `yolox/models/ttt_modules.py`:

```python
base_feat = self.backbone_stage(x_in) if precomputed_feat is None else precomputed_feat
if not run_ttt:
    return base_feat
```

Your **frozen-EMA evaluation hotfix** instead wraps `base_feat` in a parenthesized, multiline conditional expression, followed by the same `if not run_ttt:` return. Python semantics are identical, but the literal text anchor fails. The old script correctly refused to modify an unrecognized source; the failure does **not** mean the TTT architecture is incompatible.

## What this package changes

- Replaces only `tools/install_vnext2_per_image_ttt.py` with a syntax-aware installer.
- Includes the same `tools/vnext2_ttt_per_image_method.txt` from the prior patch.
- Uses Python's AST to locate `TTTAdaptiveStage._forward_residual_ttt`, validate its signature and the original `base_feat` / `run_ttt` early-return sequence.
- Inserts `_forward_residual_ttt_per_image_eval` immediately before the existing method and a **single eval-only** dispatch after the no-TTT return.
- Checks that the existing method's entire AST is unchanged once the dispatch is ignored. Training code, fast-weight math, checkpoint keys and persistent model weights remain untouched.
- Supports Windows CRLF line endings and writes a timestamped `.bak` file before atomic replacement.
- Is idempotent: running `--apply` twice does not duplicate code.
- Refuses ambiguous, partially-installed, or unexpectedly changed structures instead of editing aggressively.

## Installation (PowerShell in `D:\YOLOX-3rd`)

Extract this archive and copy **both** bundled files from `tools/` to the repo's `tools/` directory. Overwrite the **old installer** file, not the `yolox/models/ttt_modules.py` file.

```powershell
Copy-Item .\vnext2_ttt_installer_compat_hotfix\tools\install_vnext2_per_image_ttt.py .\tools\ -Force
Copy-Item .\vnext2_ttt_installer_compat_hotfix\tools\vnext2_ttt_per_image_method.txt .\tools\ -Force

python tools/install_vnext2_per_image_ttt.py
python tools/install_vnext2_per_image_ttt.py --diff
python tools/install_vnext2_per_image_ttt.py --apply
python tools/install_vnext2_per_image_ttt.py
```

Expected sequence: `CHECK_PASSED`, unified diff, `PATCHED` with backup path, `ALREADY_INSTALLED`.

After application:

```powershell
python tests/test_vnext2_per_image_eval.py
python tests/test_mask_objective_probe.py
python tests/test_refined_modules.py
```

The first two tests are from the previously delivered **vnext2_ttt_eval_independence_fix.zip** bundle; copy them into the repo if not already present.

Run one CUDA/EMA evaluation before resuming training:

```powershell
python tools/eval.py `
  --model-exp exps/default/tde_yolox_s.py `
  --data-exp exps/default/test_acdc_fog.py `
  -c YOLOX_outputs/tde_yolox_s_full_vnext2/latest_ckpt.pth `
  --weights ema --ttt-mode on -b 16 --conf 0.01
```

The `--ttt-batch-policy per_image` CLI option exists only if you separately applied the prior optional `02_eval_policy_and_logging.patch`; the model patch **defaults to per-image eval mode even without** that CLI change.

## Important boundaries

- This is **not** a vNext.3 or TTT learning-objective redesign.
- This fixes source installation and eval-time batch coupling only; it does not claim stronger ACDC AP or eliminate the previously measured small TTT updates.
- The user's actual repo is not mounted in this environment. Validation was done against a full vNext.2 source snapshot into which their **exact supplied `_forward_residual_ttt` method** was substituted, plus original compact vNext.2 source.
- On this constructed frozen-EMA source, all 6 per-image regression tests passed on CPU, including `train()` gradient checks and frozen-EMA `torch.no_grad()` inference. Actual checkpoint/CUDA evaluation remains to be checked locally.
- **Do not apply the older `01_ttt_per_image_eval_only.patch` in addition to this installer.** Both implement the same model change.
- If another error occurs, retain the `.bak` backup and show the error without manually deleting portions of the original TTT method.
