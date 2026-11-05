# Phase B: gradient-path and research acceptance review

## Train/test boundary

During **training**, labels and paired views only enter the **outer task loss**. There are two independent unlabeled inner TTT episodes: one on clean features and one on shifted features. The clean view does not contaminate the shifted online gradient. After the two backward passes, gradients of the **same predictor weights** are combined and only the predictor is updated.

At **test time**, there is exactly one unlabeled input; the sidecar runs the same `run_episode(meta_train=False)` as prior. The outer clean/shift and optional teacher losses do not exist in test-time adaptation. Episodic fast state is reset, no EMA/BN states are mutated, and `enabled=False` remains native YOLOX.

## Differentiation

Let `F_c`, `F_s` be native dark2 features of two geometry-identical source views. For each view `v`, initialize fast state `z_v=0`, compute 

    g_v = grad_z SSL(F_v, z_v)
    z'_v = z_v + bounded_step(g_v)
    pred_v = frozen_detector(F_v + spatial_adapter(F_v,z'_v))
    L_v = det_fixed_assignments(pred_v, y) + lambda_ng * per_scale_nonreg_v

Then backpropagate `clean_weight*L_clean` and `1.0*L_shifted` **sequentially**, accumulating the exact mixed derivative into the predictor. `optimizer.step()` occurs once. `g_v` must retain its autograd graph to the predictor (`create_graph=True`) during training. Stopgradient on that update would reproduce the old failure.

The spatial adapter, its finite bound and rank, mask token, and online step-logit are frozen during Phase B. No head weights, memory banks, prototype banks or backbone are optimized.

## Why fixed SimOTA targets matter

Native and adapted passes use identical IoU-derived classification targets and original GT assignments for **each individual view**. These targets can differ *between* clean and shifted views; that is intended, because each view has its own native reference. They MUST NOT change *within* a view's native-vs-adapted comparison. If this cannot be guaranteed by the current TDE head, the script fails closed.

## Scale and task non-regression

For each matched GT scale, sum box/classification/positive-objectness loss; divide by positive-anchor count in that scale before computing the one-sided non-regression penalty. Background objectness has its own per-negative-anchor normalization, not a fictitious small/medium/large attribution. Record all four task components separately. The source loss penalty is a differentiable proxy, not an AP or AR safety certificate.

## Why we did not train the teacher by default

The teacher's native clean logits may be imperfect at epoch 2. Forcing adapted shifted predictions toward them is likely to damage correct shifted predictions if the teacher is wrong. The optional teacher regularizer is restricted to source native positive anchors, uses soft probabilities, and has zero default weight. Do not infer its usefulness from the synthetic nonzero-gradient test.

## Positive result acceptance ladder

**A. Structural:** source-only, strict checkpoint, frozen weights unchanged, exact native parity, finite exact meta-gradients, same labels/geometry, deterministic source transformations; no higher-order OOM.

**B. Source episodic:** independent shifted and clean validation means improve relative to seeded initial sidecar, signal exceeds repeat/numerical error, per-GT-scale and task-component differences do not show systematic harm, holds over multiple seeds and disjoint source IDs.

**C. Clean benchmark:** use clean val without modifying native detector, compare TTT on/off, per-class and per-scale AP+AR, and measured inference latency; reject if increased complexity yields negligible improvements.

**D. Final DG:** freeze architecture and source calibration before running a genuinely uninspected adverse benchmark; report TTT off and on separately and don't use ACDC feedback for additional tuning.

Do **not** interpret a single 32-step source pilot as success. Its role is to falsify weak optimization hypotheses cheaply before any full 80-epoch training campaign.
