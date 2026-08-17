# TDE-YOLOX v2.0 (Simpler Reference Version)

> **Technical README for the simpler TDE implementation that produced the reported epoch-30 ACDC results.**
>
> This document describes the implementation **as it was run**, not the later TDE-vNext redesign. It also records known implementation caveats rather than silently correcting them, because this version is valuable precisely as an empirical reference point.

---

## 1. Executive summary

This TDE-YOLOX variant extends **YOLOX-S** with three experimental mechanisms:

1. **Functional test-time training (TTT) in Dark2**  
   A shallow GroupNorm-based stage performs one functional inner-loop update on norm parameters using a masked denoising reconstruction objective.

2. **Tribrid sparse-global neck**  
   Every PAFPN CSP block is replaced by `C2f_Tribrid`, which preserves a dense local bottleneck path and adds a globally contextual branch:
   `SimAM -> CoordinateAttention -> MBConvConditioner -> DeepSeekSparseAttention`.
   The global branch is injected through a learnable scalar residual gate initialized to **0.01**.

3. **Engram associative-memory classification head**  
   Each FPN scale projects classification features into a 128-dimensional latent space, retrieves an orthogonally initialized class prototype on a hypersphere, gates the retrieval by uncertainty and objectness, and fuses it into the **classification branch only**. Standard YOLOX regression/objectness remain memory-free.

The reported epoch-30 cross-condition result is **mixed but scientifically informative**:

- **Snow:** +1.1 AP points over baseline.
- **Night:** +1.3 AP points.
- **Rain:** +1.2 AP points.
- **Fog:** −5.1 AP points.

Across snow/night/rain only, mean AP improves by **+1.2 points**. Across all four conditions, the fog failure pulls the unweighted mean from **18.525% baseline AP to 18.150% TDE AP**, a net **−0.375 point** change.

That pattern is why this simpler version is a strong **reference architecture** rather than evidence that every TDE component is universally beneficial.

---

## 2. What this README documents

This README is intended to answer four practical questions:

- **What exactly is implemented?**
- **How do data flow and gradients move through the three TDE components?**
- **What settings produced the reported results?**
- **What should a researcher know before reproducing, modifying, or citing this version?**

It deliberately distinguishes:

- **implemented behavior** from comments/names in the code;
- **observed results** from causal interpretation;
- **reproduction of the historical run** from a strict clean-source-only ZSDA protocol.

---

## 3. Architecture at a glance

```text
Input image
   |
   v
YOLOX Focus stem (standard BN)
   |
   v
Dark2: GN-based CSP stage
   |
   +---- optional one-step functional TTT on GN/norm parameters
   |
   v
Dark3 / Dark4 / Dark5
standard YOLOX-style BN stages
   |
   v
PAFPN
  C3_p4  -> C2f_Tribrid
  C3_p3  -> C2f_Tribrid
  C3_n3  -> C2f_Tribrid
  C3_n4  -> C2f_Tribrid
   |
   +---- local dense path
   +---- sparse global path
          SimAM
            -> Coordinate Attention
            -> MBConv Conditioner
            -> 10% sparse K/V attention
          multiplied by tanh(gate), gate init = 0.01
   |
   v
P3 / P4 / P5
   |
   +-------------------------+
   |                         |
   v                         v
Regression/Objectness      Classification
standard YOLOX             cls feature
memory-free                  |
                              v
                        latent projection (128)
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          uncertainty gate             objectness mask
                 |                         |
                 +------------+------------+
                              |
                              v
                  hyperspherical Engram lookup
                              |
                              v
                    classification restoration
                              |
                              v
                         class logits
```

The wrapper-level training loss is:

```text
total_loss = YOLOX_detection_loss + 0.05 * memory_anchor_loss
```

There is no Wasserstein regression term in this reference version.

---

## 4. Main source files

The supplied implementation is organized around the following modules.

| File | Responsibility |
|---|---|
| `yolox/models/darknet.py` | TTT-aware `CSPDarknet`; Dark2 is GN/adaptive, Dark3–Dark5 are standard BN |
| `yolox/models/ttt_modules.py` | `GRN`, `TTTProjector`, `TTTAdaptiveStage`, SimAM, Coordinate Attention, MBConv conditioner, sparse attention, Engram bank, uncertainty estimator |
| `yolox/models/tribrid_neck.py` | `C2f_Tribrid`, local/global dual-path neck block |
| `yolox/models/yolo_pafpn.py` | PAFPN with all four CSP blocks replaced by `C2f_Tribrid` |
| `yolox/models/engram_head.py` | `TDE_Head`, classification-only Engram augmentation |
| `yolox/models/yolo_head.py` | standard YOLOX head machinery / SimOTA assignment |
| `yolox/models/losses.py` | standard YOLOX IoU/GIoU loss |
| `yolox/models/yolox.py` | unified wrapper, TTT probability schedule, memory-anchor loss |
| `yolox/core/trainer.py` | epoch-to-model state synchronization and TDE metric logging |
| `exps/default/tde_yolox_s.py` | YOLOX-S-scale TDE experiment construction and training settings |

---

## 5. Model scale and training configuration

The reported research configuration is YOLOX-S scale:

| Parameter | Value |
|---|---:|
| depth multiplier | `0.33` |
| width multiplier | `0.50` |
| activation | `SiLU` |
| classes | `9` |
| initial TTT LR | `0.02` in experiment |
| TTT feature noise std | `0.08` |
| configured max epochs | `80` |
| warm-up epochs | `10` |
| no-augmentation epochs | `15` |
| minimum LR ratio | `0.05` |
| base LR per image | `0.01 / 64` |
| weight decay | `0.0005` |
| momentum | `0.9` |
| configured batch size | `16` |
| gradient accumulation | `8` |
| test size | `640 x 640` |
| test confidence | `0.01` |
| NMS threshold | `0.65` |
| EMA | enabled |

The results documented here are from **epoch 30**, even though the experiment is configured for up to 80 epochs.

### Class set

The nine configured BDD-style categories are:

```text
car
bus
truck
person
rider
bike
motor
traffic light
traffic sign
```

Some ACDC condition subsets contain no valid instances for certain categories; those classes appear as `nan` in the per-class evaluator output and should not be numerically averaged as if they were zeros.

---

# 6. Component 1 — shallow functional TTT

## 6.1 Where adaptation occurs

Only **Dark2** is rebuilt around GroupNorm-aware blocks.

```text
stem:  standard BN
Dark2: GN + TTT wrapper
Dark3: standard BN
Dark4: standard BN
Dark5: standard BN
```

The TTT wrapper selects parameters whose names contain:

```python
'gn' or 'norm'
```

and learns a separate scalar inner-step learning rate for each selected parameter.

This means the intervention is structurally shallow and localized: deeper semantic stages are not directly rewritten at inference.

---

## 6.2 TTT projector

`TTTProjector` is:

```text
1x1 Conv
 -> GroupNorm
 -> GELU
 -> GRN
 -> 1x1 Conv
```

and owns a learned `mask_token`.

Its purpose is to reconstruct the current Dark2 feature representation from a feature tensor that has been corrupted by:

1. additive Gaussian noise; and
2. region masking.

---

## 6.3 The actual masking behavior

The function is called with:

```python
ratio = 0.85
threshold = quantile(scores, 0.85)
mask = scores >= threshold
```

Therefore, ignoring ties, it masks approximately the **highest 15%** of pooled variance locations — **not 85% of all positions**.

This is one of the most important documentation corrections for the historical code.

The comment "`85% Masking`" describes the quantile threshold, not the resulting masked fraction.

Formally:

```text
v(h,w) = Var_c(F[:,c,h,w])
v_pool = AvgPool_4x4(v)
q = Quantile_0.85(v_pool)
M(h,w) = 1[v_pool(h,w) >= q]
```

so:

```text
P(M = 1) ≈ 0.15
```

under a continuous score distribution.

This likely makes the historical TTT substantially less destructive than its comments imply.

---

## 6.4 Inner-loop objective

Let:

- `S_phi` be the Dark2 stage,
- `P_psi` be the projector,
- `F0 = S_phi(x)` be the initial feature,
- `epsilon ~ N(0, sigma^2)` be feature noise,
- `M` be the variance mask,
- `m` be the learned mask token.

The corrupted feature is approximately:

```text
F_corrupt = (S_phi(x) + epsilon) * (1 - M) + m * M
```

and the inner loss is:

```text
L_TTT = MSE(P_psi(F_corrupt), stopgrad(F0))
```

The code then computes the gradient only with respect to the adaptable backbone norm parameters and performs:

```text
phi' = phi - alpha_phi * grad_phi(L_TTT)
```

where each selected norm parameter can have its own learned scalar `alpha_phi`.

The final Dark2 output for that sample uses the **functional** parameter dictionary `phi'`.

The base module parameters are not permanently overwritten.

---

## 6.5 Episodic inference behavior

At inference:

```text
TTT probability = 1.0
```

so the stage performs a one-step adaptation for every test image unless explicitly bypassed.

The implementation forces a gradient-enabled context internally even if surrounding evaluation is running under `no_grad`.

After the adapted forward, the functional parameter dictionary is discarded. Therefore adaptation is image-episodic rather than continually accumulating across the test stream.

---

## 6.6 Training-time TTT schedule

The wrapper declares:

```text
warmup_end = 10
ramp_end   = 60
prob_start = 0.1
prob_end   = 0.5
```

but the actual active implementation after warm-up is:

```python
progress = (epoch - warmup_end) / (max_epochs - warmup_end)
p = prob_start + (0.7 - prob_start) * progress
```

Therefore `ramp_end` and `prob_end` are not actually used by the active schedule after epoch 10.

The historical behavior is better documented as:

```text
training:
  before epoch 10: p_TTT = 0.1
  after epoch 10:  linearly approaches 0.7 at max_epochs

inference:
  p_TTT = 1.0
```

The trainer also contains two state updates in `before_epoch` (`epoch` and then `epoch+1`), producing a possible one-epoch offset in the schedule. Preserve this only when reproducing the exact historical run; remove it in a cleaned implementation.

---

# 7. Component 2 — Tribrid sparse-global neck

## 7.1 C2f_Tribrid

Each original PAFPN CSP aggregation block is replaced by a two-path block.

### Local path

A conventional dense bottleneck sequence remains intact:

```text
BaseConv -> channel split -> Bottleneck(s)
```

This provides a standard local representation even if the global branch is not useful.

### Global path

The deepest local feature is sent through:

```text
SimAM
 -> CoordinateAttention
 -> MBConvConditioner
 -> DeepSeekSparseAttention
```

and then fused by:

```python
y[-1] = y[-1] + tanh(gate) * global_context
```

with:

```python
gate = 0.01
```

at initialization.

Because `tanh(0.01) ≈ 0.01`, the experimental global path begins at roughly **1% residual amplitude**.

This is a crucial property of the simpler TDE: the global module is present and trainable, but it cannot dominate the detector before earning a larger gate.

---

## 7.2 Sparse attention mechanics

For a feature map with:

```text
N = H * W
```

tokens, the module uses:

```text
K = max(1, floor(0.1 * N))
```

so only 10% of spatial positions are selected as keys and values.

The module:

1. scores all spatial positions with a depthwise-3x3 + pointwise-1x1 indexer;
2. chooses hard Top-K indices;
3. computes `Q, K, V` with a 1x1 projection;
4. keeps **all Q tokens**;
5. gathers only Top-K `K/V`;
6. computes attention of shape `[B, HW, K]`;
7. projects the reconstructed global context.

Conceptually:

```text
Q: all N positions
K,V: selected 0.1N positions
attention cost: O(NK), not O(N^2)
```

---

## 7.3 Important sparse-indexer caveat

As implemented:

```python
scores = indexer(x)
_, topk_indices = topk(scores, K)
```

The score values are discarded. Only the discrete integer indices are used to gather keys and values.

Therefore, ordinary detector gradients do **not** provide a smooth differentiable path from the final detection loss through the Top-K indices into the indexer score values.

Consequently:

- the global attention branch itself is active;
- Q/K/V/projection layers are trainable;
- the scalar fusion gate is trainable;
- but the saliency indexer should **not automatically be assumed to have learned task-optimal token importance**.

This limitation matters when interpreting negative or null sparse-attention ablations.

It is one of the central fixes in TDE-vNext, but it remains part of the historical simpler version documented here.

---

# 8. Component 3 — Engram associative memory

## 8.1 Scope

Engram modifies the **classification branch only**.

For each P3/P4/P5 scale:

```text
reg_feat -> box regression + objectness      (memory-free)
cls_feat -> Engram restoration -> cls logits
```

This separation is important because an uncertain semantic memory lookup is not allowed to directly overwrite bounding-box coordinates.

---

## 8.2 Training-time classification corruption

During training, before Engram retrieval:

```python
cls_feat += GaussianNoise(std=0.05)
spatial_keep_mask = Bernoulli(0.85)
cls_feat *= spatial_keep_mask
```

This is mild compared with later TDE variants.

The intent is to make the classifier learn to use memory when the observed representation is partially corrupted.

---

## 8.3 Latent projection

Each scale projects the classification feature:

```text
C -> 128
```

using a linear layer after flattening the spatial grid.

Let:

```text
z_i in R^128
```

be the latent vector at spatial position `i`.

---

## 8.4 Prototypes

Each FPN scale owns one prototype per class:

```text
P in R^(num_classes x 128)
```

The prototypes are initialized with:

```python
nn.init.orthogonal_
```

and both the query and prototypes are L2-normalized for retrieval.

The retrieval similarity is:

```text
s_ic = temperature * < normalize(z_i), normalize(p_c) >
```

with fixed:

```text
temperature = 50
```

followed by softmax across classes.

---

## 8.5 Uncertainty and objectness gating

The uncertainty estimator is:

```text
Linear(128 -> 32)
 -> ReLU
 -> Linear(32 -> 1)
 -> Sigmoid
```

and returns:

```text
u_i in [0,1]
```

Objectness is obtained from the regression branch:

```text
o_i = sigmoid(objectness_logit_i)
```

The memory bank itself returns:

```text
m_i_retrieved * u_i * o_i
```

The head then performs another uncertainty-controlled fusion:

```text
F_restored = (1 - u_i) * F_observed + u_i * F_memory
```

Since `F_memory` has already been multiplied by `u_i * o_i`, the effective memory term is approximately proportional to:

```text
u_i^2 * o_i
```

before the feature-space projection details.

That makes this historical Engram unexpectedly conservative: uncertain observations open the memory path, but the memory contribution grows quadratically with the learned uncertainty gate and remains suppressed where objectness is small.

---

## 8.6 Memory-anchor supervision

After SimOTA assignment, the head returns:

```text
classification targets
foreground masks
raw memory retrieval logits
```

and the wrapper applies:

```text
L_mem = BCEWithLogits(memory_logits_foreground, SimOTA_cls_targets)
```

with:

```text
L_total = L_detector + 0.05 * L_mem
```

Thus the intended identity memory is tied to the **same foreground assignments and IoU-weighted class targets used by YOLOX**, while keeping its auxiliary weight small.

This is arguably the strongest design principle in the simpler TDE: the memory does not learn a handcrafted weather statistic; it is directly anchored to the actual source detection task.

---

# 9. Standard detection objective

This reference version uses the standard YOLOX IoU loss:

```text
L_IoU = 1 - IoU^2
```

with standard objectness BCE, classification BCE, and optional L1 in the no-augmentation phase.

The detection loss in the supplied head is:

```text
L_det =
    5 * L_IoU
  + 1 * L_obj
  + 1 * L_cls
  + L_L1 (when enabled)
```

and the wrapper adds:

```text
+ 0.05 * L_mem
```

No SWAWIoU/Wasserstein loss is used here.

---

# 10. Training and evaluation workflow

## 10.1 Model construction

The experiment explicitly constructs:

```text
CSPDarknet(
  depth=0.33,
  width=0.50,
  ttt_lr=0.02,
  ttt_noise_std=0.08
)
```

then creates the PAFPN and overrides its internally created backbone:

```python
neck.backbone = backbone
```

Finally:

```text
YOLOX(
  backbone = TTT-aware Tribrid PAFPN,
  head     = TDE_Head
)
```

---

## 10.2 Typical training command

The exact CLI depends on the surrounding YOLOX checkout, but the standard YOLOX-style invocation is typically:

```bash
python tools/train.py   -f exps/default/tde_yolox_s.py   -d 1   -b 16   --fp16
```

If your checkout uses different flags, keep the experiment configuration itself authoritative.

### Important

The historical experiment config points `val_ann` and `test_ann` at an adverse annotation file.

That is acceptable for reconstructing the historical run, but **not** for a strict clean-source-only ZSDA claim if adverse validation affects:

- best-checkpoint selection;
- hyperparameter tuning;
- architecture decisions.

For strict ZSDA, use a source-only held-out validation split, freeze the model, and evaluate adverse sets only afterward.

---

## 10.3 Evaluation

Condition-specific ACDC evaluation uses the trained checkpoint against snow, night, rain, and fog subsets.

A typical YOLOX-style evaluation command is:

```bash
python tools/eval.py   -f exps/example/custom/test_acdc_snow.py   -c YOLOX_outputs/<experiment>/epoch_30_ckpt.pth   -b 1   -d 1   --conf 0.01
```

Repeat with the night/rain/fog experiment file.

Use the actual paths and experiment names present in your checkout.

---

# 11. Reported epoch-30 results

## 11.1 Main condition comparison

| Condition | TDE AP | Base AP | Δ AP (pp) | TDE AP50 | Δ AP50 | TDE AP75 | Δ AP75 | TDE AR100 | Δ AR100 | TDE time (ms) | Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Snow | 17.3 | 16.2 | +1.1 | 34.2 | +4.3 | 16.4 | +1.4 | 27.6 | +0.1 | 20.39 | 1.80× |
| Night | 12.0 | 10.7 | +1.3 | 25.7 | +2.7 | 9.2 | +0.6 | 21.8 | +1.2 | 15.89 | 1.90× |
| Rain | 15.9 | 14.7 | +1.2 | 27.9 | +0.0 | 17.1 | +3.1 | 27.2 | +4.3 | 15.09 | 1.94× |
| Fog | 27.4 | 32.5 | -5.1 | 46.4 | -3.9 | 24.5 | -9.0 | 41.0 | -1.4 | 15.01 | 1.94× |

### Aggregate interpretation

- Mean AP across **snow/night/rain**:
  - TDE: **15.067%**
  - Baseline: **13.867%**
  - gain: **+1.200 pp**

- Mean AP across **all four conditions**:
  - TDE: **18.150%**
  - Baseline: **18.525%**
  - difference: **-0.375 pp**

The correct conclusion is therefore **not** "TDE universally beats YOLOX." The evidence supports:

> The simpler TDE exhibits useful positive transfer in snow, night, and rain at epoch 30, but has a major fog-specific regression that dominates the four-condition mean.

---

## 11.2 Scale-specific AP deltas

| Condition | Δ AP-small | Δ AP-medium | Δ AP-large |
|---|---:|---:|---:|
| Snow | +0.5 | -0.6 | -3.6 |
| Night | -0.7 | +3.1 | -0.5 |
| Rain | +0.8 | +2.2 | -0.6 |
| Fog | -3.0 | +1.6 | -12.5 |

The fog failure is especially severe for large objects:

```text
AP-large: 34.4 vs 46.9  => -12.5 pp
AP75:     24.5 vs 33.5  =>  -9.0 pp
```

while fog medium-object AP is actually:

```text
42.0 vs 40.4 => +1.6 pp
```

This suggests the fog regression is not a simple uniform classification failure.

---

## 11.3 Per-class observations

- **Snow:** strongest AP changes: `bike` +10.697, `rider` +3.607, `bus` +1.299; weakest: `person` -1.734, `truck` -2.969, `motor` -4.264.
- **Night:** strongest AP changes: `truck` +4.694, `rider` +3.073, `person` +0.970; weakest: `bike` +0.235, `motor` -0.115, `car` -1.154.
- **Rain:** strongest AP changes: `motor` +7.635, `bus` +2.859, `car` +1.025; weakest: `person` -0.236, `truck` -0.653, `rider` -2.489.
- **Fog:** strongest AP changes: `bike` +2.436, `rider` -0.379, `car` -1.436; weakest: `truck` -4.531, `motor` -5.567, `bus` -22.534.

Two particularly visible examples:

- **Snow bike AP:** `0.743 -> 11.440`, a **+10.697 AP-point** change.
- **Fog bus AP:** `58.570 -> 36.036`, a **−22.534 AP-point** change.

Treat these as diagnostics rather than causal proof, because class counts and condition-specific support differ.

---

# 12. Internal evaluation stability at epochs 27–30

The supplied internal evaluator reports:

| Epoch | AP | AP50 | AP75 | AR100 |
|---:|---:|---:|---:|---:|
| 27 | 0.249 | 0.476 | 0.223 | 0.367 |
| 28 | 0.249 | 0.476 | 0.224 | 0.369 |
| 29 | 0.250 | 0.478 | 0.222 | 0.371 |
| 30 | 0.251 | 0.480 | 0.223 | 0.371 |

This is a stable, slowly improving plateau rather than an obvious optimization collapse around epoch 30.

That matters when interpreting the adverse results: the fog regression cannot simply be dismissed as "training blew up at epoch 30."

---

# 13. Interpreting why the simpler version can work

The strongest explanation supported by the implementation is **conservatism**.

## 13.1 The sparse-global branch can be mostly ignored

It enters through:

```text
local + tanh(0.01) * global
```

at initialization.

Therefore, even if the sparse global branch is initially noisy, the local detector remains dominant.

## 13.2 TTT modifies only a shallow subset

Only GN/norm parameters in Dark2 are adapted, one step at a time, without permanently rewriting the source model.

Moreover, the historical "85% mask" is actually close to a **15% high-variance mask**, so the intervention is milder than its comments imply.

## 13.3 Engram is task-supervised and classification-only

The memory:

- does not directly alter regression;
- uses low auxiliary weight `0.05`;
- uses foreground SimOTA targets;
- begins with orthogonal prototypes;
- is gated by both objectness and uncertainty;
- effectively receives approximately `u^2 * objectness` strength in the memory term.

Of the three components, Engram has the clearest direct path from detector supervision to its intended role.

---

# 14. Known implementation caveats

These are part of the historical version and should be documented explicitly.

## 14.1 "85% masking" is actually approximately 15%

As explained above, the code thresholds at the 85th percentile and masks values **above** the threshold.

Do not report this as 85% spatial masking.

---

## 14.2 TTT scheduler fields do not match the active formula

Declared:

```text
ramp_end = 60
prob_end = 0.5
```

Active implementation:

```text
linear schedule toward 0.7 at max_epochs
```

---

## 14.3 Trainer epoch state is updated twice

`before_epoch` calls the TDE state update with both:

```text
epoch
epoch + 1
```

depending on the code path.

That can introduce an off-by-one schedule difference.

---

## 14.4 Hard Top-K indexer is not cleanly task-differentiable

The indexer produces values, but only its hard Top-K indices are consumed by the attention branch.

Do not describe the selector as proven to be a learned detector-utility router without additional evidence.

---

## 14.5 Engram retrieval geometry and auxiliary-logit geometry differ

Actual memory retrieval uses:

```text
normalized query
normalized prototype
temperature = 50
softmax
```

but the auxiliary supervision logits are:

```python
latent_vec @ prototypes.T
```

without the same normalization or temperature.

So the training anchor is not geometrically identical to the retrieval rule.

---

## 14.6 Objectness is not detached inside the memory gate

The historical code computes:

```python
obj_mask = sigmoid(obj_output)
```

and feeds it into the memory path without `.detach()`.

Therefore the auxiliary/classification memory computation can, in principle, send gradients back toward the objectness branch through the gate.

This weakens the conceptual claim that the Engram mechanism is *strictly* isolated from objectness, although box regression features remain separately computed.

---

## 14.7 Potential batch-order mismatch in memory-anchor supervision

This is the most important latent correctness issue in the historical Engram loss.

`fg_masks` are concatenated in **batch-major anchor order**:

```text
image 0: P3 | P4 | P5
image 1: P3 | P4 | P5
...
```

but the historical auxiliary logits are built by flattening each FPN scale separately and then concatenating scales:

```text
P3: all images
P4: all images
P5: all images
```

For `B > 1`, the element counts still match, so this can fail silently while associating some foreground masks with the wrong memory logits.

Therefore:

> Reproducing the historical code is not the same as endorsing this ordering as correct.

TDE-vNext should concatenate the scale dimension inside each batch first and only then flatten.

---

## 14.8 Historical validation configuration is not strict ZSDA-safe

The experiment points validation and test annotations at an adverse set.

If adverse AP was used to choose the checkpoint or tune the architecture, the experiment is no longer a clean strict zero-shot protocol even though adverse images are absent from backpropagation.

For future strict-ZSDA claims:

```text
source train -> source val -> freeze -> target test
```

---

# 15. Reproduction checklist

Before calling a run a reproduction, verify all of the following.

- [ ] YOLOX-S `depth=0.33`, `width=0.50`.
- [ ] Nine configured classes.
- [ ] Dark2 uses GN-based components and is wrapped in `TTTAdaptiveStage`.
- [ ] Dark3/Dark4/Dark5 use ordinary BN-based YOLOX blocks.
- [ ] Experiment overrides `neck.backbone` with the configured TTT backbone.
- [ ] All four PAFPN CSP aggregation blocks are `C2f_Tribrid`.
- [ ] Tribrid global residual gate initializes to `0.01`.
- [ ] Sparse attention uses fixed `10%` Top-K K/V.
- [ ] Engram latent dimension is `128`.
- [ ] Prototypes use orthogonal initialization.
- [ ] Retrieval temperature is `50`.
- [ ] Classification corruption is `0.05` Gaussian noise and `15%` spatial dropout.
- [ ] Standard IoU loss `1 - IoU^2` is used.
- [ ] Memory anchor weight is `0.05`.
- [ ] Training-time TTT probability follows the actual wrapper code, not only comments.
- [ ] Inference TTT runs for every test image.
- [ ] Reported checkpoint is epoch 30 when comparing against the provided numbers.
- [ ] Exact evaluator confidence and NMS settings are `0.01` and `0.65`.
- [ ] Results are reported condition by condition, including the fog regression.
- [ ] Historical code caveats are disclosed rather than silently patched.

---

# 16. Recommended diagnostics for any new run

Log at least:

```text
total_loss
iou_loss
conf_loss
cls_loss
mem_loss
ttt_prob
```

and add the following research diagnostics if possible:

### TTT

```text
inner reconstruction loss
norm-gradient magnitude
learned TTT LR distribution
feature delta ||F_after - F_before||
TTT-on vs TTT-off AP
```

### Tribrid

```text
tanh(gate) per C2f_Tribrid block
selected token overlap between clean/perturbed views
foreground recall within Top-K
```

### Engram

```text
mean uncertainty
mean objectness gate
mean effective memory coefficient
prototype pairwise cosine similarity
foreground memory retrieval accuracy
per-class memory-anchor loss
```

Without these diagnostics, a full-model AP change cannot tell us which component actually contributed.

---

# 17. Relationship to TDE-vNext

This simpler TDE should be preserved as a **reference branch/checkpoint**.

It contains several design principles worth carrying forward:

```text
standard YOLOX regression
shallow localized adaptation
safe residual attention authority
classification-only memory
low-weight detector-aligned memory supervision
mild feature corruption
orthogonal source semantic prototypes
```

The later TDE-vNext patches should be evaluated against both:

```text
R0: plain YOLOX
Rref: this simpler TDE
```

rather than only against the latest complex architecture.

---

# 18. Scientific interpretation in one paragraph

The simpler TDE-YOLOX v2.0 demonstrates that a conservative combination of shallow episodic normalization adaptation, low-authority global context, and detector-supervised semantic memory can improve YOLOX transfer to several unseen adverse conditions without replacing the detector's core regression objective. At epoch 30 it improves AP on snow, night, and rain by approximately 1.1–1.3 points, with particularly large class-level gains in some difficult categories, but it fails substantially on fog, especially at AP75 and large-object AP. The implementation therefore provides a useful positive reference and a source of design principles, but not evidence of universal adverse-weather robustness. Its main value is that experimental mechanisms are comparatively constrained: the global branch begins near zero authority, TTT is localized to Dark2, and Engram receives weak foreground-aligned supervision. Several code-level caveats — especially the Top-K indexer gradient issue, TTT schedule inconsistencies, and potential batch-order mismatch in memory supervision — should be corrected in any next-generation implementation rather than attributed to the underlying concept.

---

## License / upstream notice

This document describes a modification built around a YOLOX-style codebase. Preserve the upstream project's copyright notices and licensing terms in any redistributed code.

