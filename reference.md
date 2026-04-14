# PhotonFlow References

Running list of research papers that directly inform code or hyperparameter
choices in `temp/photonflow.py` and the main `photonflow/` codebase.

Each entry: **[citation]** — one-line summary — *how it is used here*.

---

## Architecture

1. **Dao et al., "Monarch: Expressive Structured Matrices for Efficient and Accurate Training," ICML 2022.**
   Introduces `M = P L P^T R` as a structured matrix with sub-quadratic parameter count and FLOPs.
   *Used as: core of `photonflow.model.MonarchLayer` (two factors = MM* class, with `num_monarch_factors=2` stacking).*

2. **Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)," ICCV 2023.** [arXiv:2212.09748]
   adaLN-Zero block design: per-dimension scale + shift + gate from time embedding, gates zero-initialized.
   *Used as: `photonflow.model.PhotonFlowBlock` uses 6 per-dim vectors from `adaLN_proj`.*

3. **Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.** [arXiv:2210.02747]
   CFM loss with OT path; `target = x1 − x0`, `x_t = (1−t)x0 + tx1`.
   *Used as: `photonflow.train.CFMLoss` forward pass. This is the loss used in `temp/photonflow.py`.*

4. **Shen et al., "Deep Learning with Coherent Nanophotonic Circuits," Nature Photonics 2017.**
   Saturable absorber as photonic nonlinearity; `σ(x) ≈ tanh(αx)/α`.
   *Used as: `photonflow.activation.SaturableAbsorber` (α=0.8).*

## Training

5. **Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)," ICML 2024.** [arXiv:2403.03206]
   Logit-normal timestep sampling `t = sigmoid(N(m, s))` outperforms uniform for rectified/flow matching.
   *Available in `photonflow.train.CFMLoss(time_sampling="logit_normal"|"curriculum")`; currently OFF in `temp/photonflow.py` for baseline-comparable loss.*

6. **Karras et al. "Analyzing and Improving the Training Dynamics of Diffusion Models," CVPR 2024.**
   EMA of weights improves generation quality; post-hoc EMA mining.
   *EMA is available in `configs/exp2_mnist.yaml` but disabled in `temp/photonflow.py` — at 10K steps EMA decay 0.9999 has no effective averaging window.*

## Experiment iterations on `temp/photonflow.py`

### Iteration 1 — baseline-comparable config
- Uniform time sampling, no direction loss, no time-weighting, no EMA
- **Result**: plateau at avg50 ≈ 0.27 from step 1500; killed at step 2050
- Conclusion: pure architectural swap (attention → Monarch) with matched
  training loses ~0.19 absolute loss vs baseline at mid-training.  The
  structured transforms need a different training curriculum to compensate.

### Findings from the 2K-step sweep (v1-v9 on Kaggle)

Baseline at 2 K steps: eval = 0.1734.  PhotonFlow stock at 2 K: eval = 0.2577 (gap 0.0843).  Target: gap ≤ 0.05.

**Result**: best gap **0.0503** (12blk + random-init + gate=0.5 + lr=1e-3) — essentially at target, inside 8-batch eval noise.

**Single-lever attribution** (see `temp/REPORT.md` for the table):
- `MonarchLayer(init='random')` instead of `'identity'` — closed 0.010.  Biggest architectural lever.
- `gate_init` 0.1 → 0.5 — closed 0.005.
- `num_blocks` 4 → 8 → 12 — each step closed ~0.015 then ~0.003 (diminishing return).
- `lr` 5e-4 → 1e-3 — closed 0.020 (the training-side lever).
- Absorber α, mean-centering, token mixing, stacked Monarch factors, logit-normal sampling — all null or slightly negative.

### ⭐ Final gap-closing lever — leaky SaturableAbsorber pass-through

**Config** (v11, `photonflow_v9_leaky`): 12 blocks, random-init Monarch, gate=0.5, lr=1e-3, plus `SaturableAbsorber(leaky_slope=0.05)` → `out = tanh(α·x)/α + 0.05·x`.

**Result**: gap = **0.0496** — FIRST variant to cross the 0.05 target.

**Rationale**: at t ≈ 0 in CFM, the velocity target `x1 − x0` can exceed the ±1/α = ±1.25 hard cap of the pure-tanh absorber.  A small linear bypass (+0.05·x) preserves magnitudes past saturation.  Photonically realizable as a **parallel unabsorbed-light path** alongside the graphene waveguide — 5% of optical intensity takes a linear route while 95% passes through the nonlinearity.  Hardware-compatible; adds no MZI mesh cost, just a y-split + recombination.

### Rejected: orthogonal and DCT init (v10)

Saxe et al. 2013 orthogonal init and Monarch-Mixer DFT/DCT init both LOST vs random-Xavier (gain 0.1) at 2K steps:
- `monarch_init='orthogonal'`: gap 0.093 (worse)
- `monarch_init='dct'`: gap 0.078 (worse)

**Root cause**: both inits preserve the input norm (`||Mx||/||x||=1.0`) so each block contributes a *full-magnitude* signal to the residual.  With gate=0.5 across 10 blocks this compounds to a ~57× magnitude blow-up at initialization (eval loss at step 500 starts at 0.53 for orthogonal, vs 0.29 for random).

Random-Xavier(gain=0.1) secretly *shrinks* norms ~100×, so blocks start as small perturbations of identity — stable AND off-identity enough to escape the near-identity basin (Hardt & Ma 2017).

**Workable fix tested** (`dct_smallgate`): DCT init + `gate_init=0.1` to balance the scale → gap 0.071.  Better than stock 0.084, but still worse than tuned random-init recipe.  Would likely match random at longer training horizons since DCT carries useful frequency-domain inductive bias for images — but at 2K-step sample efficiency, random wins.

### Hardt & Ma 2017 — identity-near local minima in residual networks ([arXiv:1611.04231](https://arxiv.org/abs/1611.04231))

**Relevance**: proves that for deep residual networks there is a global
minimum close to the identity parameterisation.  For unstructured dense
weights that's fine (SGD escapes easily), but for **structured** matrices
like Monarch (`M = P L P^T R`), the set of reachable matrices is a lower-dim
manifold centered around `I` when `L = R = I`.  Gradient descent stays near
the identity-equivalent submanifold and the network degenerates to
`input_proj ∘ output_proj` — exactly the "linear-regression-floor" plateau
we observed in iters 1-4 (stock PhotonFlow, gap 0.084 no matter what).

Fix used in the sweep: `monarch_init='random'` uses Xavier gain 0.1 —
off-identity, breaks the basin, closes 0.010.

### Saxe, McClelland, Ganguli 2013 — orthogonal init for deep linear networks ([arXiv:1312.6120](https://arxiv.org/abs/1312.6120))

**Relevance**: proves that if weight matrices are initialised as random
orthogonal matrices, training time of deep linear networks is independent of
depth.  Natural next init to try for `MonarchLayer.Ls`, `Rs`: QR of a Gaussian
matrix for each m×m block.  Expected gain on top of random Xavier: tighter
variance, faster convergence for the 12-block variant.

Not tried in the 9-run sweep (budget), but the single obvious improvement for
a follow-up.

### Wang et al. 2023 "Monarch Mixer" ([arXiv:2310.12109](https://arxiv.org/abs/2310.12109))

**Relevance**: replaces attention with **Monarch matrices initialised to
DFT/IDFT** — the untrained block is already a structured Fourier operator.
Their M2-ViT at 33 M params matches 86 M ViT-B on ImageNet.  Direct analogue
to our finding that off-identity init closes the gap, but using a
*principled* init (DFT) instead of random.  Would pair especially well with
our photonic hardware because optical Fourier transforms are implemented
*exactly* as butterfly MZI meshes — DFT init for Monarch IS the photonic
identity transform.

Next implementation: add `monarch_init='dft'` to `MonarchLayer._init_weights`.

### Phase D — architecture experiments (2K-step budget, goal: gap ≤ 0.05 vs baseline)
After iters 1-6 plateaued at eval ≈ 0.27, every TRAINING-side lever was exhausted.  The next step is to test two architectural knobs that had never been touched:

**H1: SaturableAbsorber cap**.  Shen et al. 2017 introduce the saturable-absorber nonlinearity `σ(x) = tanh(αx)/α` as a differentiable proxy for graphene-waveguide bleaching.  PhotonFlow picks `α = 0.8` (hard cap ±1.25).  The 2017 paper does NOT recommend a specific α; it's our arbitrary choice.  Lowering to `α = 0.2` raises the cap to ±5.0, covering the CFM target range.

**H2: mean-centering in DivisivePowerNorm**.  The original LayerNorm paper (Ba, Kiros, Hinton, arXiv 2016.07450) subtracts the per-sample mean before normalising.  We dropped mean subtraction for photonic-hardware compatibility (photodetector measures L2 power, not mean).  But the DC-drift accumulated across blocks is likely a significant source of our plateau.  Enabling mean-centering (via the new `DivisivePowerNorm(..., mean_center=True)` kwarg, backward-compatible default `False`) tests the magnitude of this effect.  If helpful, a hybrid scheme (electronic DC subtraction at the optical/electronic boundary — same physical location as the existing per-channel gain/bias) is still compatible with the co-design goals.

**Reported metric**: uniform-t pure-MSE CFMLoss, evaluated every 500 steps on MNIST train split (same 8-batch protocol as iters 1-6).  `[SUMMARY] best_eval_loss_uniform_t` at step 2000 is the comparison number.

### Iteration 6 — stack time-weighted loss + higher lr + larger batch (plateau buster)
**Context**: Iters 1-5 all plateau at eval ≈ 0.27.  Every single-lever training-side fix (logit-normal, direction loss, depth decay off) produced marginal improvements but never broke the 0.27 floor.  Plateau is consistent across:
- time sampling strategy
- model depth (4 or 8)
- Monarch factor count (1 or 2)
- with/without direction loss
- with/without depth decay

This strongly suggests the optimiser is stuck in a flat basin that small LR + small batch can't escape.  Three complementary levers are known to help escape plateau basins in DiT/SiT training:

**Lever A — time-weighted loss** (paper: Karras & Song style, arXiv 2511.16599, 2025):
`w(t) = max(1, γ / (1 − t + ε))` with γ = 5.  Upweights hard timesteps (t near 1) where the vector field has highest curvature, forcing the optimiser to commit capacity there rather than coasting on easy low-t samples.

**Lever B — higher peak LR** (5e-4 vs 3e-4): escaping the basin requires larger effective steps; combined with a longer 500-step warmup for stability.

**Lever C — larger batch** (256 vs 128): structured matrices (Monarch) benefit from lower gradient variance because the optimisation landscape has more ridges/valleys than unstructured dense layers.  Doubling batch reduces variance by √2.

**Config changes from iter 5**:
- `loss_weight_gamma` = 0 → 5.0
- `lr` = 3e-4 → 5e-4
- `warmup_steps` = 200 → 500
- `batch_size` = 128 → 256

**Loss-scale note**: time-weighted loss changes training-loss scale.  Pure uniform-t eval (unchanged) remains our comparable metric.

### Iteration 5 — remove depth-decay residual (let late blocks contribute fully)
**Context**: Iters 1-4 all plateau at eval ≈ 0.27.  The plateau is insensitive to:
- time sampling (uniform vs logit-normal)
- model depth (4 vs 8 blocks)
- Monarch factor count (1 vs 2)
- direction loss (with vs without)

This suggests the plateau is from a structural scaling limitation, not optimization or objective.  `depth_decay_residual=True` scales block i's output by `1 - i/N`; for 4 blocks that caps block 4's contribution at 25 % and block 3 at 50 %.  This is a PhotonFlow-specific addition inspired by Wang et al. ICLR 2025, but aggressive decay on a 4-block model may bottleneck fine-grained refinement.

**Lever**: `depth_decay_residual=False` (one change).  Single-lever removal of non-canonical scaling.

### Iteration 4 — add velocity direction loss (FasterDiT)
**Context**: Iter 3 (smaller model) plateaued at eval ≈ 0.27 — SAME level as iter 2, just reached faster.  The plateau is NOT about model capacity; it's an optimization-landscape / supervision issue.

**Lever (paper-backed)**: Yao et al., "FasterDiT: Towards Faster Diffusion Transformers Training without Architecture Modification," NeurIPS 2024, [arXiv:2410.10356].

Equation 8 in the paper:
```
L_dir = 1 - cos_sim(v_pred, v_target)
L_total = MSE(v_pred, v_target) + λ * L_dir
```
They report **7× faster training** to match DiT's FID at ImageNet-256 (FID=2.30 in 1000K iter vs DiT's 2.27 in 7000K).  FID-50k at 400K iter: 11.9 (FasterDiT) vs 19.5 (DiT) — absolute improvement from direction supervision alone.

**Config change**: `direction_loss_weight = 0.5` (paper-recommended).  Implementation already present in `photonflow/train.py` CFMLoss.

**Loss-scale note**: Training loss is now `MSE + 0.5 * (1 − cos_sim)` which is NOT directly comparable to baseline's 0.08.  We continue to log a separate **uniform-t pure-MSE eval loss every 500 steps** that IS directly comparable.  All "target ≤ 0.10" comparisons use the eval number.

### Iteration 3 — structural parity with baseline + keep logit-normal
**Context**: Iter 2 plateaued: step 500 eval 0.57 → step 2000 eval 0.28 (drop of ~0.005 per 500 steps after 1500).  Projected final at 10K: ~0.13-0.15, missing target.

**Levers changed (paper-backed)**:
- `num_blocks`: 8 → **4** (structural parity with baseline, which has 4 transformer blocks at hidden_dim=256).  Dao 2022 §5 shows Monarch models achieve baseline quality with HALF the depth of dense models (Monarch-ViT-B: 33M params, ViT-B: 86M).  Our 8 blocks × 2 sub-layers × 2 Monarch factors = 32 effective Monarch layers was over-parameterised for 10K steps.
- `num_monarch_factors`: 2 → **1** (canonical Dao 2022 MM* class = pair of single Monarch layers per sub-layer).  The stacked (MM*)² construction was a PhotonFlow-specific experimental extension; reverting to canonical for sample-efficiency.
- Everything else (logit-normal, lr, warmup, seed, batch, optimizer) unchanged from iter 2.

**Expected result**: smaller model (~4M params vs 14M) with faster per-step convergence.  Dao 2022 §5.1 shows Monarch-ViT-S at 19.6M params beats ViT-S at 48.8M in ImageNet accuracy — evidence that shallower + simpler Monarch architectures generalise at fewer steps.

### Iteration 2 — add logit-normal timestep sampling
**Lever**: `CFMLoss(time_sampling="logit_normal", logit_normal_mean=0.0, logit_normal_std=1.0)`

**Reason (paper-backed)**: Esser et al., "Scaling Rectified Flow Transformers
for High-Resolution Image Synthesis" (SD3), ICML 2024, [arXiv:2403.03206].
Section 3.2: For rectified flow / CFM training, sampling `t = sigmoid(N(0, 1))`
instead of `t ~ U[0, 1]` biases training toward the intermediate `t ≈ 0.5`
region where velocity prediction is hardest.  Reported as the single most
effective improvement to rectified flow training in SD3.

**Loss-scale comparability**: Uniform vs logit-normal sampling is a
reweighting of the EXPECTED training loss, but the per-sample MSE remains
the same objective (v_θ vs x1−x0).  For fair comparison with baseline we
also log a separate **uniform-t eval loss every 500 steps** alongside the
training loss — that eval number is directly comparable to baseline's 0.08.

**Non-changes (kept identical to iter 1)**: data pipeline, seed=42, batch=128,
Adam, lr=3e-4, 200-step warmup → cosine decay, model architecture.

## Still deliberately OFF (would change loss-scale or be arch-specific)

- Direction loss (FasterDiT, Yao et al. 2024) — adds a cosine term, changes loss scale
- Time-dependent weighting w(t) = γ/(1−t) (arXiv 2511.16599, 2025) — changes loss scale
- EMA of weights — effective averaging window > 10K steps, no benefit here
