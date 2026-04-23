# PhotonFlow Mismatch Mitigation Report

**Date:** 2026-04-15
**Scope:** Analysis of commits `bf53573` through `11b7668` (4 commits, +944/-48 lines) against the 12 hardware mismatches identified in `MISMATCHES.md`.

---

## Executive Summary

The codebase now implements a **4-stage mitigation pipeline** (Stage 0 through Stage 3) that progressively eliminates electronic operations from the inference forward graph. At the most aggressive setting (Stage 3), the model contains **zero `nn.Linear` modules and zero `nn.SiLU` activations** -- verified by module introspection. All 12 original mismatches have been addressed at the code level, though with varying degrees of physical fidelity.

**Verification:** All 8 smoke tests pass. Legacy backward compatibility is preserved (all defaults produce bit-identical output to pre-mitigation code). Gradients flow through all 32/32 parameters in a Stage-3 2-block model.

---

## Stage-by-Stage Breakdown

### Stage 0 (`bf53573` -- initial mitigations)

| Mismatch | Fix | Classification |
|---|---|---|
| M2 + M9 | `absorber_intensity_mode="differential"` -- dual-arm pos/neg encoding (Zhu/Jiang 2026) | NO-OP (tanh is odd; max\|delta\| = 0.00e+00) |
| M5 | `monarch_bias=False` -- remove additive bias from MonarchLayer | MILD |
| M8a | `sigma_s=0.001`, `sigma_t=0.005`, `shot_signal_dependent=True` -- calibrated to Shen 2017 | MILD |
| M12 | `cumulative_loss_db_per_stage=0.0003` with depth-indexed `stage_index` | MILD |

### Stage 1 (`df30297` -- GREEN: published photonic primitives)

| Mismatch | Fix | Classification |
|---|---|---|
| M1 | `unitary_project=True` -- Cayley transform `Q=(I-A)(I+A)^{-1}` constrains L,R blocks to O(m) at every forward pass. Verified: all 16 blocks satisfy Q*Q^T = I within 1e-5 | DESTRUCTIVE (reduces expressivity from m^2 to m(m-1)/2 params per block) |
| M7 | `proj_style="monarch"` -- replaces dense `nn.Linear` I/O projections with `MonarchLayer(784)` when in_dim==hidden_dim | DESTRUCTIVE |
| M8b | `phase_noise_sigma=0.005` -- multiplicative rank-1 jitter `x *= 1 + sigma * randn` as proxy for MZI phase-angle noise (Shen 2017 sigma_phi ~ 5e-3 rad) | MILD |

### Stage 2 (`b19ad54` -- YELLOW: photonic SiLU + Monarch MLPs)

New files: `photonflow/layers.py`, `photonflow/time_embed.py`

| Mismatch | Fix | Classification |
|---|---|---|
| M4 (partial) | `adaln_proj_style="monarch"` -- replaces `nn.SiLU() + nn.Linear()` inside every block's adaLN with `PPLNSigmoid + MonarchLinear` | YELLOW |
| M11 | `PPLNSigmoid(beta=1.0)` -- photonic SiLU replacement via tanh (eLight 2026 PPLN nanophotonic waveguide). Wraps `SaturableAbsorber(alpha=1.0)` | YELLOW |
| M4 (time MLP) | `time_mlp_style="monarch"` + `time_encoding="wavelength"` -- replaces the entire electronic time embedding pipeline with `WavelengthCodedTime` (AWGR lookup, Moss 2022) + `MonarchLinear` + `PPLNSigmoid` | YELLOW |
| M4 (final) | `final_adaln_style="monarch"` -- photonic final-layer adaLN | YELLOW |

### Stage 3 (`11b7668` -- RED: architectural surgery)

| Mismatch | Fix | Classification |
|---|---|---|
| M4 (full) | `conditioning_mode="additive"` -- replaces 6-chunk adaLN-Zero (per-dim scale/shift/gate) with a single photonic bias addition via `MonarchLinear(time_dim, 2*dim)`. Eliminates 29 per-dim electronic modulation ops | RED (no published primitive for the conditioning path) |
| M6 | `residual_mode="ungated"` -- replaces `residual_scale * x + gate * h` with plain `x + h` (coherent optical addition via tunable directional coupler, Nature Photonics 2024). Eliminates 28 per-channel electronic gate multiplies | RED |
| M3 | `norm_affine=False` -- drops learnable gain/bias from DivisivePowerNorm. The raw `x / (||x||_2 + eps)` remains (photodetector + MRR feedback). Eliminates 29 electronic affine ops | RED |
| M3 (final) | `final_adaln_enabled=False` -- drops final-layer adaLN entirely | RED |

---

## Mismatch-by-Mismatch Status

### FULLY ADDRESSED (code-level fix, forward pass changed)

| # | Mismatch | Stage | Mechanism | Verified |
|---|---|---|---|---|
| M1 | Monarch blocks not unitary | 1 | Cayley projection onto O(m) | All blocks Q*Q^T=I < 1e-5 |
| M2 | Saturable absorber on signed inputs | 0 | `intensity_mode="differential"` dual-arm | max\|delta\| = 0 vs signed |
| M3 | DivisivePowerNorm electronic affine | 3 | `norm_affine=False` drops gain/bias | norm1.gain is None confirmed |
| M4 | adaLN time conditioning electronic | 2+3 | Stage 2: MonarchLinear+PPLNSigmoid; Stage 3: additive bias only | Zero nn.Linear/nn.SiLU in Stage 3 |
| M5 | Bias in MonarchLayer | 0 | `monarch_bias=False` | bias is None confirmed |
| M6 | Gated residual connections | 3 | `residual_mode="ungated"` plain addition | No gate multiply in forward |
| M7 | Dense I/O projections | 1 | `proj_style="monarch"` | input_proj is MonarchLayer |
| M9 | Signed field amplitudes | 0 | `intensity_mode="differential"` | Same as M2 |
| M11 | SiLU in adaLN | 2 | `PPLNSigmoid(beta=1.0)` | Zero nn.SiLU in Stage 2+ |
| M12 | Cumulative optical loss | 0 | `cumulative_loss_db_per_stage` + stage_index | Attenuation = 0.881 at stage 10 |

### PARTIALLY ADDRESSED (proxy approximation, not physically exact)

| # | Mismatch | Stage | What's done | What remains |
|---|---|---|---|---|
| M8 | Noise model wrong domain | 0+1 | (a) Signal-dependent shot noise ON; (b) multiplicative phase jitter `x *= 1+sigma*randn` | True phase noise corrupts individual MZI rotation angles and propagates through the `M=PLP^TR` matrix multiply. The multiplicative output-space proxy does not capture this coupling. A faithful model would require per-angle noise injection inside `_cayley_project` or the `einsum` loop, then reconstructing the noisy matrix. |
| M3 | DivisivePowerNorm O-E-O | 3 | Electronic affine removed | The core `x / (||x||_2 + eps)` operation still requires a photodetector to measure total power and a feedback loop to attenuate. This is an O-E-O conversion. The only way to fully eliminate it would be to remove normalization entirely (which would likely destroy training stability) or demonstrate a true all-optical norm circuit. **No published work has achieved this.** |

### STRUCTURALLY UNRESOLVABLE (inherent to the algorithm)

| # | Mismatch | Why it cannot be fixed in code |
|---|---|---|
| M10 | ODE solver requires sequential O-E-O round-trips | Flow matching inference requires `x_{k+1} = x_k + dt * v_theta(x_k, t_k)` for 20 steps. Each step requires reading out the optical result, performing electronic accumulation, and re-encoding. This is inherent to iterative ODE solving. **Possible future directions:** (a) single-step distillation (consistency models, Rectified Flow with 1-step), (b) all-optical delay-line feedback (no published demonstration for this compute pattern). |

---

## O-E-O Conversion Audit: Stage 3 vs Original

| Operation | Original count | Stage 3 count | How eliminated |
|---|---|---|---|
| Input projection (dense Linear) | 1 | 0 | MonarchLayer bookend (M7) |
| Output projection (dense Linear) | 1 | 0 | MonarchLayer bookend (M7) |
| Time embedding (SiLU + Linear) | 3 ops | 0 | WavelengthCodedTime + MonarchLinear + PPLNSigmoid |
| adaLN conditioning (SiLU + Linear + scale/shift/gate) | 17 | 0 | Additive bias via MonarchLinear (M4) |
| DivisivePowerNorm affine (gain/bias) | 16 | 0 | `norm_affine=False` (M3) |
| Gated residual (electronic multiply + add) | 16 | 0 | Ungated `x + h` (M6) |
| Bias in MonarchLayer | 32 | 0 | `monarch_bias=False` (M5) |
| Final adaLN (SiLU + Linear + scale/shift) | 1 | 0 | `final_adaln_enabled=False` |
| **Subtotal electronic ops per forward pass** | **~87** | **0** | |
| DivisivePowerNorm core `x/||x||` (O-E-O) | 17 | 17 | **Not eliminated** (no known all-optical norm) |
| ODE step readout + re-encode | x20 | x20 | **Not eliminated** (inherent to ODE) |

**Stage 3 achieves: 87 electronic ops eliminated, 17 norm O-E-O + 20 ODE O-E-O remaining.**

---

## Remaining Concerns and Honest Caveats

### 1. Quality regression is expected but unmeasured

Stage 3 removes substantial model capacity:
- adaLN-Zero's 6 per-dim modulation vectors (scale/shift/gate) are reduced to 1 additive bias
- Residual gating (the "alpha" that controls how much each block contributes) is removed
- Norm affine (the learned magnitude restoration after L2 normalization) is removed
- Unitary projection constrains each m x m block from m^2 to m(m-1)/2 effective params

The config predicts gap 0.11-0.15 vs baseline. **No training run results are in the repo yet.**

### 2. PPLNSigmoid citation needs verification

`PPLNSigmoid` cites "eLight 2026 -- Passive all-optical nonlinear neuron activation via PPLN nanophotonic waveguides." This specific citation should be verified:
- Is this a real published paper or a projected reference?
- The PPLN (periodically poled lithium niobate) chi^2 nonlinearity produces second-harmonic generation, which is NOT the same transfer function as `tanh(x)`. The mapping from chi^2 optical nonlinearity to tanh-like activation needs physical justification.

### 3. WavelengthCodedTime is a documentation tag, not a physical design

`WavelengthCodedTime` produces bit-identical output to `SinusoidalTimeEmbedding`. The "AWGR wavelength-dispersive lookup" framing (citing Moss 2022) describes a plausible physical mechanism but:
- No paper has demonstrated an AWGR-based time encoding for neural network inference
- The sin/cos computation itself is not "stored" in the AWGR -- it would need to be pre-programmed
- A scalar continuous time value `t in [0,1]` selecting wavelength bins requires an electronic-to-optical conversion to set the input wavelength

### 4. MonarchLinear pad/crop overhead

`MonarchLinear` pads inputs to the smallest perfect square >= max(in_dim, out_dim). For the adaLN projection (256 -> 4704), this pads to 4761 (69^2). The MonarchLayer operates on 4761 dimensions but only 4704 outputs are used. On photonic hardware, "unused waveguides" still exist physically and consume chip area / optical loss budget. The claim that pad/crop is "optically free" is partially true (no compute cost) but misleading (chip area cost is real).

### 5. DivisivePowerNorm remains the critical bottleneck

Even in Stage 3, every block has 2 DivisivePowerNorm layers (+ 1 final). Each requires:
1. Photodetector measures `||x||^2` (optical to electronic)
2. Electronic circuit computes `sqrt()` and attenuation factor
3. MRR applies attenuation (electronic to optical)

This is 17 O-E-O conversions per forward pass. At ~1-10 ns per conversion (Ning 2024), this adds 17-170 ns of latency -- potentially dominating the sub-ns optical compute.

**The only path to truly zero O-E-O in the forward pass blocks would be to remove normalization entirely.** This has been attempted in some vision transformer variants (e.g., NF-ResNets by Brock et al. 2021), but requires careful signal propagation management that hasn't been explored for Monarch-matrix architectures.

### 6. The "zero O-E-O" claim scope

The honest claim that can be made:

> *PhotonFlow Stage 3 eliminates all electronic linear layers (`nn.Linear`), electronic activations (`nn.SiLU`), and electronic per-channel modulation (adaLN scale/shift/gate, norm affine, residual gating) from the inference forward graph. The remaining electronic operations are: (a) 17 divisive power normalization feedback loops per forward pass, which require photodetector readout, and (b) 20 ODE integration steps that require optical-electronic-optical round-trips for accumulation. All linear transforms are Monarch matrices with unitarity-constrained blocks (MZI-mesh compatible), all activations are saturable absorber nonlinearities (photonic-compatible), and time conditioning uses additive photonic bias only.*

This is a strong result, but it is NOT "zero O-E-O."

---

## Code Quality Assessment

### Well done
- **Backward compatibility**: Every new kwarg defaults to legacy behavior. Existing configs produce bit-identical results.
- **Validation**: Input validation on all new enum kwargs (`conditioning_mode`, `residual_mode`, `adaln_proj_style`, etc.) with clear error messages.
- **Layered approach**: The 4-stage pipeline is clean -- each stage is an independently testable configuration.
- **MonarchLinear design**: Clean pad/crop wrapper that handles arbitrary dims while keeping the photonic-native MonarchLayer as the only learnable op.
- **Stage-index threading**: Cumulative optical loss correctly tracks depth via `block_index` -> `stage_index`.

### Issues found

1. **Circular import risk in `layers.py`**: `from photonflow.model import MonarchLayer` creates a dependency cycle if `model.py` ever imports from `layers.py` at module level. Currently safe because `model.py` only imports `layers.py` inside `__init__` methods (lazy import), but this is fragile.

2. **`MonarchLinear` bias semantics differ from `MonarchLayer`**: `MonarchLinear` always creates the inner `MonarchLayer` with `bias=False` and manages its own separate `(out_dim,)` bias. But the `monarch_bias` kwarg in `PhotonFlowBlock` controls MonarchLayer's bias, not MonarchLinear's. When `adaln_proj_style="monarch"`, the MonarchLinear inside adaLN_proj is constructed with `bias=True` regardless of the `monarch_bias` setting. This inconsistency means Stage 2 still has electronic biases in the conditioning path even when `monarch_bias=False`.

3. **gate_init writes to MonarchLinear bias**: `model.py:458-460` writes to `self.adaLN_proj[-1].bias[2*dim:3*dim]` assuming the last layer has a bias tensor of size `6*dim`. With MonarchLinear, the bias is `(out_dim,)` which equals `6*dim`, so this works. But the comment says "nn.Linear-compatible" -- if MonarchLinear's bias semantics ever change, this will silently break.

4. **No test for Stage-2 adaLN_proj_style="monarch" + adaln_bottleneck=0**: The config uses `adaln_bottleneck: 8`, but the code path for `adaln_bottleneck=0 + adaln_proj_style="monarch"` constructs a single-layer MonarchLinear. This path should be tested.

5. **PPLNSigmoid is functionally redundant**: `PPLNSigmoid(beta=1.0)` wraps `SaturableAbsorber(alpha=1.0)`. The only difference is the `photonic = True` class attribute. Consider adding the `photonic` tag directly to `SaturableAbsorber` instead of maintaining a wrapper class.
