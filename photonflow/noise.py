"""
photonflow/noise.py

Photonic noise injection: PhotonicNoise.

Two noise sources are modeled, injected after each Monarch layer during
training only. At evaluation time the module is a no-op (pass-through).

────────────────────────────────────────────────────────────────────────────
1. SHOT NOISE  (sigma_s = 0.02)
────────────────────────────────────────────────────────────────────────────
Physics:
    Light arrives at a photodetector as discrete photons. Even for a
    perfectly stable laser, photon arrivals are a Poisson process. At
    typical operating power levels the Poisson distribution is well
    approximated by a Gaussian with std proportional to sqrt(photon count).

    SNR = sqrt(n_photons)  →  noise_std proportional to sqrt(signal power)
                            proportional to |signal_amplitude|

    The physically accurate model (Shen et al. 2017, Methods section) is
    SIGNAL-DEPENDENT: each output element v_i gets noise drawn from
    N(0, (sigma_D * |v_i|)^2), i.e. noise scales with the signal magnitude.

    PhotonFlow uses a SIMPLIFIED additive model (signal-independent):

        noise_shot[i] ~ N(0, sigma_s^2)   independently for every i

    This is a first-order approximation. It is common in photonic-NN
    literature (Ning 2024: "shot noise modeled using a Gaussian distribution")
    and is computationally convenient, but strictly the noise floor should
    scale with signal strength. Use shot_signal_dependent=True for the
    physically accurate variant.

    sigma_s = 0.02: PhotonFlow design choice. Shen 2017 reports sigma_D ~ 0.1%
    (=0.001) for their experimental setup — 20x smaller. The larger value here
    is intentional: it acts as stronger noise regularization during training,
    following the rationale of Ning et al. 2025 (StrC-ONN) that more aggressive
    noise injection during training improves hardware robustness.

────────────────────────────────────────────────────────────────────────────
2. THERMAL CROSSTALK  (sigma_t = 0.01)
────────────────────────────────────────────────────────────────────────────
Physics:
    Thermo-optic phase shifters work by locally heating a waveguide.
    When one heater fires, it slightly warms its neighbours, shifting
    their phases. This is *correlated* noise: adjacent phase shifters
    experience similar perturbations.

    IMPORTANT physical distinction:
    In a real MZI chip (Shen 2017, Methods), thermal crosstalk corrupts the
    PHASE ANGLES of individual interferometers — perturbing the weight matrix
    itself, not the output features. Shen 2017 models this as:
        delta_theta_i, delta_phi_i ~ N(0, sigma_Phi^2)  applied to phases.

    PhotonFlow applies noise to the OUTPUT TENSOR after the Monarch transform.
    This is a proxy approximation: it is tractable and captures the correlated
    spatial structure of thermal diffusion, but does not faithfully represent
    phase-level perturbation propagated through the matrix multiply.

    The correlation model (heat diffusion → nearest-neighbour coupling):

        raw[i]        ~ N(0, 1)   iid
        corr[i]  = 0.5  * raw[i]
                 + 0.25 * raw[i-1]   (left neighbour)
                 + 0.25 * raw[i+1]   (right neighbour)
        noise_thermal[i] = sigma_t * corr[i]

    Implementation: a 1D convolution with kernel [0.25, 0.50, 0.25] across the
    last dimension (the feature/channel dimension), with CIRCULAR boundary
    padding (heat diffusion wraps at chip edge — preferable to zero-padding
    which artificially reduces noise at the boundary features).

────────────────────────────────────────────────────────────────────────────
References:
    Shen et al. 2017 (Nature Photonics) — identifies shot noise (photodetection
        noise sigma_D) and thermal crosstalk (phase-encoding noise sigma_Phi) as
        the two dominant noise sources in silicon photonic neural networks.
        Methods section gives the exact signal-dependent shot noise model and
        the per-phase Gaussian crosstalk model. sigma_D ~ 0.1% experimentally.

    Ning et al. 2024 (J. Lightwave Technol.) — Section V.C confirms that
        "shot noise and thermal noise are modeled using a Gaussian distribution
        by measuring on-chip photonic multiplication results." Validates the
        Gaussian noise approximation approach in general.

    Ning et al. 2025 (StrC-ONN) — hardware-aware training with noise injection
        ("dynamic noise injection" via DPE) recovers accuracy lost to on-chip
        non-idealities. PhotonFlow follows the same training principle.
        sigma_s=0.02, sigma_t=0.01 are PhotonFlow design choices calibrated to
        be within the range of experimentally reported photonic noise levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotonicNoise(nn.Module):
    """Photonic noise injection layer.

    Injects shot noise and thermal crosstalk after each Monarch layer
    during *training only*. In eval mode the input is returned unchanged.

    Placement in the block (per CLAUDE.md spec):
        Monarch L → [PhotonicNoise] → Monarch R → [PhotonicNoise]
        → SaturableAbsorber → DivisivePowerNorm → + TimeEmbed

    Physical accuracy notes:
        - Shot noise: Shen 2017 specifies signal-DEPENDENT noise (sigma ~ |v|).
          The default simplified mode uses signal-INDEPENDENT noise (fixed sigma).
          Set shot_signal_dependent=True to use the physically accurate variant.
        - Thermal noise: applied to output features as a proxy. In hardware it
          actually perturbs MZI phase angles (corrupting the weight matrix itself),
          but output-space injection is a tractable first-order approximation.
        - Circular boundary padding is used for the thermal conv kernel — this
          models heat wrapping at chip edges more accurately than zero-padding.

    Args:
        sigma_s (float): Shot noise standard deviation. Default 0.02.
                         Simplified model: additive iid Gaussian on every element.
                         Physically: Shen 2017 reports sigma_D ~ 0.1% (=0.001)
                         experimentally; 0.02 is intentionally larger for stronger
                         noise-regularization during training.
        sigma_t (float): Thermal crosstalk standard deviation. Default 0.01.
                         Correlated Gaussian via nearest-neighbour conv kernel
                         [0.25, 0.50, 0.25] across the feature dimension.
        enabled (bool):  Master switch. Set False to ablate noise entirely
                         (used for Experiment 2 vs Experiment 3 comparison).
        shot_signal_dependent (bool): If True, use the physically accurate
                         signal-dependent shot noise model from Shen 2017:
                         noise_i ~ N(0, (sigma_s * |x_i|)^2).
                         If False (default), use the simplified additive model:
                         noise_i ~ N(0, sigma_s^2), independent of signal.
        phase_noise_sigma (float): M8b honesty mitigation.  Shen 2017 reports
                         a separate sigma_phi ~= 5e-3 rad phase-encoding noise
                         that lives on the MZI phase angles, not on the output
                         tensor.  We model its effect on the Monarch output as
                         a rank-1 multiplicative Gaussian jitter
                         `x *= 1 + phase_noise_sigma * randn_like(x)`.  This is
                         a first-order proxy only -- the true effect couples
                         through the matrix multiply -- but it at least
                         exercises a MULTIPLICATIVE noise channel during
                         training.  Default 0.0 (OFF; preserves legacy output).
        cumulative_loss_db_per_stage (float): M12 honesty mitigation.  Every
                         photonic stage (MZI column, saturable absorber, norm)
                         attenuates the signal by the factor
                         `10^(-db_per_stage * (stage_index+1) / 20)`.  Shen 2017
                         Methods reports single-MZI transmission ~= 0.9993, i.e.
                         ~= 0.0003 dB per stage, NOT the 0.1 dB upper-bound the
                         original mismatches.md cited.  Applied as a scalar
                         multiplier after the noise injection.  Default 0.0
                         (OFF).
        stage_index (int): Zero-based index of this noise module in the forward
                         depth order; cumulative loss grows with it.  Default 0.
    """

    def __init__(
        self,
        sigma_s: float = 0.02,
        sigma_t: float = 0.01,
        enabled: bool = True,
        shot_signal_dependent: bool = False,
        phase_noise_sigma: float = 0.0,
        cumulative_loss_db_per_stage: float = 0.0,
        stage_index: int = 0,
    ) -> None:
        super().__init__()
        if sigma_s < 0:
            raise ValueError(f"sigma_s must be >= 0, got {sigma_s}")
        if sigma_t < 0:
            raise ValueError(f"sigma_t must be >= 0, got {sigma_t}")
        if phase_noise_sigma < 0:
            raise ValueError(
                f"phase_noise_sigma must be >= 0, got {phase_noise_sigma}"
            )
        if cumulative_loss_db_per_stage < 0:
            raise ValueError(
                "cumulative_loss_db_per_stage must be >= 0, "
                f"got {cumulative_loss_db_per_stage}"
            )
        if stage_index < 0:
            raise ValueError(f"stage_index must be >= 0, got {stage_index}")
        self.sigma_s = sigma_s
        self.sigma_t = sigma_t
        self.enabled = enabled
        self.shot_signal_dependent = shot_signal_dependent
        self.phase_noise_sigma = float(phase_noise_sigma)
        self.cumulative_loss_db_per_stage = float(cumulative_loss_db_per_stage)
        self.stage_index = int(stage_index)
        # Pre-compute the cumulative transmission factor once; it is a fixed
        # scalar (no gradients, not learnable).  At 0 dB/stage this is 1.0
        # exactly, so the forward pass is a true NO-OP when the kwarg is off.
        if self.cumulative_loss_db_per_stage > 0.0:
            self._cumulative_atten = 10.0 ** (
                -self.cumulative_loss_db_per_stage * (self.stage_index + 1) / 20.0
            )
        else:
            self._cumulative_atten = 1.0

        # Nearest-neighbour thermal crosstalk kernel: [left, self, right]
        # shape (out_channels=1, in_channels=1, kernel_size=3) for F.conv1d
        kernel = torch.tensor([[[0.25, 0.50, 0.25]]])
        # Register as a buffer so it moves to the right device with .to(device)
        # and is saved/loaded with state_dict, but is NOT a learnable parameter.
        self.register_buffer("_thermal_kernel", kernel)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def set_noise_scale(self, scale: float) -> None:
        """Set noise scaling factor (0.0 = no noise, 1.0 = full noise).

        Used for noise warmup: gradually increase noise from 0 to full
        sigma over the first N training steps. This lets the model learn
        basic structure before hardening against hardware noise.

        Args:
            scale: float in [0, 1]. Multiplied with sigma_s and sigma_t.
        """
        self._noise_scale = max(0.0, min(1.0, scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject photonic noise (training only).

        Args:
            x: Input tensor. Any shape is accepted. Noise is applied
               along the last dimension (feature dimension).

        Returns:
            x + shot_noise + thermal_noise  during training.
            x                               during evaluation (no-op).
        """
        # Eval mode or disabled: pass through unchanged.
        if not self.training or not self.enabled:
            return x

        # Noise warmup scaling (default 1.0 = full noise)
        scale = getattr(self, '_noise_scale', 1.0)
        if scale == 0.0 and self.phase_noise_sigma == 0.0 and self._cumulative_atten == 1.0:
            # Absolutely nothing to do — true no-op.
            return x

        # ── Phase-space noise (M8b honesty proxy) ──────────────────
        # Applied BEFORE the additive shot/thermal term so it sees the clean
        # Monarch output.  Rank-1 multiplicative jitter on the output tensor
        # stands in for the actual sigma_phi perturbation of the MZI angles.
        if self.phase_noise_sigma > 0.0:
            jitter = 1.0 + self.phase_noise_sigma * torch.randn_like(x)
            x = x * jitter

        # ── Shot noise ──────────────────────────────────────────────
        if self.shot_signal_dependent:
            # Physically accurate model (Shen 2017, Methods):
            #   noise_i ~ N(0, (sigma_s * |x_i|)^2)
            # Noise standard deviation scales with signal amplitude,
            # consistent with shot-noise-limited detection: SNR = sqrt(n).
            shot = self.sigma_s * x.abs() * torch.randn_like(x)
        else:
            # Simplified additive model (CLAUDE.md spec, default):
            #   noise_i ~ N(0, sigma_s^2)  iid, signal-independent.
            shot = self.sigma_s * torch.randn_like(x)

        # ── Thermal crosstalk ────────────────────────────────────────
        # Step 1: draw iid base noise.
        raw = torch.randn_like(x)                   # (..., dim)

        # Step 2: apply nearest-neighbour kernel via 1D convolution.
        # F.conv1d expects shape (N, C, L).
        # We flatten all leading dimensions into the batch axis.
        orig_shape = raw.shape
        dim = orig_shape[-1]
        flat = raw.reshape(-1, 1, dim)              # (batch_flat, 1, dim)

        # Circular padding: models heat wrapping at chip feature boundaries.
        # More physically accurate than zero-padding, which would artificially
        # reduce thermal noise at the first and last feature elements.
        flat_padded = F.pad(flat, (1, 1), mode='circular')  # (batch_flat, 1, dim+2)
        corr = F.conv1d(
            flat_padded,
            self._thermal_kernel,
            padding=0,   # no extra padding — we already padded manually
        )                                            # (batch_flat, 1, dim)

        thermal = self.sigma_t * corr.reshape(orig_shape)

        out = x + scale * (shot + thermal)
        # ── Cumulative optical loss (M12) ──────────────────────────
        # Multiplicative attenuation that grows with stage depth.  No-op at
        # `cumulative_loss_db_per_stage == 0.0` because `_cumulative_atten == 1.0`.
        if self._cumulative_atten != 1.0:
            out = out * self._cumulative_atten
        return out

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        extras = [
            f"sigma_s={self.sigma_s}",
            f"sigma_t={self.sigma_t}",
            f"enabled={self.enabled}",
            f"shot_signal_dependent={self.shot_signal_dependent}",
        ]
        if self.phase_noise_sigma > 0.0:
            extras.append(f"phase_noise_sigma={self.phase_noise_sigma}")
        if self.cumulative_loss_db_per_stage > 0.0:
            extras.append(
                f"cumulative_loss_db_per_stage={self.cumulative_loss_db_per_stage}"
            )
            extras.append(f"stage_index={self.stage_index}")
        return ", ".join(extras)


# ---------------------------------------------------------------------------
# Self-contained tests — run with: python photonflow/noise.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    noise_fn = PhotonicNoise(sigma_s=0.02, sigma_t=0.01)

    print("Testing PhotonicNoise...")
    print()

    # --- Test 1: eval mode is a clean pass-through ---
    noise_fn.eval()
    x = torch.randn(8, 64)
    out = noise_fn(x)
    assert torch.equal(out, x), "Eval mode changed the input — must be a no-op"
    print("  [PASS] Test 1 - eval mode pass-through: output identical to input")

    # --- Test 2: train mode adds noise (output != input) ---
    noise_fn.train()
    x2 = torch.randn(8, 64)
    out2 = noise_fn(x2)
    assert not torch.equal(out2, x2), "Train mode did not add any noise"
    max_diff = (out2 - x2).abs().max().item()
    print(f"  [PASS] Test 2 - train mode adds noise: max|noise| = {max_diff:.4f}")

    # --- Test 3: shot noise std matches sigma_s (simplified additive mode) ---
    # Isolate shot noise by disabling thermal. Use zero input so signal-dependent
    # mode would also give zero noise (verifying additive mode specifically).
    shot_only = PhotonicNoise(sigma_s=0.02, sigma_t=0.0)
    shot_only.train()
    x3 = torch.zeros(10000, 64)   # large tensor for good statistics
    noise_samples = shot_only(x3) - x3
    empirical_std = noise_samples.std().item()
    assert abs(empirical_std - 0.02) < 0.002, (
        f"Shot noise std = {empirical_std:.4f}, expected approx 0.02"
    )
    print(f"  [PASS] Test 3 - shot noise std (additive): empirical={empirical_std:.4f}, target=0.0200")

    # --- Test 3b: signal-dependent shot noise (Shen 2017 physical model) ---
    # With shot_signal_dependent=True and constant input amplitude A,
    # the noise std should be sigma_s * A.
    shot_dep = PhotonicNoise(sigma_s=0.02, sigma_t=0.0, shot_signal_dependent=True)
    shot_dep.train()
    A = 2.0
    x3b = torch.full((10000, 64), A)
    noise_dep = shot_dep(x3b) - x3b
    empirical_std_dep = noise_dep.std().item()
    expected_dep = 0.02 * A   # sigma_s * |x|
    assert abs(empirical_std_dep - expected_dep) < 0.005, (
        f"Signal-dependent shot noise std = {empirical_std_dep:.4f}, expected approx {expected_dep:.4f}"
    )
    print(f"  [PASS] Test 3b - shot noise std (signal-dep, Shen 2017): empirical={empirical_std_dep:.4f}, "
          f"expected={expected_dep:.4f} (sigma_s * |x| = 0.02 * {A})")

    # --- Test 4: thermal crosstalk std matches expected value ---
    thermal_only = PhotonicNoise(sigma_s=0.0, sigma_t=0.01)
    thermal_only.train()
    x4 = torch.zeros(10000, 64)
    thermal_noise = thermal_only(x4) - x4
    thermal_std = thermal_noise.std().item()
    # With circular padding the interior and boundary elements are equivalent.
    # The convolution kernel [0.25, 0.50, 0.25] has L2 norm = sqrt(0.375) approx 0.612.
    # Effective std approx 0.01 * 0.612 approx 0.006. Allow +/-0.002 tolerance.
    assert abs(thermal_std - 0.006) < 0.003, (
        f"Thermal noise std = {thermal_std:.4f}, expected approx 0.006"
    )
    print(f"  [PASS] Test 4 - thermal noise std: empirical={thermal_std:.4f}, expected~0.006")

    # --- Test 5: thermal noise is correlated (adjacent elements co-vary) ---
    # If corr(noise[i], noise[i+1]) > 0, neighbours share noise (as intended).
    # Theory: with kernel [0.25, 0.50, 0.25] on iid raw noise,
    #   Cov(corr[i], corr[i+1]) = (0.25*0.5 + 0.5*0.25) = 0.25
    #   Var(corr[i]) = 0.25^2 + 0.5^2 + 0.25^2 = 0.375
    #   Expected correlation = 0.25 / 0.375 = 0.667
    x5 = torch.zeros(50000, 32)
    tn = thermal_only(x5) - x5          # shape (50000, 32)
    col_i   = tn[:, 15] - tn[:, 15].mean()
    col_ip1 = tn[:, 16] - tn[:, 16].mean()
    corr = (col_i * col_ip1).mean() / (col_i.std() * col_ip1.std() + 1e-8)
    assert corr.item() > 0.2, (
        f"Thermal noise not correlated enough: corr={corr.item():.4f} (expected > 0.2)"
    )
    print(f"  [PASS] Test 5 - thermal correlation: corr(noise[i], noise[i+1]) = {corr.item():.4f} "
          f"(theory: 0.667)")

    # --- Test 5b: circular padding — boundary elements also get neighbour noise ---
    # With zero-padding, noise[0] would only see noise[1] on the right (no left).
    # With circular padding, noise[0] also sees noise[-1] from the right boundary.
    # Check that corr(noise[0], noise[-1]) > 0 (circular wrap-around).
    x5b = torch.zeros(50000, 32)
    tn5b = thermal_only(x5b) - x5b
    col_first = tn5b[:, 0] - tn5b[:, 0].mean()
    col_last  = tn5b[:, -1] - tn5b[:, -1].mean()
    corr_wrap = (col_first * col_last).mean() / (col_first.std() * col_last.std() + 1e-8)
    assert corr_wrap.item() > 0.1, (
        f"Circular padding: noise[0] and noise[-1] should be correlated, "
        f"got corr={corr_wrap.item():.4f}"
    )
    print(f"  [PASS] Test 5b - circular boundary: corr(noise[0], noise[-1]) = {corr_wrap.item():.4f} "
          f"(confirms circular padding, not zero-padding)")

    # --- Test 6: enabled=False always passes through (even in train mode) ---
    disabled = PhotonicNoise(sigma_s=0.02, sigma_t=0.01, enabled=False)
    disabled.train()
    x6 = torch.randn(4, 32)
    out6 = disabled(x6)
    assert torch.equal(out6, x6), "enabled=False in train mode still added noise"
    print("  [PASS] Test 6 - enabled=False: always pass-through")

    # --- Test 7: shape preserved for arbitrary input shapes ---
    shapes = [(4,), (4, 64), (4, 16, 64), (2, 4, 8, 64)]
    noise_fn.train()
    for shape in shapes:
        x7 = torch.randn(*shape)
        out7 = noise_fn(x7)
        assert out7.shape == x7.shape, f"Shape mismatch for input {shape}"
    print(f"  [PASS] Test 7 - arbitrary shapes: tested {[str(s) for s in shapes]}")

    print()
    print("All tests passed.")
    print(f"  {noise_fn}")
    print()
    print("Physical accuracy summary:")
    print("  Shot noise (default):    additive iid Gaussian (simplified, matches spec)")
    print("  Shot noise (Shen 2017):  signal-dependent N(0, (sigma_s*|x|)^2)")
    print("  Thermal crosstalk:       correlated via kernel [0.25,0.50,0.25], circular boundary")
    print("  Injection point:         output features (proxy for true phase-level perturbation)")
