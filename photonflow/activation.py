"""
photonflow/activation.py

Photonic activation function: SaturableAbsorber.

Physical mapping:
    A graphene waveguide insert acts as a saturable absorber — low-intensity
    light is absorbed, high-intensity light passes through (bleaching).

    Shen et al. 2017 (Eq. 1) gives the full Selden physical model:

        sigma * tau_s * I0 = (1/2) * ln(Tm / T0) / (1 - Tm)
        Iout = I0 * Tm(I0)

    where sigma is the absorption cross-section, tau_s the radiative lifetime,
    T0 the initial transmittance, I0 the incident intensity, and Tm the
    transmittance solved implicitly from the equation above. This model only
    acts on positive optical intensities (Iin >= 0).

    PhotonFlow uses the differentiable approximation:

        sigma(x) = tanh(alpha * x) / alpha

    This captures the essential properties of the bleaching curve (zero at
    origin, monotonic, saturating) while accepting signed field amplitudes
    as used in coherent neural networks. It is a standard simplification in
    the photonic-NN literature and is NOT the literal Selden equation.

    NOTE: alpha = 0.8 is PhotonFlow's design choice — it is not measured
    from hardware or stated in any cited paper. It sets the slope near the
    origin (unit gain at x=0 for any alpha) and the saturation level
    (|output| < 1/alpha = 1.25).

    NOTE: Shen et al. 2017 simulated the saturable absorber nonlinearity on
    a conventional computer between OIU stages; they did not integrate a
    physical graphene SA on-chip. On-chip graphene saturable absorbers have
    been demonstrated in principle (refs. 38 in Shen 2017) but are not yet
    standard in deployed photonic-NN chips.

Reference:
    Shen et al., "Deep Learning with Coherent Nanophotonic Circuits,"
    Nature Photonics, 2017.
    - Identifies saturable absorption as a native photonic nonlinearity
      primitive (Section: ONN device architecture, ONU discussion).
    - Gives the full Selden physical model (Eq. 1 and Supplementary Sec. 2).
    - tanh(alpha*x)/alpha is PhotonFlow's differentiable proxy for this model.
"""

import torch
import torch.nn as nn


class SaturableAbsorber(nn.Module):
    """Saturable-absorber nonlinearity: sigma(x) = tanh(alpha * x) / alpha.

    Maps to a graphene waveguide insert in a silicon photonic chip (proposed;
    see module docstring — Shen 2017 simulated this electronically).

    alpha controls the slope near the origin (how quickly the output
    saturates). At alpha=0.8 the function is close to the identity for
    small x and saturates to ±1.25 for large |x|.

    Properties:
        f(0) = 0          (zero input → zero output, photonically correct)
        f'(0) = 1.0       (unit gain at origin for any alpha)
        |f(x)| < 1/alpha  (bounded output — never exceeds ±1/alpha)

    Note on physical accuracy:
        The true physical model (Shen 2017, Eq. 1) is the Selden equation,
        which acts on positive intensities only. tanh(alpha*x)/alpha is a
        signed, differentiable approximation — consistent with coherent
        field-amplitude representations but not the literal hardware equation.
        alpha=0.8 is PhotonFlow's design choice with no direct hardware basis.

    Args:
        alpha (float): Saturation slope parameter. Default 0.8.
                       When `learnable_alpha=True`, this is the INITIAL value of
                       a per-module learnable parameter; the optimizer can move
                       it to represent, say, a thermally-tuned absorber.
        learnable_alpha (bool): If True, alpha becomes a learnable parameter
                       (one scalar per SaturableAbsorber instance). Default False
                       (matches Shen 2017 fixed-operating-regime assumption).
        leaky_slope (float): Coefficient of an optional linear pass-through:
                       output = tanh(a*x)/a + leaky_slope * x.  Default 0.0
                       (pure tanh, same as before).  A small positive leaky
                       term (e.g., 0.05) preserves a linear "bypass" past the
                       saturation cap, which is photonically realisable as a
                       parallel unabsorbed-light path; empirically helps escape
                       the ±1/alpha magnitude ceiling when the target range
                       exceeds it (e.g., CFM velocity targets at t≈0).
    """

    def __init__(
        self,
        alpha: float = 0.8,
        learnable_alpha: bool = False,
        leaky_slope: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.learnable_alpha = learnable_alpha
        self.leaky_slope = float(leaky_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply saturable-absorber transfer function.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape as x. With leaky_slope=0 values lie in
            (-1/alpha, +1/alpha); with leaky_slope>0 the output is unbounded but
            bounded-error vs the pure-tanh form.
        """
        # Clamp alpha > eps to avoid 1/0 during training.
        a = self.alpha if not self.learnable_alpha else self.alpha.clamp(min=1e-3)
        out = torch.tanh(a * x) / a
        if self.leaky_slope != 0.0:
            out = out + self.leaky_slope * x
        return out

    def extra_repr(self) -> str:
        a_val = self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha
        extras = [f"alpha={a_val:.3f}"]
        if self.learnable_alpha:
            extras.append("learnable_alpha=True")
        if self.leaky_slope != 0.0:
            extras.append(f"leaky_slope={self.leaky_slope}")
        return ", ".join(extras)


# ---------------------------------------------------------------------------
# Self-contained tests — run with: python photonflow/activation.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    sa = SaturableAbsorber(alpha=0.8)
    sa.eval()

    # --- Test 1: shape preserved ---
    x = torch.randn(4, 16, 32)
    out = sa(x)
    assert out.shape == x.shape, (
        f"Shape mismatch: expected {x.shape}, got {out.shape}"
    )
    print(f"  [PASS] Test 1 — shape preserved: {tuple(out.shape)}")

    # --- Test 2: f(0) = 0 ---
    zero_in = torch.zeros(8)
    zero_out = sa(zero_in)
    assert torch.allclose(zero_out, torch.zeros(8), atol=1e-6), (
        f"f(0) != 0: got {zero_out}"
    )
    print(f"  [PASS] Test 2 — f(0) = 0: max|f(0)| = {zero_out.abs().max().item():.2e}")

    # --- Test 3: output bounded in (-1/alpha, +1/alpha) ---
    bound = 1.0 / sa.alpha  # 1.25 for alpha=0.8
    x_large = torch.randn(1000) * 100  # huge inputs
    out_large = sa(x_large)
    # tanh saturates to exactly ±1 in float32 for very large inputs,
    # so the bound is non-strict: |f(x)| <= 1/alpha.
    assert (out_large.abs() <= bound + 1e-6).all(), (
        f"Output exceeded bound ±{bound:.4f}: max={out_large.abs().max().item():.4f}"
    )
    print(f"  [PASS] Test 3 — range bounded: all outputs in [-{bound:.4f}, +{bound:.4f}]")

    # --- Test 4: gradients flow ---
    x_grad = torch.randn(4, 8, requires_grad=True)
    out_grad = sa(x_grad)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradient is None — backward pass failed"
    assert not torch.isnan(x_grad.grad).any(), "NaN gradients detected"
    # Verify gradient at 0 is 1.0: d/dx[tanh(a*x)/a]|_{x=0} = a * sech^2(0) / a = 1
    x_zero = torch.zeros(1, requires_grad=True)
    sa(x_zero).backward()
    grad_at_zero = x_zero.grad.item()
    assert math.isclose(grad_at_zero, 1.0, abs_tol=1e-5), (
        f"Gradient at 0 should be 1.0, got {grad_at_zero}"
    )
    print(f"  [PASS] Test 4 — gradients flow: grad at x=0 = {grad_at_zero:.6f}")

    print()
    print("All tests passed.")
    print(f"  SaturableAbsorber(alpha=0.8): {sa}")
