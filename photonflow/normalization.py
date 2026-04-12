"""
photonflow/normalization.py

Photonic normalization layer: DivisivePowerNorm.

Physical mapping (proposed — see accuracy note below):
    A photodetector measures the total optical power of the output beam.
    Optical power = sum(|Ei|^2) = ||E||_2^2, so the L2 norm ||E||_2 is
    sqrt(total power), measurable passively at the speed of light.
    A microring resonator in a feedback loop uses this measured norm to
    uniformly attenuate the signal, dividing every element by it.
    The result is a unit-norm output vector.

    Formula:  y = x / (||x||_2 + eps)

NOTE on physical accuracy:
    DivisivePowerNorm is PhotonFlow's OWN proposed all-optical replacement
    for LayerNorm. It does NOT appear in any cited paper. In practice:

    - Shen et al. 2017: nonlinearity was simulated on a computer between
      optical stages; no on-chip normalization was described.
    - Ning et al. 2025 (StrC-ONN / CirPTC): "batch normalization (BN),
      pooling, and nonlinear activation are executed on digital processors."
      Existing photonic-NN systems run ALL normalization electronically.

    The microring-resonator-as-normalizer is physically plausible (MRRs
    can attenuate based on feedback) but has not been experimentally
    demonstrated for this purpose in any referenced work.

Why not LayerNorm?
    LayerNorm:          (x - mean(x)) / sqrt(var(x) + eps)
                         ↑ mean AND variance require electronic circuits
                         ↑ two statistics computed over all features

    DivisivePowerNorm:  x / (||x||_2 + eps)
                         ↑ only L2 norm needed (= sqrt of total power)
                         ↑ photodetector measures total power natively

    Trade-off vs LayerNorm:
        LayerNorm centers the output (subtracts mean). DivisivePowerNorm
        does NOT subtract a mean — it only normalizes the magnitude.
        If Monarch layers produce large-mean outputs, this difference in
        centering may affect training stability and expressivity.

References:
    Ning et al., "Photonic-Electronic Integrated Circuits for High-Performance
    Computing and AI Accelerators," J. Lightwave Technol., 2024.
    (Surveys photonic hardware constraints and confirms O-E-O is the
    bottleneck. Confirms existing systems use electronic normalization.
    Does NOT describe DivisivePowerNorm — it is PhotonFlow's design.)

    Shen et al., "Deep Learning with Coherent Nanophotonic Circuits,"
    Nature Photonics, 2017.
    (Photodetectors measure optical power — the physical basis for the
    L2-norm measurement step. Does not describe the divisive normalization.)
"""

import torch
import torch.nn as nn


class DivisivePowerNorm(nn.Module):
    """Divisive power normalization: output = x / (||x||_2 + eps).

    PhotonFlow's proposed all-optical replacement for LayerNorm.
    Maps to a photodetector + microring resonator feedback loop on a
    silicon photonic chip (proposed; see module docstring for accuracy caveats).

    The photodetector measures total optical power P = sum(|Ei|^2) = ||x||_2^2.
    The microring resonator feedback attenuates all channels by 1 / (sqrt(P) + eps)
    — equivalent to dividing by (||x||_2 + eps).

    Normalizes over the last dimension (the feature/channel dimension).
    For input shape (batch, dim) this normalizes each sample's feature
    vector independently. For (batch, seq, dim) it normalizes each token.

    Properties:
        ||output||_2 ≈ 1.0  (unit-norm output, up to eps rounding)
        x = 0 → output = 0  (zero input is safe — eps prevents division by zero)
        Gradients flow cleanly (no discontinuities)
        Direction preserved: cosine_similarity(x, output) = 1.0

    Difference from LayerNorm:
        LayerNorm also subtracts the mean (centering). This layer does NOT.
        If the mean of x is large, that offset is preserved in the output
        direction. This is an accepted trade-off for photonic-hardware
        compatibility — mean subtraction requires electronic circuits.

    Args:
        eps (float): Small constant added to the norm for numerical stability.
                     At 4-bit precision the norm can be very small, so eps
                     prevents division by near-zero values.
                     Default: 1e-6.
        dim (int): Dimension along which to compute the L2 norm.
                   Default: -1 (last dimension, the feature dimension).
    """

    def __init__(self, eps: float = 1e-6, dim: int = -1) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply divisive power normalization.

        Args:
            x: Input tensor of any shape. Normalization is applied along
               self.dim (default: last dimension).

        Returns:
            Tensor of the same shape as x. Each slice along self.dim has
            L2 norm approximately equal to 1.0.
        """
        # Compute L2 norm along the feature dimension.
        # keepdim=True so we can broadcast-divide without reshaping.
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)

        # Divide by (norm + eps). The eps prevents NaN when x = 0.
        return x / (norm + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, dim={self.dim}"


# ---------------------------------------------------------------------------
# Self-contained tests — run with: python photonflow/normalization.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    norm_fn = DivisivePowerNorm(eps=1e-6)
    norm_fn.eval()

    print("Testing DivisivePowerNorm...")
    print()

    # --- Test 1: shape preserved ---
    x = torch.randn(4, 16, 64)
    out = norm_fn(x)
    assert out.shape == x.shape, (
        f"Shape mismatch: expected {x.shape}, got {out.shape}"
    )
    print(f"  [PASS] Test 1 — shape preserved: {tuple(out.shape)}")

    # --- Test 2: unit norm output ---
    # After DivisivePowerNorm each feature vector should have L2 norm ≈ 1.0
    x2 = torch.randn(8, 128)
    out2 = norm_fn(x2)
    norms = torch.norm(out2, p=2, dim=-1)  # shape: (8,)
    # Expected: all close to 1.0 (exact when eps is negligible vs ||x||)
    assert torch.allclose(norms, torch.ones(8), atol=1e-4), (
        f"Output norms not ≈ 1.0: {norms}"
    )
    max_deviation = (norms - 1.0).abs().max().item()
    print(f"  [PASS] Test 2 — unit norm: max deviation from 1.0 = {max_deviation:.2e}")

    # --- Test 3: zero-input safety (no NaN, no Inf) ---
    # If x = 0, the norm = 0, so we'd divide by eps → output = 0.
    # Critical: this must NOT produce NaN or Inf.
    x3 = torch.zeros(4, 32)
    out3 = norm_fn(x3)
    assert not torch.isnan(out3).any(), "NaN detected for zero input"
    assert not torch.isinf(out3).any(), "Inf detected for zero input"
    # Zero input → zero output (0 / eps = 0)
    assert torch.allclose(out3, torch.zeros_like(out3), atol=1e-6), (
        f"Zero input should give zero output, got max={out3.abs().max().item()}"
    )
    print(f"  [PASS] Test 3 — zero-input safety: no NaN/Inf, output is zero")

    # --- Test 4: gradients flow ---
    x4 = torch.randn(4, 64, requires_grad=True)
    out4 = norm_fn(x4)
    loss = out4.sum()
    loss.backward()
    assert x4.grad is not None, "Gradient is None — backward pass failed"
    assert not torch.isnan(x4.grad).any(), "NaN gradients detected"
    assert not torch.isinf(x4.grad).any(), "Inf gradients detected"
    max_grad = x4.grad.abs().max().item()
    print(f"  [PASS] Test 4 — gradients flow: max|grad| = {max_grad:.4f}, no NaN/Inf")

    # --- Test 5: direction preserved (only magnitude changes) ---
    # DivisivePowerNorm should NOT change the direction of the vector,
    # only its magnitude. cosine_similarity(x, out) should be ≈ 1.0
    x5 = torch.randn(16, 64)
    out5 = norm_fn(x5)
    cos_sim = torch.nn.functional.cosine_similarity(x5, out5, dim=-1)
    assert torch.allclose(cos_sim, torch.ones(16), atol=1e-5), (
        f"Direction changed after normalization: min cosine = {cos_sim.min().item():.6f}"
    )
    print(f"  [PASS] Test 5 — direction preserved: min cosine similarity = {cos_sim.min().item():.6f}")

    print()
    print("All tests passed.")
    print(f"  {norm_fn}")
    print()
    print("Physical interpretation:")
    print("  Photodetector measures total optical power  => L2 norm of field amplitudes")
    print("  Microring resonator feedback attenuates     => divides all elements by the norm")
    print("  Result: unit-norm output vector, fully in the optical domain")
