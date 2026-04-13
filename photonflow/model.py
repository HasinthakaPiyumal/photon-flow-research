"""
photonflow/model.py

Core architecture: MonarchLayer, PhotonFlowBlock, PhotonFlowModel.

Hardware-algorithm co-design principle:
    Every operation in this file maps to a native MZI-mesh primitive.
    No operation requires electronic offloading (O-E-O conversion).

    MonarchLayer (M = PLP^TR)  <->  MZI mesh array (beamsplitter cascade)
    SaturableAbsorber          <->  Graphene waveguide insert
    DivisivePowerNorm          <->  Microring resonator + photodetector feedback
    PhotonicNoise              <->  Shot noise + thermal crosstalk (training only)
    SinusoidalTimeEmbedding    <->  Electronic preprocessing (not on-chip)

References:
    Dao et al., "Monarch: Expressive Structured Matrices for Efficient and
    Accurate Training," ICML 2022.
    - Definition 3.1: M = PLP^TR, L and R block-diagonal with m blocks of m×m,
      P is the stride permutation (reshape → transpose → reshape).
    - Section 3.1: Algorithm for computing Mx in O(n^{3/2}) FLOPs.

    Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023.
    - Zero-initialized residual (alpha=0 trick) for stable deep-network training.

    Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.
    - CFM objective: vθ(xt, t) predicts the flow field (x1 - x0).
"""

import math

import torch
import torch.nn as nn

from photonflow.activation import SaturableAbsorber
from photonflow.normalization import DivisivePowerNorm
from photonflow.noise import PhotonicNoise

__all__ = ["MonarchLayer", "PhotonFlowBlock", "PhotonFlowModel"]


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (electronics-side preprocessing)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional encoding for continuous time t in [0, 1].

    Produces embeddings of shape (B, dim) from scalar time values (B,).
    This runs on electronic hardware (not photonic) as a preprocessing step.
    A two-layer SiLU MLP in PhotonFlowModel projects these to time_dim.

    Args:
        dim (int): Output embedding dimension. Should be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float tensor with values in [0, 1].

        Returns:
            (B, dim) sinusoidal embeddings.
        """
        half = self.dim // 2
        # Geometric frequency spacing: freqs[k] = exp(-log(10000) * k / (half-1))
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None, :]      # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# ---------------------------------------------------------------------------
# MonarchLayer: M = PLP^T R
# ---------------------------------------------------------------------------

class MonarchLayer(nn.Module):
    """Monarch matrix layer: M = PLP^T R (Dao et al. 2022, Definition 3.1).

    Maps to an MZI mesh array on a silicon photonic chip. The computation
    graph of a Monarch matrix is structurally identical to a cascade of MZI
    beamsplitters:
        - Block-diagonal L, R  <->  columns of 2×2 MZI unitary gates
        - Permutation P        <->  waveguide routing (free — no energy cost)

    Formula:
        y = M x = P L P^T R x

    Forward algorithm (Dao 2022, Section 3.1):
        1. Reshape x to (B, m, m) — 2D view, m = sqrt(dim)
        2. Multiply by R (block-diagonal):  x = einsum('bki,kij->bkj', x, R)
        3. Apply P^T (stride permutation = transpose of dims 1, 2)
        4. Multiply by L (block-diagonal):  x = einsum('bki,kij->bkj', x, L)
        5. Apply P (transpose back)
        6. Reshape to (B, dim)

    FLOPs: O(B × n^{3/2}) vs O(B × n²) for a dense linear layer.
    Parameters: 2 × m³ = 2 × n^{3/2} (L and R combined) vs n² for dense.

    Constraint:
        dim must be a perfect square (n = m²). Valid: 4, 16, 64, 256, 784, 1024.

    Args:
        dim  (int):  Feature dimension. Must be a perfect square.
        bias (bool): If True, adds a learnable bias to the output. Default True.
                     Note: bias is PhotonFlow's addition — Dao 2022 Definition 3.1
                     defines M = PLP^TR with no additive term.
    """

    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        m = math.isqrt(dim)
        if m * m != dim:
            raise ValueError(
                f"MonarchLayer requires dim to be a perfect square (n = m²), "
                f"got dim={dim}. Valid examples: 4, 16, 64, 256, 784, 1024."
            )
        self.dim = dim
        self.m = m

        # L and R: m blocks of (m × m) each.
        # Shape (m, m, m) = (num_blocks, in_per_block, out_per_block).
        # L and R are both block-diagonal factors in M = PLP^TR.
        #
        # Convention note: Dao 2022 Eq. 2 indexes L_{j,ℓ,k} and R_{k,j,i}
        # as [block, output, input]. We store [block, input, output] and use
        # einsum 'bki,kij->bkj' which computes x @ W (input-on-left),
        # equivalent to the paper's W^T @ x per block. Same expressive power
        # — just a transposed storage convention within each block.
        self.L = nn.Parameter(torch.empty(m, m, m))
        self.R = nn.Parameter(torch.empty(m, m, m))
        # Bias: PhotonFlow addition (not in Dao 2022 Def 3.1, which defines
        # M = PLP^TR with no additive term). Standard in NN layers.
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize L and R as identity blocks.

        With L[i] = R[i] = I for all i, M = PIP^TI = PP^T = I (identity).
        This means the layer initially passes through its input unchanged,
        which is stable for flow matching (trivial initial vector field).
        """
        for i in range(self.m):
            nn.init.eye_(self.L.data[i])
            nn.init.eye_(self.R.data[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Monarch transform y = PLP^TR x.

        Args:
            x: (B, dim) input tensor.

        Returns:
            (B, dim) output tensor.
        """
        B = x.shape[0]

        # Reshape to 2D view: x[b, k, i] = x_flat[b, k*m + i]
        x = x.reshape(B, self.m, self.m)                         # (B, m, m)

        # Step 1 — Multiply by R (block-diagonal):
        #   result[b, k, j] = sum_i x[b, k, i] * R[k, i, j]
        #   For each block k: x[b, k, :] @ R[k]  (R[k] is m×m, indexed [in, out])
        x = torch.einsum("bki,kij->bkj", x, self.R)              # (B, m, m)

        # Step 2 — Apply P^T (stride permutation = transpose of dims 1 and 2):
        #   Swaps block index and within-block index, implementing the
        #   "reshape → transpose → reshape" stride permutation of Dao 2022.
        #   Key property: P is symmetric (P = P^T = P^{-1}), so applying P
        #   and P^T are both the same transpose(1,2) operation.
        x = x.transpose(1, 2).contiguous()                        # (B, m, m)

        # Step 3 — Multiply by L (block-diagonal):
        #   result[b, k, j] = sum_i x[b, k, i] * L[k, i, j]
        x = torch.einsum("bki,kij->bkj", x, self.L)              # (B, m, m)

        # Step 4 — Apply P (same transpose as Step 2, since P = P^T):
        x = x.transpose(1, 2).contiguous()                        # (B, m, m)

        # Flatten to (B, dim)
        x = x.reshape(B, self.dim)

        if self.bias is not None:
            x = x + self.bias

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, m={self.m}, "
            f"params={2 * self.m ** 3} (vs {self.dim ** 2} dense), "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# PhotonFlowBlock
# ---------------------------------------------------------------------------

class PhotonFlowBlock(nn.Module):
    """One PhotonFlow block: the photonic analog of a transformer layer.

    Block order (CLAUDE.md spec + paper draft Fig. 1):
        x_in
          -> MonarchL (first Monarch layer)
          -> [PhotonicNoise]             # training only, optional
          -> MonarchR (second Monarch layer)
          -> [PhotonicNoise]             # training only, optional
          -> SaturableAbsorber           # tanh(alpha*x)/alpha, alpha=0.8
          -> DivisivePowerNorm           # x / (||x||_2 + eps) * gain + bias
          -> residual: x_in + alpha * photonic_out + time_proj(t_emb)
        Note: time_proj is OUTSIDE the alpha gate so that time conditioning
        contributes from step 0 (alpha starts near zero). This separates the
        photonic path (gated by alpha) from electronic time conditioning.

    Hardware mapping:
        MonarchL, R   <->  MZI mesh array (two columns of beamsplitters)
        SaturableAbs  <->  Graphene waveguide insert
        PowerNorm     <->  Microring resonator + photodetector feedback
        PhotonicNoise <->  Shot noise + thermal crosstalk (training simulation)

    Residual scaling (inspired by Peebles & Xie 2023, DiT):
        self.alpha is a learned scalar nn.Parameter initialized to 1.0.
        Initially: output = x_in + 1.0 * norm(absorber(monarch(x_in))) + time_proj(t_emb).
        The photonic path is a full contributor from step 0, giving Monarch
        layers meaningful gradients (~ 0.02 scale with output_proj gain=0.02).

        Why NOT alpha=0 (DiT style)?
            DiT uses alpha=0 + per-dimension time-conditioned gates (adaLN-Zero)
            which gradually open gradient paths via learned gate vectors.
            PhotonFlow has NO per-dimension gating — only a scalar alpha and
            a separate additive time_proj. With alpha=0, the photonic path is
            completely dead (zero gradient to Monarch layers). Even alpha=1e-6
            proved insufficient in experiments (loss stuck at 0.75, pure noise
            output). alpha=1.0 is required for the model to learn.

        Simplification vs DiT paper (Figure 3, page 4199):
            DiT uses per-dimension alpha VECTORS regressed from timestep+class
            via an MLP (with MLP final layer zero-initialized), and uses TWO
            separate alphas per block (alpha_1 for attention, alpha_2 for MLP).
            PhotonFlow uses a single time-INDEPENDENT scalar alpha per block.
            Rationale: (1) a scalar is more hardware-friendly (one attenuation
            factor for the entire waveguide array), and (2) time-conditioning
            is handled by the SEPARATE additive time_proj(t_emb) path (outside
            the alpha gate, so it contributes from step 0).

    Args:
        dim      (int):   Feature dimension (must be perfect square for MonarchLayer).
        time_dim (int):   Dimension of the pre-computed time embedding.
        use_noise (bool): If True, inject PhotonicNoise after each Monarch layer.
        sigma_s  (float): Shot noise std (default 0.02).
        sigma_t  (float): Thermal crosstalk std (default 0.01).
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        use_noise: bool = True,
        sigma_s: float = 0.02,
        sigma_t: float = 0.01,
    ) -> None:
        super().__init__()
        self.dim = dim

        # Monarch layer pair (replaces self-attention)
        self.monarch_l = MonarchLayer(dim)
        self.monarch_r = MonarchLayer(dim)

        # Photonic noise injection (training only, disabled at eval)
        if use_noise:
            self.noise_l = PhotonicNoise(sigma_s=sigma_s, sigma_t=sigma_t)
            self.noise_r = PhotonicNoise(sigma_s=sigma_s, sigma_t=sigma_t)
        else:
            self.noise_l = None
            self.noise_r = None

        # Photonic activation + normalization
        self.absorber = SaturableAbsorber()          # tanh(0.8x)/0.8
        self.norm = DivisivePowerNorm(num_features=dim)  # x / (||x||_2 + eps) * gain + bias

        # Per-block time embedding projection (electronics-side)
        self.time_proj = nn.Linear(time_dim, dim)

        # Residual scale (inspired by DiT, Peebles & Xie 2023).
        # Initialized to 1.0 so the photonic path is a FULL contributor from
        # step 0. Unlike DiT which uses per-dimension time-conditioned gates
        # (adaLN-Zero) to gradually open gradient paths, PhotonFlow has a
        # scalar alpha + separate additive time_proj. There is no per-dimension
        # gating mechanism to "open" the path later, so alpha MUST start at a
        # meaningful value. alpha=1.0 with identity-initialized Monarch layers
        # means the block initially acts as x + x/||x||*gain + time_proj(t_emb)
        # which is stable (DivisivePowerNorm bounds the photonic path magnitude).
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, dim) input features.
            t_emb: (B, time_dim) pre-computed time embedding from PhotonFlowModel.

        Returns:
            (B, dim) output features.
        """
        residual = x

        # --- Photonic core (MZI mesh) ---
        x = self.monarch_l(x)
        if self.noise_l is not None:
            x = self.noise_l(x)

        x = self.monarch_r(x)
        if self.noise_r is not None:
            x = self.noise_r(x)

        # --- Photonic nonlinearity + normalization ---
        x = self.absorber(x)
        x = self.norm(x)

        # --- Near-zero residual skip (DiT alpha trick) ---
        # Time embedding is added OUTSIDE the alpha gate so that time
        # conditioning contributes from step 0 (even when alpha ≈ 0).
        # Physically: photonic path (Monarch→absorber→norm) is gated by
        # alpha; electronic time conditioning is a separate additive signal.
        return residual + self.alpha * x + self.time_proj(t_emb)

    def extra_repr(self) -> str:
        use_noise = self.noise_l is not None
        return f"dim={self.dim}, use_noise={use_noise}, alpha={self.alpha.item():.4f}"


# ---------------------------------------------------------------------------
# PhotonFlowModel
# ---------------------------------------------------------------------------

class PhotonFlowModel(nn.Module):
    """Full PhotonFlow vector-field network: v_theta(x_t, t).

    Used as the backbone for Conditional Flow Matching (CFM). Given a noisy
    sample x_t and time t, predicts the flow field (x_1 - x_0).

    Architecture:
        Input (B, in_dim)
          -> Linear(in_dim, hidden_dim)     # input projection (electronics)
          -> PhotonFlowBlock × num_blocks   # photonic core
          -> Linear(hidden_dim, in_dim)     # output projection (electronics)
          -> Output (B, in_dim)

    Time embedding pipeline (electronics-side, runs before photonic blocks):
        t (B,) in [0, 1]
          -> SinusoidalTimeEmbedding(hidden_dim)     # sinusoidal encoding
          -> Linear(hidden_dim, time_dim) -> SiLU()  # 2-layer MLP
          -> Linear(time_dim, time_dim)
          -> t_emb (B, time_dim)  passed to every block

    Dimension constraint:
        hidden_dim must be a perfect square (required by MonarchLayer).
        Default hidden_dim=256 (= 16²). Other valid choices: 784 (28²), 1024 (32²).

    Args:
        in_dim     (int):   Input/output dimension (784 for MNIST, 3072 for CIFAR-10).
        hidden_dim (int):   Internal feature dimension. Must be a perfect square.
                            Default: 256 (= 16²).
        num_blocks (int):   Number of PhotonFlowBlocks. Default: 6.
        time_dim   (int):   Time embedding dimension (after MLP). Default: 256.
        use_noise  (bool):  Inject PhotonicNoise in each block. Default: True.
        sigma_s    (float): Shot noise std. Default: 0.02.
        sigma_t    (float): Thermal crosstalk std. Default: 0.01.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        time_dim: int = 256,
        use_noise: bool = True,
        sigma_s: float = 0.02,
        sigma_t: float = 0.01,
    ) -> None:
        super().__init__()
        # Validate hidden_dim is a perfect square
        m = math.isqrt(hidden_dim)
        if m * m != hidden_dim:
            raise ValueError(
                f"hidden_dim must be a perfect square for MonarchLayer, "
                f"got {hidden_dim}. Valid: 256 (16²), 784 (28²), 1024 (32²)."
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # --- Input / output projections (electronics-side) ---
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_dim)

        # Small-random output projection initialization.
        # DiT (Peebles & Xie 2023) uses zero-init here, relying on per-dimension
        # time-conditioned gates (adaLN-Zero) to gradually open gradient paths.
        # PhotonFlow uses a simpler scalar alpha, which cannot overcome a zero
        # output_proj — gradients to the photonic blocks are exactly zero when
        # Wout = 0, causing complete gradient starvation. Small Xavier init
        # (gain=0.02) gives blocks non-zero gradients from step 0 while keeping
        # initial predictions small. The bias stays zero.
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)

        # --- Time embedding pipeline (electronics-side) ---
        # SinusoidalTimeEmbedding → 2-layer SiLU MLP → time_dim
        # SiLU is used here (not SaturableAbsorber) because this runs on electronics.
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- Photonic blocks ---
        self.blocks = nn.ModuleList([
            PhotonFlowBlock(
                dim=hidden_dim,
                time_dim=time_dim,
                use_noise=use_noise,
                sigma_s=sigma_s,
                sigma_t=sigma_t,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the flow field v_theta(x_t, t) = (x_1 - x_0).

        Args:
            x: (B, in_dim) noisy sample x_t at time t.
            t: (B,) float tensor of time values in [0, 1].

        Returns:
            (B, in_dim) predicted flow field.
        """
        # Project input to hidden_dim (electronics → photonics interface)
        x = self.input_proj(x)

        # Compute time embedding once, share across all blocks
        t_emb = self.time_mlp(t)                  # (B, time_dim)

        # Pass through photonic blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Project back to in_dim (photonics → electronics interface)
        return self.output_proj(x)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"num_blocks={self.num_blocks}"
        )

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    torch.manual_seed(42)
    print("Testing photonflow/model.py ...\n")

    # --- Test 1: MonarchLayer shape preservation ---
    layer = MonarchLayer(dim=256)
    layer.eval()
    x = torch.randn(4, 256)
    out = layer(x)
    assert out.shape == (4, 256), f"Shape mismatch: {out.shape}"
    print("  [PASS] Test 1 - MonarchLayer shape: (4, 256) -> (4, 256)")

    # --- Test 2: Parameter count is O(n^{3/2}) ---
    m = 16   # sqrt(256)
    expected_params = 2 * m ** 3 + 256  # L + R + bias
    actual_params = sum(p.numel() for p in layer.parameters())
    assert actual_params == expected_params, (
        f"Param count {actual_params} != {expected_params}"
    )
    dense_params = 256 * 256
    ratio = dense_params / (2 * m ** 3)
    print(
        f"  [PASS] Test 2 - Param count: {actual_params - 256} (L+R) "
        f"vs {dense_params} (dense) = {ratio:.1f}x fewer"
    )

    # --- Test 3: Identity initialization — M ≈ I ---
    # With L[i] = R[i] = I: M = PIP^TI = PP^T = I, so output ≈ input.
    # Bias is zero by construction, so MonarchLayer(x) ≈ x.
    layer_id = MonarchLayer(dim=256, bias=False)
    layer_id.eval()
    x3 = torch.randn(8, 256)
    out3 = layer_id(x3)
    max_diff = (out3 - x3).abs().max().item()
    assert max_diff < 1e-5, f"Identity init: max|Mx - x| = {max_diff:.2e} > 1e-5"
    print(f"  [PASS] Test 3 - Identity init: max|Mx - x| = {max_diff:.2e} (should be ~0)")

    # --- Test 4: Gradients flow through L, R, bias ---
    layer4 = MonarchLayer(dim=64)
    layer4.train()
    x4 = torch.randn(4, 64)
    loss = layer4(x4).sum()
    loss.backward()
    assert layer4.L.grad is not None, "No gradient for L"
    assert layer4.R.grad is not None, "No gradient for R"
    assert layer4.bias.grad is not None, "No gradient for bias"
    print("  [PASS] Test 4 - Gradients flow through L, R, bias")

    # --- Test 5: Invalid dim raises ValueError ---
    try:
        _ = MonarchLayer(dim=15)
        print("  [FAIL] Test 5 - Should have raised ValueError for dim=15")
        sys.exit(1)
    except ValueError as e:
        print(f"  [PASS] Test 5 - Invalid dim rejected: {e}")

    # --- Test 6: Multiple valid dims ---
    valid_dims = [4, 16, 64, 256, 784]
    for d in valid_dims:
        lyr = MonarchLayer(dim=d, bias=False)
        lyr.eval()
        xi = torch.randn(2, d)
        oi = lyr(xi)
        assert oi.shape == (2, d), f"Shape mismatch for dim={d}"
    print(f"  [PASS] Test 6 - Valid dims: {valid_dims}")

    # --- Test 7: PhotonFlowBlock shape + photonic path contribution ---
    block = PhotonFlowBlock(dim=256, time_dim=256)
    block.eval()
    x7 = torch.randn(4, 256)
    t7 = torch.rand(4, 256)   # pre-computed t_emb
    out7 = block(x7, t7)
    assert out7.shape == (4, 256), f"Block output shape: {out7.shape}"
    assert not torch.isnan(out7).any(), "NaN in block output"
    assert not torch.isinf(out7).any(), "Inf in block output"
    # With alpha=1.0, photonic path is a full contributor:
    # output = residual + alpha * norm(absorber(monarch(x))) + time_proj(t_emb)
    # Verify the photonic path contributes meaningfully (not negligible)
    photonic_contrib = out7 - x7 - block.time_proj(t7)  # = alpha * norm_output
    pc_max = photonic_contrib.abs().max().item()
    assert pc_max > 0.1, (
        f"Photonic path should contribute meaningfully, got max={pc_max:.4f}"
    )
    assert pc_max < 100.0, (
        f"Photonic path should not explode, got max={pc_max:.4f}"
    )
    print(
        f"  [PASS] Test 7 - PhotonFlowBlock: (4,256) -> (4,256), "
        f"alpha=1.0: photonic path max contrib={pc_max:.2f}"
    )

    # --- Test 8: PhotonFlowModel full forward (MNIST config) ---
    model = PhotonFlowModel(in_dim=784, hidden_dim=256, num_blocks=6, use_noise=False)
    model.eval()
    x8 = torch.randn(4, 784)
    t8 = torch.rand(4)
    out8 = model(x8, t8)
    assert out8.shape == (4, 784), f"Model output shape: {out8.shape}"
    assert not torch.isnan(out8).any(), "NaN in model output"
    assert not torch.isinf(out8).any(), "Inf in model output"
    # Initial output should be small (output_proj has gain=0.02, time_proj
    # contributions are projected through small output_proj). Not zero though —
    # time_proj is active from step 0.
    max_out8 = out8.abs().max().item()
    assert max_out8 < 10.0, f"Initial output unexpectedly large: {max_out8}"
    # Gradient check
    model.train()
    out8_train = model(x8, t8)
    out8_train.sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Params with no gradient: {no_grad}"
    n_params = model.count_parameters()
    print(
        f"  [PASS] Test 8 - PhotonFlowModel MNIST: (4,784) -> (4,784), "
        f"no NaN/Inf, init max|out|={max_out8:.4f}, grads flow, {n_params:,} params"
    )

    # --- Test 9: Noise toggle — deterministic in eval, stochastic in train ---
    model_noise = PhotonFlowModel(in_dim=784, hidden_dim=256, num_blocks=2, use_noise=True)
    x9 = torch.randn(4, 784)
    t9 = torch.rand(4)
    # Eval mode: two identical forward passes must give identical results
    model_noise.eval()
    out9a = model_noise(x9, t9)
    out9b = model_noise(x9, t9)
    assert torch.equal(out9a, out9b), "Eval mode should be deterministic"
    # Train mode: two forward passes should differ (noise is stochastic).
    # With alpha=1.0 (default init) and non-zero output_proj (Xavier gain=0.02),
    # photonic noise is clearly visible in the output without any manual override.
    model_noise.train()
    out9c = model_noise(x9, t9)
    out9d = model_noise(x9, t9)
    assert not torch.equal(out9c, out9d), "Train mode with noise should be stochastic"
    print("  [PASS] Test 9 - Noise toggle: eval=deterministic, train=stochastic")

    print()
    print("All 9 tests passed.")
    print()
    print("Architecture summary:")
    model_summary = PhotonFlowModel(in_dim=784, hidden_dim=256, num_blocks=6)
    print(f"  MonarchLayer(256): {sum(p.numel() for p in MonarchLayer(256).parameters()):,} params "
          f"(vs {256*256:,} dense = {256*256 / (2*16**3):.1f}x fewer)")
    print(f"  PhotonFlowModel(784, 256, 6): {model_summary.count_parameters():,} total params")
    print(f"  {model_summary}")
