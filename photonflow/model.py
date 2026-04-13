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
    adaLN scale/shift          <->  Electronic boundary (same as norm gain/bias)

References:
    Dao et al., "Monarch: Expressive Structured Matrices for Efficient and
    Accurate Training," ICML 2022.
    - Definition 3.1: M = PLP^TR, L and R block-diagonal with m blocks of m*m,
      P is the stride permutation (reshape -> transpose -> reshape).
    - Section 4: Monarch replaces dense weight matrices in BOTH attention
      projections AND FFN blocks of existing architectures (ViT, GPT-2).

    Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023.
    - Figure 3, page 4199: adaLN-Zero block design with per-dimension
      scale + shift + gate from time embedding.
    - Figure 5: adaLN-Zero dramatically outperforms additive conditioning.

    Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.
    - CFM objective: v_theta(x_t, t) predicts the flow field (x1 - x0).
    - Architecture-agnostic: any network mapping (x_t, t) -> v works.
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
        - Block-diagonal L, R  <->  columns of 2x2 MZI unitary gates
        - Permutation P        <->  waveguide routing (free -- no energy cost)

    Formula:
        y = M x = P L P^T R x

    Forward algorithm (Dao 2022, Section 3.1):
        1. Reshape x to (B, m, m) -- 2D view, m = sqrt(dim)
        2. Multiply by R (block-diagonal):  x = einsum('bki,kij->bkj', x, R)
        3. Apply P^T (stride permutation = transpose of dims 1, 2)
        4. Multiply by L (block-diagonal):  x = einsum('bki,kij->bkj', x, L)
        5. Apply P (transpose back)
        6. Reshape to (B, dim)

    FLOPs: O(B * n^{3/2}) vs O(B * n^2) for a dense linear layer.
    Parameters: 2 * m^3 = 2 * n^{3/2} (L and R combined) vs n^2 for dense.

    Constraint:
        dim must be a perfect square (n = m^2). Valid: 4, 16, 64, 256, 784, 1024.

    Args:
        dim  (int):  Feature dimension. Must be a perfect square.
        bias (bool): If True, adds a learnable bias to the output. Default True.
                     Note: bias is PhotonFlow's addition -- Dao 2022 Definition 3.1
                     defines M = PLP^TR with no additive term.
    """

    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        m = math.isqrt(dim)
        if m * m != dim:
            raise ValueError(
                f"MonarchLayer requires dim to be a perfect square (n = m^2), "
                f"got dim={dim}. Valid examples: 4, 16, 64, 256, 784, 1024."
            )
        self.dim = dim
        self.m = m

        self.L = nn.Parameter(torch.empty(m, m, m))
        self.R = nn.Parameter(torch.empty(m, m, m))
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

        x = x.reshape(B, self.m, self.m)                         # (B, m, m)
        x = torch.einsum("bki,kij->bkj", x, self.R)              # (B, m, m)
        x = x.transpose(1, 2).contiguous()                        # (B, m, m)
        x = torch.einsum("bki,kij->bkj", x, self.L)              # (B, m, m)
        x = x.transpose(1, 2).contiguous()                        # (B, m, m)
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
# PhotonFlowBlock (Round 3 redesign: pre-norm + adaLN)
# ---------------------------------------------------------------------------

class PhotonFlowBlock(nn.Module):
    """One PhotonFlow block: the photonic analog of a transformer layer.

    Two sub-layers per block, mirroring DiT (Peebles 2023, Figure 3):

      Sub-layer 1 -- "Attention" (spatial mixing via Monarch pair):
        norm1(x) -> adaLN(scale1, shift1) -> MonarchL -> [Noise] -> MonarchR
        -> [Noise] -> Absorber -> gated residual (gate1)

      Sub-layer 2 -- "MLP" (nonlinear capacity via second Monarch pair):
        norm2(x) -> adaLN(scale2, shift2) -> MonarchL2 -> Absorber2
        -> MonarchR2 -> gated residual (gate2)

    Dao 2022 Section 4: Monarch replaces weights in BOTH attention AND FFN.
    Sub-layer 2 is the photonic FFN -- absorber between two Monarch layers
    acts like the activation in a standard MLP (Linear -> GELU -> Linear).

    adaLN-Zero produces 6 per-dim vectors: scale1, shift1, gate1,
    scale2, shift2, gate2 (zero-initialized so block starts as identity).

    Hardware mapping:
        DivisivePowerNorm  <->  Microring resonator + photodetector feedback
        adaLN scale/shift  <->  Electronic boundary (same as norm gain/bias)
        adaLN gate         <->  Per-channel variable optical attenuator
        MonarchL, R        <->  MZI mesh array (two columns of beamsplitters)
        SaturableAbsorber  <->  Graphene waveguide insert
        PhotonicNoise      <->  Shot noise + thermal crosstalk (training sim)

    Args:
        dim       (int):   Feature dimension (must be perfect square).
        time_dim  (int):   Dimension of the pre-computed time embedding.
        use_noise (bool):  If True, inject PhotonicNoise after each Monarch layer.
        sigma_s   (float): Shot noise std (default 0.02).
        sigma_t   (float): Thermal crosstalk std (default 0.01).
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

        # --- adaLN-Zero: 6 per-dim vectors (DiT Figure 3, page 4199) ---
        # scale1, shift1, gate1 for sub-layer 1 ("attention")
        # scale2, shift2, gate2 for sub-layer 2 ("MLP")
        # Zero-initialized so both sub-layers start as identity.
        self.adaLN_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_proj[-1].weight)
        nn.init.zeros_(self.adaLN_proj[-1].bias)

        # --- Sub-layer 1: "Attention" (Monarch pair for spatial mixing) ---
        self.norm1 = DivisivePowerNorm(num_features=dim)
        self.monarch_l = MonarchLayer(dim)
        self.monarch_r = MonarchLayer(dim)
        self.absorber1 = SaturableAbsorber()

        # --- Photonic noise injection (sub-layer 1 only, training only) ---
        if use_noise:
            self.noise_l = PhotonicNoise(sigma_s=sigma_s, sigma_t=sigma_t)
            self.noise_r = PhotonicNoise(sigma_s=sigma_s, sigma_t=sigma_t)
        else:
            self.noise_l = None
            self.noise_r = None

        # --- Sub-layer 2: "MLP" (Monarch-Absorber-Monarch, photonic FFN) ---
        # Mirrors standard MLP: Linear -> Activation -> Linear
        # MonarchL2 -> Absorber2 -> MonarchR2
        self.norm2 = DivisivePowerNorm(num_features=dim)
        self.monarch_l2 = MonarchLayer(dim)
        self.absorber2 = SaturableAbsorber()
        self.monarch_r2 = MonarchLayer(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, dim) input features.
            t_emb: (B, time_dim) pre-computed time embedding.

        Returns:
            (B, dim) output features.
        """
        # --- 6 conditioning vectors from time embedding ---
        cond = self.adaLN_proj(t_emb)  # (B, 6*dim)
        s1, sh1, g1, s2, sh2, g2 = cond.chunk(6, dim=-1)  # each (B, dim)

        # --- Sub-layer 1: "Attention" (spatial mixing) ---
        h = (1 + s1) * self.norm1(x) + sh1
        h = self.monarch_l(h)
        if self.noise_l is not None:
            h = self.noise_l(h)
        h = self.monarch_r(h)
        if self.noise_r is not None:
            h = self.noise_r(h)
        h = self.absorber1(h)
        x = x + g1 * h

        # --- Sub-layer 2: "MLP" (nonlinear feature transform) ---
        h = (1 + s2) * self.norm2(x) + sh2
        h = self.monarch_l2(h)
        h = self.absorber2(h)       # activation BETWEEN two linears (like GELU in MLP)
        h = self.monarch_r2(h)
        x = x + g2 * h

        return x

    def extra_repr(self) -> str:
        use_noise = self.noise_l is not None
        return f"dim={self.dim}, use_noise={use_noise}"


# ---------------------------------------------------------------------------
# PhotonFlowModel (Round 3: hidden_dim=784, fixed time embed, final norm)
# ---------------------------------------------------------------------------

class PhotonFlowModel(nn.Module):
    """Full PhotonFlow vector-field network: v_theta(x_t, t).

    Used as the backbone for Conditional Flow Matching (CFM). Given a noisy
    sample x_t and time t, predicts the flow field (x_1 - x_0).

    Round 3 architecture (paper-informed redesign):
        Input (B, in_dim)
          -> Linear(in_dim, hidden_dim)       # input projection (electronics)
          -> PhotonFlowBlock x num_blocks     # photonic core with adaLN
          -> DivisivePowerNorm(hidden_dim)    # final norm (like DiT)
          -> Linear(hidden_dim, in_dim)       # output projection (electronics)
          -> Output (B, in_dim)

    Time embedding (electronics-side, fixed 256-dim regardless of hidden_dim):
        t (B,) in [0, 1]
          -> SinusoidalTimeEmbedding(256)       # fixed sinusoidal encoding
          -> Linear(256, time_dim) -> SiLU()    # 2-layer MLP
          -> Linear(time_dim, time_dim)
          -> t_emb (B, time_dim)  passed to every block's adaLN

    Key design choices (from reading Dao 2022, Peebles 2023, Lipman 2023):
        - hidden_dim=784 (28^2) for MNIST: eliminates the 784->256 information
          bottleneck that prevented the model from learning. Monarch(784) with
          m=28 naturally mixes pixels in groups of 28 with stride permutation.
        - adaLN conditioning: per-dimension scale+shift from time embedding,
          enabling time-dependent feature modulation (DiT Figure 5 shows this
          dramatically outperforms additive conditioning).
        - Final norm before output_proj for training stability.

    Args:
        in_dim     (int):   Input/output dimension (784 for MNIST).
        hidden_dim (int):   Internal feature dimension. Must be a perfect square.
                            Default: 784 (= 28^2) for MNIST.
        num_blocks (int):   Number of PhotonFlowBlocks. Default: 6.
        time_dim   (int):   Time embedding dimension (after MLP). Default: 256.
        use_noise  (bool):  Inject PhotonicNoise in each block. Default: True.
        sigma_s    (float): Shot noise std. Default: 0.02.
        sigma_t    (float): Thermal crosstalk std. Default: 0.01.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 784,
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
                f"got {hidden_dim}. Valid: 256 (16^2), 784 (28^2), 1024 (32^2)."
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # --- Input / output projections (electronics-side) ---
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, in_dim)

        # Zero-init output projection (DiT pattern, Peebles 2023 page 4199).
        # Combined with zero-init adaLN gates, every block is identity at step 0,
        # so v_theta = 0 initially. output_proj learns the mean flow first, then
        # gates gradually open to let Monarch layers contribute corrections.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # --- Time embedding pipeline (electronics-side) ---
        # Fixed 256-dim sinusoidal encoding regardless of hidden_dim.
        # Avoids wasteful high-dim sinusoidal when hidden_dim is large (784).
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(256),
            nn.Linear(256, time_dim),
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

        # --- Final norm before output projection (like DiT) ---
        self.final_norm = DivisivePowerNorm(num_features=hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the flow field v_theta(x_t, t) = (x_1 - x_0).

        Args:
            x: (B, in_dim) noisy sample x_t at time t.
            t: (B,) float tensor of time values in [0, 1].

        Returns:
            (B, in_dim) predicted flow field.
        """
        # Project input to hidden_dim (electronics -> photonics interface)
        x = self.input_proj(x)

        # Compute time embedding once, share across all blocks
        t_emb = self.time_mlp(t)                  # (B, time_dim)

        # Pass through photonic blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final norm + output projection (photonics -> electronics interface)
        x = self.final_norm(x)
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

    # --- Test 3: Identity initialization -- M = I ---
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

    # --- Test 6: Multiple valid dims (including 784 for MNIST) ---
    valid_dims = [4, 16, 64, 256, 784]
    for d in valid_dims:
        lyr = MonarchLayer(dim=d, bias=False)
        lyr.eval()
        xi = torch.randn(2, d)
        oi = lyr(xi)
        assert oi.shape == (2, d), f"Shape mismatch for dim={d}"
    print(f"  [PASS] Test 6 - Valid dims: {valid_dims}")

    # --- Test 7: PhotonFlowBlock with 2 sub-layers + adaLN-Zero ---
    block = PhotonFlowBlock(dim=256, time_dim=256, use_noise=False)
    block.eval()
    x7 = torch.randn(4, 256)
    t7 = torch.rand(4, 256)   # pre-computed t_emb
    out7 = block(x7, t7)
    assert out7.shape == (4, 256), f"Block output shape: {out7.shape}"
    assert not torch.isnan(out7).any(), "NaN in block output"
    assert not torch.isinf(out7).any(), "Inf in block output"
    # With zero-init adaLN (all 6 vectors = 0):
    # gate1=0 kills sub-layer 1, gate2=0 kills sub-layer 2
    # Block is identity at initialization
    max_diff7 = (out7 - x7).abs().max().item()
    assert max_diff7 < 1e-5, (
        f"Block with gates=0 should be identity, got diff={max_diff7:.2e}"
    )
    # Verify 2 sub-layers exist
    assert hasattr(block, 'monarch_l2'), "Missing MLP sub-layer (monarch_l2)"
    assert hasattr(block, 'absorber2'), "Missing MLP sub-layer (absorber2)"
    print(
        f"  [PASS] Test 7 - PhotonFlowBlock(256): 2 sub-layers, "
        f"gates=0 -> identity (diff={max_diff7:.2e})"
    )

    # --- Test 8: PhotonFlowModel full forward (MNIST: hidden_dim=784) ---
    model = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=6, use_noise=False)
    model.eval()
    x8 = torch.randn(4, 784)
    t8 = torch.rand(4)
    out8 = model(x8, t8)
    assert out8.shape == (4, 784), f"Model output shape: {out8.shape}"
    assert not torch.isnan(out8).any(), "NaN in model output"
    assert not torch.isinf(out8).any(), "Inf in model output"
    # With zero output_proj + zero gates: v_theta = 0 at initialization
    max_out8 = out8.abs().max().item()
    assert max_out8 < 1e-5, f"Init output should be ~0 (zero output_proj), got {max_out8}"
    # Gradient check — output_proj gets gradient even with zero init
    # because grad = -2 * target * h_final^T (h_final is non-zero from input_proj)
    model.train()
    out8_train = model(x8, t8)
    out8_train.sum().backward()
    # With gate=0, Monarch params get zero gradient (expected — gates haven't opened)
    # But output_proj and adaLN_proj MUST get gradient (bootstrapping)
    assert model.output_proj.weight.grad is not None, "output_proj needs gradient"
    assert model.output_proj.weight.grad.abs().max().item() > 0, "output_proj grad must be non-zero"
    for i, blk in enumerate(model.blocks):
        assert blk.adaLN_proj[-1].weight.grad is not None, f"Block {i} adaLN needs gradient"
    n_params = model.count_parameters()
    print(
        f"  [PASS] Test 8 - PhotonFlowModel(784,784,6): (4,784) -> (4,784), "
        f"v_theta=0 at init, output_proj+adaLN get grad, {n_params:,} params"
    )

    # --- Test 9: Noise toggle -- deterministic in eval, stochastic in train ---
    model_noise = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=2, use_noise=True)
    x9 = torch.randn(4, 784)
    t9 = torch.rand(4)
    # Eval mode: deterministic (even with noise layers present)
    model_noise.eval()
    out9a = model_noise(x9, t9)
    out9b = model_noise(x9, t9)
    assert torch.equal(out9a, out9b), "Eval mode should be deterministic"
    # Train mode: need to open gates AND set output_proj non-zero
    # to see noise propagation (gate=0 kills signal, zero output_proj kills output)
    model_noise.train()
    for blk in model_noise.blocks:
        blk.adaLN_proj[-1].bias.data[2 * blk.dim:].fill_(1.0)  # open gates
    nn.init.normal_(model_noise.output_proj.weight)  # non-zero output_proj
    out9c = model_noise(x9, t9)
    out9d = model_noise(x9, t9)
    assert not torch.equal(out9c, out9d), "Train mode with open gates + noise should be stochastic"
    print("  [PASS] Test 9 - Noise toggle: eval=deterministic, train(gates open)=stochastic")

    print()
    print("All 9 tests passed.")
    print()
    print("Architecture summary:")
    model_summary = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=6)
    print(f"  MonarchLayer(784): m=28, {sum(p.numel() for p in MonarchLayer(784).parameters()):,} params "
          f"(vs {784*784:,} dense = {784*784 / (2*28**3):.1f}x fewer)")
    print(f"  PhotonFlowModel(784, 784, 6): {model_summary.count_parameters():,} total params")
    print(f"  {model_summary}")
