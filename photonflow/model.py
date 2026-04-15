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
from photonflow.time_embed import WavelengthCodedTime
from photonflow.layers import MonarchLinear, PPLNSigmoid

__all__ = ["MonarchLayer", "PhotonFlowBlock", "PhotonFlowModel"]


# ---------------------------------------------------------------------------
# Photon-native time embedding: WavelengthCodedTime is imported from
# photonflow.time_embed (Moss 2022 AWGR wavelength-dispersive look-up).
# The legacy SinusoidalTimeEmbedding class was DELETED as part of the
# strict-photonic rewrite -- its electronic sin/cos/exp computation has no
# place in a zero-OEO forward graph.
# ---------------------------------------------------------------------------


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
        init (str):  Block init mode (see `_init_weights`).  Default 'random'
                     (Xavier gain 0.1) -- 'identity' traps the network in
                     a near-linear basin (Hardt & Ma 2017) so it is no
                     longer the default.
        num_factors (int): Number of stacked Monarch factors.

    Notes (photon-native rewrite):
        * `bias` kwarg DELETED.  Dao 2022 Def 3.1 has no additive term; adding
          a per-channel bias optically would require an active DAC+modulator
          per channel (non-photonic).
        * `unitary_project` kwarg DELETED and hard-wired to True.  The MZI
          mesh realises only unitary matrices (Shen 2017 + Clements 2016), so
          L_i, R_i are Cayley-projected onto O(m) on every forward.  Cayley
          (Arjovsky 2016, Lezcano-Casado 2019) is the standard parameterisa-
          tion of orthogonal matrices and admits analytic-gradient MZI
          training (Zhan *LPR* 2024).
    """

    def __init__(self, dim: int, init: str = "random",
                 num_factors: int = 1) -> None:
        super().__init__()
        m = math.isqrt(dim)
        if m * m != dim:
            raise ValueError(
                f"MonarchLayer requires dim to be a perfect square (n = m^2), "
                f"got dim={dim}. Valid examples: 4, 16, 64, 256, 784, 1024."
            )
        self.dim = dim
        self.m = m
        self.num_factors = num_factors

        # Stacked Monarch factors (Dao 2022, Section 3.2):
        # M = M_k * ... * M_2 * M_1, where each M_i = P L_i P^T R_i.
        # The product is NOT closed — M1*M2 is strictly more expressive.
        # Maps to cascading multiple MZI mesh stages on photonic hardware.
        self.Ls = nn.ParameterList([
            nn.Parameter(torch.empty(m, m, m)) for _ in range(num_factors)
        ])
        self.Rs = nn.ParameterList([
            nn.Parameter(torch.empty(m, m, m)) for _ in range(num_factors)
        ])
        # No bias -- Dao Def 3.1 does not permit it and the photonic MZI mesh
        # has no additive-bias primitive.
        self.bias = None

        self._init_mode = init
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise the m x m blocks of every L and R factor.

        Modes:
          - 'identity' (default): each block = I.  Trains into near-linear
             collapse for deep structured nets (Hardt & Ma 2017).
          - 'random': Xavier normal, gain 0.1.  Breaks the identity basin.
          - 'orthogonal': each block is a random orthogonal m x m matrix (QR
             of Gaussian).  Saxe, McClelland & Ganguli 2013 show this makes
             deep-linear training time independent of depth.
          - 'dct': each block is the DCT-II basis matrix of size m.  The
             real-valued analogue of the DFT basis used in Monarch Mixer
             (Wang 2023); photonically this IS the Fourier-mesh operation.
        """
        for f in range(self.num_factors):
            if self._init_mode == "random":
                for i in range(self.m):
                    nn.init.xavier_normal_(self.Ls[f].data[i], gain=0.1)
                    nn.init.xavier_normal_(self.Rs[f].data[i], gain=0.1)
            elif self._init_mode == "orthogonal":
                for i in range(self.m):
                    # Random Gaussian -> QR -> take Q (m x m orthogonal).
                    qL, _ = torch.linalg.qr(torch.randn(self.m, self.m))
                    qR, _ = torch.linalg.qr(torch.randn(self.m, self.m))
                    self.Ls[f].data[i].copy_(qL)
                    self.Rs[f].data[i].copy_(qR)
            elif self._init_mode == "dct":
                # DCT-II basis: C[k, n] = sqrt(2/N) * cos(pi*(2n+1)*k/(2N)),
                # with C[0, n] = sqrt(1/N). Orthogonal and frequency-ordered.
                n_idx = torch.arange(self.m, dtype=torch.float32)
                k_idx = torch.arange(self.m, dtype=torch.float32)
                # angles[k, n] = pi * (2n+1) * k / (2m)
                ang = math.pi * (2 * n_idx.unsqueeze(0) + 1) * k_idx.unsqueeze(1) / (2 * self.m)
                C = math.sqrt(2.0 / self.m) * torch.cos(ang)
                C[0] = math.sqrt(1.0 / self.m)
                for i in range(self.m):
                    self.Ls[f].data[i].copy_(C)
                    self.Rs[f].data[i].copy_(C)
            else:
                # Identity init: each factor = I, so product = I.
                for i in range(self.m):
                    nn.init.eye_(self.Ls[f].data[i])
                    nn.init.eye_(self.Rs[f].data[i])

    def _cayley_project(self, raw: torch.Tensor) -> torch.Tensor:
        """Cayley transform: map the m x m parameter blocks onto O(m).

        Q = (I - A)(I + A)^{-1}, where A = (raw - raw^T) / 2 is skew-symmetric.
        Returns a batched orthogonal matrix of the same shape as `raw`
        (expected shape: (m, m, m)).  Called inside `forward` only when
        `unitary_project=True` -- otherwise the unconstrained parameters are
        used directly, matching the legacy behaviour.
        """
        m = raw.shape[-1]
        I = torch.eye(m, device=raw.device, dtype=raw.dtype)
        I = I.expand(raw.shape[0], -1, -1)                   # (m, m, m)
        A = 0.5 * (raw - raw.transpose(-1, -2))              # skew-symm
        # Solve (I + A) Q = (I - A)  =>  Q = (I + A)^{-1}(I - A), then take
        # the transpose so that Q Q^T = I (Cayley convention used in
        # Arjovsky 2016, Lezcano-Casado 2019).
        return torch.linalg.solve(I + A, I - A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stacked Monarch transform y = M_k ... M_2 M_1 x.

        Each factor: M_i x = P L_i P^T R_i x.

        Args:
            x: (B, dim) input tensor.

        Returns:
            (B, dim) output tensor.
        """
        B = x.shape[0]

        # Hard-wired unitary projection: the Cayley transform is the only
        # parameterisation compatible with the MZI mesh realising U Sigma V^T
        # (Shen 2017, Clements 2016).  Every forward is through O(m).
        for f in range(self.num_factors):
            L = self._cayley_project(self.Ls[f])
            R = self._cayley_project(self.Rs[f])

            x = x.reshape(B, self.m, self.m)
            x = torch.einsum("bki,kij->bkj", x, R)
            x = x.transpose(1, 2).contiguous()
            x = torch.einsum("bki,kij->bkj", x, L)
            x = x.transpose(1, 2).contiguous()
            x = x.reshape(B, self.dim)

        return x   # no bias -- Dao Def 3.1 has none

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, m={self.m}, num_factors={self.num_factors}, "
            f"params={2 * self.num_factors * self.m ** 3} "
            f"(vs {self.dim ** 2} dense), unitary=True, bias=False"
        )


# ---------------------------------------------------------------------------
# PhotonFlowBlock (Round 3 redesign: pre-norm + adaLN)
# ---------------------------------------------------------------------------

class PhotonFlowBlock(nn.Module):
    """One photon-native PhotonFlow block.

    Two sub-layers per block, DiT-style but with every electronic op
    replaced by a photonic primitive:

      Sub-layer 1 -- spatial mixing (one MZI mesh pair):
        norm1(x) + cond_bias1   (additive-only conditioning, Nature Photonics 2024)
        -> MonarchL (unitary) -> MonarchR (unitary) -> SaturableAbsorber
        -> PhotonicNoise -> x + h   (coherent optical addition, no gate)

      Sub-layer 2 -- photonic FFN (second MZI mesh pair):
        norm2(x) + cond_bias2
        -> MonarchL2 (unitary) -> SaturableAbsorber -> MonarchR2 (unitary)
        -> PhotonicNoise -> x + h

    Hardware map (strict photon-native):
        DivisivePowerNorm  <->  microring + photodetector feedback + fixed SOA gain
        SaturableAbsorber  <->  graphene waveguide insert (Shen 2017)
        MonarchLayer       <->  Cayley-unitary MZI mesh (Shen 2017 + Clements 2016)
        MonarchLinear      <->  padded MZI mesh for non-square projections
        cond_bias_proj     <->  a single wavelength-coded bias injection
                                (no per-dim scale/shift/gate -- no adaLN)
        coherent add       <->  tunable directional coupler
        PhotonicNoise      <->  shot + thermal + phase-space noise (training only)

    Every kwarg controls a photonic hyperparameter (dimensions, α, σ_s, σ_t);
    there are no electronic-mode switches.

    Args:
        dim        (int):   Feature dimension (must be perfect square).
        time_dim   (int):   Dimension of the time embedding.
        use_noise  (bool):  Inject PhotonicNoise after each Monarch pair.
        sigma_s    (float): Shot noise σ_D (Shen 2017: 0.001).
        sigma_t    (float): Thermal crosstalk σ_φ proxy (Shen 2017: 0.005).
        shot_signal_dependent (bool): Shen's physically-accurate shot noise.
        monarch_init (str): Monarch-block init mode (default 'random').
        adaln_init_std (float): init magnitude for the cond_bias_proj Monarch.
        num_monarch_factors (int): number of stacked Monarch factors per layer.
        absorber_alpha (float): saturable-absorber α.
        absorber_leaky_slope (float): SA linear-bypass slope.
        learnable_absorber_alpha (bool): make α a trainable parameter.
        mean_center_norm (bool): subtract per-sample mean in DPN (off by default).
        phase_noise_sigma (float): σ_φ multiplicative jitter (Shen 2017 M8b).
        cumulative_loss_db_per_stage (float): Shen's 0.0003 dB/stage optical loss.
        block_index (int): zero-based block index (for cumulative-loss staging).
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        *,
        use_noise: bool = True,
        sigma_s: float = 0.001,
        sigma_t: float = 0.005,
        shot_signal_dependent: bool = True,
        monarch_init: str = "random",
        adaln_init_std: float = 0.02,
        num_monarch_factors: int = 1,
        absorber_alpha: float = 0.8,
        absorber_leaky_slope: float = 0.05,
        learnable_absorber_alpha: bool = False,
        mean_center_norm: bool = False,
        phase_noise_sigma: float = 0.005,
        cumulative_loss_db_per_stage: float = 0.0003,
        block_index: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.block_index = int(block_index)
        self._stage1 = 2 * self.block_index
        self._stage2 = 2 * self.block_index + 1

        # ---- Photon-native conditioning: single additive bias per sub-layer ----
        # `cond_bias_proj` maps the time embedding (B, time_dim) to a (B, 2*dim)
        # bias which we chunk into (cb1, cb2), added to each sub-layer's
        # normalised features as a coherent optical offset.  No per-channel
        # scale, shift, or gate -- those are deleted along with adaLN.
        _cb_scale = adaln_init_std if adaln_init_std > 0 else 0.02
        self.cond_bias_proj = MonarchLinear(
            time_dim, 2 * dim, init_scale=_cb_scale, bias=True
        )

        # ---- Sub-layer 1: Spatial mixing (Monarch pair + SA) ----
        # DivisivePowerNorm has a FIXED buffer gain (no learnable affine).
        self.norm1 = DivisivePowerNorm(
            num_features=dim,
            mean_center=mean_center_norm,
        )
        _monarch_kwargs = dict(
            init=monarch_init,
            num_factors=num_monarch_factors,
        )
        self.monarch_l = MonarchLayer(dim, **_monarch_kwargs)
        self.monarch_r = MonarchLayer(dim, **_monarch_kwargs)
        self.absorber1 = SaturableAbsorber(
            alpha=absorber_alpha,
            learnable_alpha=learnable_absorber_alpha,
            leaky_slope=absorber_leaky_slope,
            intensity_mode="differential",   # photon-native: dual-λ MRM proxy
        )

        # ---- Photonic noise modules (training only) ----
        _noise_kwargs = dict(
            sigma_s=sigma_s,
            sigma_t=sigma_t,
            shot_signal_dependent=shot_signal_dependent,
            phase_noise_sigma=float(phase_noise_sigma),
            cumulative_loss_db_per_stage=float(cumulative_loss_db_per_stage),
        )
        if use_noise:
            self.noise1 = PhotonicNoise(**_noise_kwargs, stage_index=self._stage1)
            self.noise2 = PhotonicNoise(**_noise_kwargs, stage_index=self._stage2)
        else:
            self.noise1 = None
            self.noise2 = None

        # ---- Sub-layer 2: Photonic FFN (Monarch-SA-Monarch) ----
        self.norm2 = DivisivePowerNorm(
            num_features=dim,
            mean_center=mean_center_norm,
        )
        self.monarch_l2 = MonarchLayer(dim, **_monarch_kwargs)
        self.absorber2 = SaturableAbsorber(
            alpha=absorber_alpha,
            learnable_alpha=learnable_absorber_alpha,
            leaky_slope=absorber_leaky_slope,
            intensity_mode="differential",
        )
        self.monarch_r2 = MonarchLayer(dim, **_monarch_kwargs)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, dim) input features.
            t_emb: (B, time_dim) time embedding.

        Returns:
            (B, dim) output features.
        """
        # Photon-native conditioning: single wavelength-encoded bias per sub-layer
        cb = self.cond_bias_proj(t_emb)        # (B, 2*dim)
        cb1, cb2 = cb.chunk(2, dim=-1)

        # Sub-layer 1: norm -> bias -> Monarch_L -> Monarch_R -> SA -> noise -> +x
        h = self.norm1(x) + cb1
        h = self.monarch_l(h)
        h = self.monarch_r(h)
        h = self.absorber1(h)
        if self.noise1 is not None:
            h = self.noise1(h)
        x = x + h                              # coherent optical addition

        # Sub-layer 2: norm -> bias -> Monarch_L2 -> SA -> Monarch_R2 -> noise -> +x
        h = self.norm2(x) + cb2
        h = self.monarch_l2(h)
        h = self.absorber2(h)
        h = self.monarch_r2(h)
        if self.noise2 is not None:
            h = self.noise2(h)
        x = x + h
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, photon_native=True, block_index={self.block_index}"


# ---------------------------------------------------------------------------
# PhotonFlowModel (Round 3: hidden_dim=784, fixed time embed, final norm)
# ---------------------------------------------------------------------------

class PhotonFlowModel(nn.Module):
    """Full photon-native PhotonFlow vector-field network: v_theta(x_t, t).

    Every op in the forward graph maps to a published on-chip photonic
    primitive:

        Input (B, in_dim == hidden_dim, perfect square)
          -> MonarchLayer(hidden_dim)            # Cayley-unitary MZI mesh
          -> PhotonFlowBlock x num_blocks        # photon-native core
          -> DivisivePowerNorm(hidden_dim)       # microring + fixed SOA gain
          -> MonarchLayer(hidden_dim)            # Cayley-unitary MZI mesh
          -> Output (B, in_dim)

    Time embedding (photon-native):
        t (B,) in [0, 1]
          -> WavelengthCodedTime(256)                     # AWGR look-up
          -> MonarchLinear(256, time_dim)                 # Monarch on padded MZI mesh
          -> PPLNSigmoid(beta=1.0)                        # PPLN chi^2 nonlinearity
          -> MonarchLinear(time_dim, time_dim)            # Monarch MZI mesh
          -> t_emb (B, time_dim)  passed to every block

    There is NO adaLN, NO gated residual, NO final-layer scale/shift, NO
    learnable per-channel affine inside the norm.  Time conditioning is a
    single additive wavelength-coded bias per sub-layer.

    Args:
        in_dim       (int):    Input/output dimension.  Must equal hidden_dim.
        hidden_dim   (int):    Internal dimension (perfect square).  Default 784.
        num_blocks   (int):    Number of PhotonFlowBlocks.  Default 14.
        time_dim     (int):    Time embedding dim (perfect square).  Default 256.
        use_noise    (bool):   Train-time PhotonicNoise injection.  Default True.
        sigma_s      (float):  Shen 2017 σ_D shot noise.  Default 0.001.
        sigma_t      (float):  Shen 2017 σ_φ proxy.  Default 0.005.
        shot_signal_dependent (bool): Shen's physically-accurate shot noise.
                                      Default True.
        monarch_init (str):    Monarch-block init mode (default 'random').
        adaln_init_std (float): init magnitude for the `cond_bias_proj`
                                MonarchLinear inside each block.  Default 0.02.
        num_monarch_factors (int): stacked Monarch factors per layer.  Default 1.
        absorber_alpha (float): saturable-absorber α.  Default 0.8.
        absorber_leaky_slope (float): SA linear-bypass slope.  Default 0.05.
        learnable_absorber_alpha (bool): trainable α.  Default False.
        mean_center_norm (bool): subtract per-sample mean in DPN.  Default False.
        phase_noise_sigma (float): σ_φ multiplicative jitter.  Default 0.005.
        cumulative_loss_db_per_stage (float): Shen 2017 per-MZI loss.
                                              Default 0.0003.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 784,
        num_blocks: int = 14,
        time_dim: int = 256,
        *,
        use_noise: bool = True,
        sigma_s: float = 0.001,
        sigma_t: float = 0.005,
        shot_signal_dependent: bool = True,
        monarch_init: str = "random",
        adaln_init_std: float = 0.02,
        num_monarch_factors: int = 1,
        absorber_alpha: float = 0.8,
        absorber_leaky_slope: float = 0.05,
        learnable_absorber_alpha: bool = False,
        mean_center_norm: bool = False,
        phase_noise_sigma: float = 0.005,
        cumulative_loss_db_per_stage: float = 0.0003,
    ) -> None:
        super().__init__()
        # hidden_dim must be a perfect square for MonarchLayer
        m_h = math.isqrt(hidden_dim)
        if m_h * m_h != hidden_dim:
            raise ValueError(
                f"hidden_dim must be a perfect square, got {hidden_dim}. "
                "Valid: 256 (16^2), 784 (28^2), 1024 (32^2)."
            )
        if in_dim != hidden_dim:
            raise ValueError(
                f"photon-native PhotonFlow requires in_dim == hidden_dim; "
                f"got in_dim={in_dim}, hidden_dim={hidden_dim}.  "
                "Use patch tokenisation or a MonarchLinear bookend outside "
                "the model for mismatched dimensions."
            )
        m_t = math.isqrt(time_dim)
        if m_t * m_t != time_dim:
            raise ValueError(
                f"time_dim must be a perfect square, got {time_dim}.  "
                "Valid: 256 (16^2), 1024 (32^2), etc."
            )
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # --- Photon-native input/output bookends: Cayley-unitary Monarch meshes
        _bookend_kwargs = dict(init=monarch_init, num_factors=num_monarch_factors)
        self.input_proj  = MonarchLayer(hidden_dim, **_bookend_kwargs)
        self.output_proj = MonarchLayer(hidden_dim, **_bookend_kwargs)

        # --- Photon-native time embedding pipeline ---
        # WavelengthCodedTime + MonarchLinear + PPLNSigmoid + MonarchLinear.
        self.time_mlp = nn.Sequential(
            WavelengthCodedTime(256),
            MonarchLinear(256, time_dim, init_scale=0.1, bias=True),
            PPLNSigmoid(beta=1.0),
            MonarchLinear(time_dim, time_dim, init_scale=0.1, bias=True),
        )

        # --- Photonic blocks (all strictly photon-native) ---
        self.blocks = nn.ModuleList([
            PhotonFlowBlock(
                dim=hidden_dim,
                time_dim=time_dim,
                use_noise=use_noise,
                sigma_s=sigma_s,
                sigma_t=sigma_t,
                shot_signal_dependent=shot_signal_dependent,
                monarch_init=monarch_init,
                adaln_init_std=adaln_init_std,
                num_monarch_factors=num_monarch_factors,
                absorber_alpha=absorber_alpha,
                absorber_leaky_slope=absorber_leaky_slope,
                learnable_absorber_alpha=learnable_absorber_alpha,
                mean_center_norm=mean_center_norm,
                phase_noise_sigma=phase_noise_sigma,
                cumulative_loss_db_per_stage=cumulative_loss_db_per_stage,
                block_index=i,
            )
            for i in range(num_blocks)
        ])

        # --- Final divisive-power normalisation ---
        # No final adaLN, no per-channel scale/shift at the output -- the
        # Stage-3 surgery (§14 of photonflow_experiments_report.md) established
        # that those ops have no photonic equivalent.  The forward ends with
        # norm -> output_proj.  Fixed-buffer gain is still a photonic primitive
        # (pre-set SOA amplifier), and is present in DivisivePowerNorm.
        self.final_norm = DivisivePowerNorm(
            num_features=hidden_dim,
            mean_center=mean_center_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the flow field v_theta(x_t, t) = (x_1 - x_0).

        Args:
            x: (B, in_dim) noisy sample x_t at time t.
            t: (B,) float tensor of time values in [0, 1].

        Returns:
            (B, in_dim) predicted flow field.
        """
        # Photon-native input projection (Cayley-unitary MZI mesh)
        x = self.input_proj(x)

        # Photon-native time embedding (WavelengthCodedTime + Monarch MLP)
        t_emb = self.time_mlp(t)                  # (B, time_dim)

        # Photonic blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Final norm -> photon-native output projection
        x = self.final_norm(x)
        return self.output_proj(x)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"num_blocks={self.num_blocks}, photon_native=True"
        )

    def set_noise_scale(self, scale: float) -> None:
        """Set noise warmup scale on all PhotonicNoise modules.

        Args:
            scale: 0.0 = no noise, 1.0 = full noise.
                   Call this each step with scale = min(1, step / warmup_steps).
        """
        for module in self.modules():
            if hasattr(module, 'set_noise_scale') and module is not self:
                module.set_noise_scale(scale)

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

    # --- Test 2: Parameter count is O(n^{3/2}), NO BIAS (Dao Def 3.1) ---
    m = 16   # sqrt(256)
    expected_params = 2 * m ** 3  # L + R, no bias
    actual_params = sum(p.numel() for p in layer.parameters())
    assert actual_params == expected_params, (
        f"Param count {actual_params} != {expected_params} (no-bias photon-native)"
    )
    dense_params = 256 * 256
    ratio = dense_params / (2 * m ** 3)
    print(
        f"  [PASS] Test 2 - Param count: {actual_params} (L+R, no bias) "
        f"vs {dense_params} (dense) = {ratio:.1f}x fewer"
    )

    # --- Test 3: Cayley unitary forward produces norm-preserving output ---
    # Monarch is hardcoded to Cayley-unitary projection.  For a single factor,
    # each L, R block is orthogonal, so ||Mx||_2 == ||x||_2 exactly.
    layer_u = MonarchLayer(dim=256, init="random").eval()
    x3 = torch.randn(8, 256)
    out3 = layer_u(x3)
    diff_norms = (out3.norm(dim=-1) - x3.norm(dim=-1)).abs().max().item()
    assert diff_norms < 1e-4, f"Cayley-unitary should preserve L2 norm, got max|diff|={diff_norms:.2e}"
    print(f"  [PASS] Test 3 - Cayley-unitary Monarch preserves ||x||_2 (max diff {diff_norms:.2e})")

    # --- Test 4: Gradients flow through Ls, Rs (no bias to test) ---
    layer4 = MonarchLayer(dim=64)
    layer4.train()
    x4 = torch.randn(4, 64)
    loss = layer4(x4).sum()
    loss.backward()
    assert layer4.Ls[0].grad is not None, "No gradient for L"
    assert layer4.Rs[0].grad is not None, "No gradient for R"
    assert layer4.bias is None, "MonarchLayer must have no bias (Dao Def 3.1)"
    print("  [PASS] Test 4 - Gradients flow through Ls[0], Rs[0]; bias is None")

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
        lyr = MonarchLayer(dim=d)
        lyr.eval()
        xi = torch.randn(2, d)
        oi = lyr(xi)
        assert oi.shape == (2, d), f"Shape mismatch for dim={d}"
    print(f"  [PASS] Test 6 - Valid dims: {valid_dims}")

    # --- Test 7: PhotonFlowBlock photon-native forward ---
    block = PhotonFlowBlock(dim=256, time_dim=256, use_noise=False)
    block.eval()
    x7 = torch.randn(4, 256)
    t7 = torch.rand(4, 256)   # pre-computed t_emb
    out7 = block(x7, t7)
    assert out7.shape == (4, 256), f"Block output shape: {out7.shape}"
    assert not torch.isnan(out7).any(), "NaN in block output"
    assert not torch.isinf(out7).any(), "Inf in block output"
    # No adaLN so block is NOT identity at init -- but it must be bounded.
    assert out7.abs().max().item() < 200.0, "Block output blew up"
    # Verify photon-native attributes
    assert hasattr(block, 'cond_bias_proj'), "Missing cond_bias_proj"
    assert not hasattr(block, 'adaLN_proj') or block.__dict__.get('adaLN_proj') is None, (
        "adaLN_proj should not exist in photon-native block"
    )
    assert hasattr(block, 'monarch_l2'), "Missing monarch_l2"
    assert hasattr(block, 'absorber2'), "Missing absorber2"
    print(f"  [PASS] Test 7 - PhotonFlowBlock(256) photon-native: shape OK, max|out|={out7.abs().max():.2f}")

    # --- Test 7b: Module-tree audit -- no nn.Linear, no nn.SiLU ---
    n_lin  = sum(1 for mod in block.modules() if isinstance(mod, nn.Linear))
    n_silu = sum(1 for mod in block.modules() if isinstance(mod, nn.SiLU))
    assert n_lin == 0, f"PhotonFlowBlock contains {n_lin} nn.Linear modules"
    assert n_silu == 0, f"PhotonFlowBlock contains {n_silu} nn.SiLU modules"
    print(f"  [PASS] Test 7b - Photon-native block: 0 nn.Linear, 0 nn.SiLU")

    # --- Test 8: PhotonFlowModel full forward (MNIST: hidden_dim=784) ---
    model = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=6, use_noise=False)
    model.eval()
    x8 = torch.randn(4, 784)
    t8 = torch.rand(4)
    out8 = model(x8, t8)
    assert out8.shape == (4, 784), f"Model output shape: {out8.shape}"
    assert not torch.isnan(out8).any(), "NaN in model output"
    assert not torch.isinf(out8).any(), "Inf in model output"
    # Gradient check: all params get gradient from step 0 (no adaLN-Zero
    # stall; photon-native model has no zero-init bottleneck).
    model.train()
    out8_train = model(x8, t8)
    out8_train.sum().backward()
    n_params = model.count_parameters()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().max() > 0)
    n_total_p = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total_p, (
        f"Only {n_with_grad}/{n_total_p} parameters have gradients"
    )
    # Module-tree audit: zero nn.Linear, zero nn.SiLU in the whole model
    n_lin_m  = sum(1 for mod in model.modules() if isinstance(mod, nn.Linear))
    n_silu_m = sum(1 for mod in model.modules() if isinstance(mod, nn.SiLU))
    assert n_lin_m == 0 and n_silu_m == 0, (
        f"Model contains {n_lin_m} nn.Linear + {n_silu_m} nn.SiLU"
    )
    print(
        f"  [PASS] Test 8 - PhotonFlowModel(784,784,6) photon-native: "
        f"(4,784)->(4,784), {n_params:,} params, 0 electronic ops"
    )

    # --- Test 9: Noise toggle -- deterministic in eval, stochastic in train ---
    model_noise = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=2, use_noise=True)
    x9 = torch.randn(4, 784)
    t9 = torch.rand(4)
    # Eval mode: deterministic
    model_noise.eval()
    out9a = model_noise(x9, t9)
    out9b = model_noise(x9, t9)
    assert torch.equal(out9a, out9b), "Eval mode should be deterministic"
    # Train mode: stochastic (PhotonicNoise fires on every forward)
    model_noise.train()
    out9c = model_noise(x9, t9)
    out9d = model_noise(x9, t9)
    assert not torch.equal(out9c, out9d), "Train mode with noise should be stochastic"
    print("  [PASS] Test 9 - Noise toggle: eval=deterministic, train=stochastic")

    print()
    print("All 9 tests passed.")
    print()
    print("Architecture summary (photon-native):")
    model_summary = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=6)
    print(f"  MonarchLayer(784): m=28, {sum(p.numel() for p in MonarchLayer(784).parameters()):,} params "
          f"(vs {784*784:,} dense = {784*784 / (2*28**3):.1f}x fewer), unitary-only")
    print(f"  PhotonFlowModel(784, 784, 6): {model_summary.count_parameters():,} total params")
    print(f"  {model_summary}")
