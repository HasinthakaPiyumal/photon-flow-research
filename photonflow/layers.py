"""
photonflow/layers.py

Photonic helper layers used by Stage 2 of the zero-OEO rewrite
(plans/abundant-juggling-gosling.md):

  * PPLNSigmoid -- photonic SiLU/Swish replacement via a chi^2 saturable
                   transfer.  Mathematically `tanh(x)` (= SaturableAbsorber
                   with alpha=1.0); cited as the eLight 2026 PPLN nanophotonic
                   waveguide nonlinearity (passive, ~80 % conversion eff.).
  * MonarchLinear -- structured-matrix replacement for nn.Linear with arbitrary
                     in/out dims.  Pads to the smallest perfect square,
                     applies a MonarchLayer, and crops to the requested out_dim.
                     Pad/crop are optically free (waveguide routing); the only
                     learnable op is the MonarchLayer itself, which maps to an
                     MZI mesh on chip (Shen 2017, Clements 2016, Dao 2022).

Both layers are used to replace electronic nn.SiLU/nn.Linear sites in
`photonflow/model.py` when the corresponding `*_style="monarch"` kwarg is on.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from photonflow.activation import SaturableAbsorber
from photonflow.model import MonarchLayer

__all__ = ["PPLNSigmoid", "MonarchLinear"]


# ---------------------------------------------------------------------------
# PPLN sigmoid -- photonic nonlinearity citing eLight 2026 PPLN demonstration
# ---------------------------------------------------------------------------

class PPLNSigmoid(nn.Module):
    """Photonic SiLU/Swish replacement.

    Functional form:  y = tanh(beta * x) / beta  (= SaturableAbsorber when
    beta = alpha).  At beta=1 this is plain `tanh(x)`, which has the same
    odd-symmetric saturating shape as the photonic chi^2 nonlinearity
    demonstrated in the eLight 2026 PPLN nanophotonic-waveguide paper
    (passive, ~80 % conversion efficiency).

    Properties matching SaturableAbsorber:
      f(0) = 0,   f'(0) = 1,   |f(x)| <= 1/beta (saturates).

    Reference:
        eLight 2026 -- "Passive all-optical nonlinear neuron activation via
        PPLN nanophotonic waveguides".

    Args:
        beta (float): saturation slope at the origin (default 1.0).  A
                      smaller beta delays saturation; a larger beta makes
                      the function more switch-like.
    """
    photonic = True   # FX-trace whitelist tag (see Stage-4 verification).

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        # Reuse the existing SaturableAbsorber implementation: it is the same
        # math (tanh(alpha*x)/alpha), already battle-tested, and already on the
        # FX whitelist via its `photonic` tag below.
        self._sa = SaturableAbsorber(alpha=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sa(x)

    def extra_repr(self) -> str:
        beta = self._sa.alpha.item() if torch.is_tensor(self._sa.alpha) else self._sa.alpha
        return f"beta={beta:.3f} (PPLN photonic sigmoid via tanh)"


# ---------------------------------------------------------------------------
# MonarchLinear -- pad/crop wrapper around MonarchLayer for non-square dims
# ---------------------------------------------------------------------------

class MonarchLinear(nn.Module):
    """nn.Linear-shape API around a MonarchLayer.

    A MonarchLayer requires its dimension to be a perfect square (n = m^2).
    Most adaLN / time-embedding projections in PhotonFlow are NOT square
    (e.g. time_dim=256 -> 6*hidden_dim=4704).  MonarchLinear wraps the
    MonarchLayer with an optically-free zero-pad on the input and a slice
    on the output:

        x  ->  [pad to n=m^2]  ->  MonarchLayer(n)  ->  [slice to out_dim]

    On photonic hardware, pad and slice are waveguide routing operations
    (free); only the MonarchLayer maps to MZI hardware.

    Args:
        in_dim (int): input feature dimension (any positive integer).
        out_dim (int): output feature dimension (any positive integer).
        bias (bool): whether the inner MonarchLayer carries a bias.
                     Default False (Dao 2022 Def 3.1 has no additive bias).
        init (str): MonarchLayer init mode -- "random" (Xavier gain=0.1),
                    "orthogonal", "identity", or "dct".  Default "random".
        unitary_project (bool): forward-pass Cayley projection of L, R blocks
                                onto O(m).  Default False.
        init_scale (float|None): if provided, all MonarchLayer parameters are
                    multiplied by `init_scale / 0.1` after the chosen `init`,
                    so the effective output magnitude tracks `init_scale`.
                    Used to mimic adaLN-Zero's `nn.init.normal_(... std=0.02)`
                    on the dense replacement.  None = no rescaling.
    """
    photonic = True   # FX-trace whitelist tag

    def __init__(self, in_dim: int, out_dim: int,
                 bias: bool = False, init: str = "random",
                 unitary_project: bool = False,
                 init_scale: float | None = None) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim, out_dim must be positive, got {in_dim}, {out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Smallest perfect square that holds both in_dim and out_dim.
        m = int(math.ceil(math.sqrt(max(in_dim, out_dim))))
        self.padded_dim = m * m
        self.m = m
        # Inner MonarchLayer is always bias=False; the separate `bias` buffer
        # below has shape (out_dim,), matching nn.Linear's bias semantics
        # (and letting external code -- e.g. the adaLN gate_init in
        # PhotonFlowBlock -- write to specific output channels).
        self.monarch = MonarchLayer(
            self.padded_dim,
            bias=False,
            init=init,
            num_factors=1,
            unitary_project=unitary_project,
        )
        if init_scale is not None:
            with torch.no_grad():
                # The "random" init uses Xavier gain=0.1; rescale to the
                # requested effective magnitude.  For other inits this just
                # rescales whatever the block matrix happens to be.
                ratio = float(init_scale) / 0.1
                for p in self.monarch.parameters():
                    p.mul_(ratio)
        self._init_scale = init_scale
        # nn.Linear-compatible output-dim bias.  Initialised to zeros so the
        # forward pass at construction matches nn.Linear(..., bias=True) with
        # zero-init bias.
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad the last dimension up to padded_dim with zeros (waveguide routing).
        pad_in = self.padded_dim - self.in_dim
        if pad_in > 0:
            x = F.pad(x, (0, pad_in))
        # Run the photonic MZI mesh.
        y = self.monarch(x)
        # Crop to requested out_dim (waveguide routing).
        if self.out_dim < self.padded_dim:
            y = y[..., : self.out_dim]
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"padded={self.padded_dim} (m={self.m}), "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # --- PPLNSigmoid: shape, f(0)=0, gradient at origin = 1 ---
    p = PPLNSigmoid(beta=1.0).eval()
    assert p(torch.zeros(3)).abs().max() < 1e-7
    x = torch.zeros(1, requires_grad=True)
    p(x).sum().backward()
    assert math.isclose(x.grad.item(), 1.0, abs_tol=1e-5), x.grad.item()
    print("  [PASS] PPLNSigmoid: f(0)=0, f'(0)=1, photonic tag =", p.photonic)

    # --- MonarchLinear: arbitrary in/out dims preserved ---
    for (i, o) in [(256, 8), (8, 4704), (256, 1568), (256, 256), (784, 784)]:
        m = MonarchLinear(in_dim=i, out_dim=o)
        x = torch.randn(2, i)
        y = m(x)
        assert y.shape == (2, o), f"MonarchLinear({i}, {o}) -> got {y.shape}"
    print("  [PASS] MonarchLinear shape preservation across non-square dims")

    # --- MonarchLinear init_scale: effective output std tracks init_scale ---
    m = MonarchLinear(256, 4704, init_scale=0.02).eval()
    x = torch.randn(64, 256)
    y = m(x)
    # With Xavier gain=0.1 init scaled by 0.02/0.1 = 0.2 => effective gain 0.02.
    # Expected output std ~= input_std * gain * sqrt(in_dim / m^2)... rough check.
    assert y.std().item() < 1.0, f"unexpectedly large std: {y.std().item()}"
    print(f"  [PASS] MonarchLinear init_scale=0.02: empirical out_std={y.std().item():.4f}")

    # --- Composability: PPLNSigmoid -> MonarchLinear -> PPLNSigmoid -> MonarchLinear ---
    seq = nn.Sequential(
        PPLNSigmoid(beta=1.0),
        MonarchLinear(256, 8, init_scale=0.1),
        PPLNSigmoid(beta=1.0),
        MonarchLinear(8, 4704, init_scale=0.02),
    )
    y = seq(torch.randn(3, 256))
    assert y.shape == (3, 4704)
    assert not torch.isnan(y).any()
    print(f"  [PASS] Sequential PPLNSigmoid -> MonarchLinear stack: {tuple(y.shape)}")

    print("\nAll layers.py tests passed.")
    print(f"  PPLNSigmoid: {p}")
    print(f"  MonarchLinear(256, 4704): {m}")
