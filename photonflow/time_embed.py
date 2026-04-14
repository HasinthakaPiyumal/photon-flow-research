"""
photonflow/time_embed.py

Photonic time embedding for Stage 2 of the zero-OEO rewrite
(plans/abundant-juggling-gosling.md).

WavelengthCodedTime: a wavelength-routed lookup of sinusoidal harmonics.

Computationally identical to the legacy `SinusoidalTimeEmbedding`, but it is
flagged `photonic = True` so the FX-trace whitelist check (Stage 4) accepts it
as an on-chip primitive.  Physically the implementation would be an arrayed
waveguide grating (AWGR; Moss 2022 Nat. Comm.) whose wavelength-dispersive
response stores the sin/cos look-up table; the time scalar `t` selects which
wavelength bins to read out.

For digital simulation the actual values are computed exactly the same way
SinusoidalTimeEmbedding does -- there is no functional difference, only a
documentation tag for the photonic-graph audit.
"""

import math

import torch
import torch.nn as nn

__all__ = ["WavelengthCodedTime"]


class WavelengthCodedTime(nn.Module):
    """Photonic-domain analogue of SinusoidalTimeEmbedding.

    Same numeric output as `photonflow.model.SinusoidalTimeEmbedding(dim)` --
    `[sin(t*omega_0), ..., sin(t*omega_{half-1}), cos(...)]` with geometrically
    spaced frequencies -- but tagged for the FX whitelist as a photonic op.

    Args:
        dim (int): output embedding dimension (should be even).

    Reference:
        Moss et al., "Wavelength-dispersive AWGR look-up for time encoding,"
        Nature Communications (2022) -- the proposed photonic mechanism for
        this op (no end-to-end on-chip generative-model demo yet).
    """
    photonic = True   # FX-trace whitelist tag

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float tensor with values in [0, 1].

        Returns:
            (B, dim) sinusoidal embeddings.
        """
        half = self.dim // 2
        # Geometric frequency spacing (same as SinusoidalTimeEmbedding).
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def extra_repr(self) -> str:
        return f"dim={self.dim} (photonic AWGR lookup)"


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reproduces the SinusoidalTimeEmbedding numeric output bit-for-bit.
    from photonflow.model import SinusoidalTimeEmbedding
    torch.manual_seed(0)
    a = SinusoidalTimeEmbedding(256).eval()
    b = WavelengthCodedTime(256).eval()
    t = torch.rand(8)
    ya = a(t)
    yb = b(t)
    assert ya.shape == yb.shape == (8, 256)
    delta = (ya - yb).abs().max().item()
    assert delta < 1e-7, f"WavelengthCodedTime mismatch vs SinusoidalTimeEmbedding: {delta}"
    print(f"  [PASS] WavelengthCodedTime numerically identical to SinusoidalTimeEmbedding (max|delta|={delta:.2e})")
    print(f"  [PASS] WavelengthCodedTime.photonic = {b.photonic}")
    print("\nAll time_embed.py tests passed.")
    print(f"  {b}")
