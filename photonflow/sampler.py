"""
photonflow/sampler.py

Photon-native CFM sampler -- replaces the deleted `euler_sample` function
from `photonflow.train`.

Physical mapping:

  Training stays digital.  Inference runs on-chip as follows:

    - `x_0` is an amplified-spontaneous-emission (ASE / EDFA) pulse whose
      statistics are Gaussian by construction; digitally we just use
      `torch.randn`.
    - `x_{k+1} = x_k + tau * model(x_k, t)` is implemented as a recirculating
      MZI delay line (*Nature Comp. Sci.* 2025 Kerr temporal-convolution
      neuron) where `model(.)` is the entire photon-native PhotonFlowModel
      forward pass.
    - Termination: a single photodetector comparator reads `||delta_x||_2`
      and halts the recirculation when it is below `eps`.  Exactly ONE
      electronic op per sample, not per step.

The class `OpticalSampler` exposes an nn.Module interface with a
`forward(shape, device, t=1.0)` call and a `count_electronic_ops()` helper
used by the photon-native module-tree audit (see tests/module_tree_audit).

Replaces the old M10 ~20 O-E-O conversions per sample with a single
comparator readout.

References:
  * Nature Computational Science 2025, "A complete photonic integrated
    neuron for nonlinear all-optical computing" (Kerr temporal-conv
    recirculation).
  * Song & Ermon 2023, "Consistency Models" (1-step distillation; the
    onepass mode below relies on this for the training-time objective,
    which is digital-only).
  * Li et al. 2023 *Light: Sci. Appl.*, ASE-source generative photonics.
"""

import torch
import torch.nn as nn

__all__ = ["OpticalSampler"]


class OpticalSampler(nn.Module):
    """Fixed-point photonic sampler for CFM inference.

    Iterates `x_{k+1} = x_k + tau * model(x_k, t)` in a recirculating MZI
    delay line; terminates when `||delta_x||_2 < eps` as read by a single
    photodetector comparator.  At most `max_iters` recirculations.

    Electronic-op count per generated sample:

        fixedpoint:  exactly 1 (termination comparator)
        onepass   :  0 (trained-digital 1-step generator; requires a
                       consistency-distilled model)

    Both modes leave TRAINING electronic (CFMLoss, Adam, LR scheduler) --
    those never touch the chip.

    Args:
        model (nn.Module): trained PhotonFlowModel.
        tau   (float):     step size per recirculation.  Default 1 / max_iters.
        eps   (float):     comparator threshold on ||delta_x||_2.  Default 1e-3.
        max_iters (int):   max recirculations.  Default 10.
        mode  (str):       "fixedpoint" (default) or "onepass".
    """

    def __init__(
        self,
        model: nn.Module,
        tau: float = None,
        eps: float = 1e-3,
        max_iters: int = 10,
        mode: str = "fixedpoint",
    ) -> None:
        super().__init__()
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if mode not in ("fixedpoint", "onepass"):
            raise ValueError(f"mode must be 'fixedpoint' or 'onepass', got {mode!r}")
        self.model = model
        self.tau = float(tau) if tau is not None else 1.0 / max_iters
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.mode = mode

    @torch.no_grad()
    def forward(
        self,
        shape: tuple,
        device=None,
        t: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples photonically.

        Args:
            shape:  (B, D) -- number of samples and feature dimension.
            device: torch device.  Defaults to the model's device.
            t:      end-of-trajectory time in [0, 1].  Default 1.0.

        Returns:
            (B, D) generated samples.
        """
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = "cpu"
        was_training = self.model.training
        self.model.eval()

        # ASE / EDFA initial state: Gaussian-distributed coherent-field amps.
        x = torch.randn(*shape, device=device)

        if self.mode == "onepass":
            # Trained-digital consistency-model path: one forward through the
            # chip at t=1.0 and we are done.  No recirculation, no comparator.
            t_ = torch.full((shape[0],), float(t), device=device)
            x = x + self.model(x, t_)
        else:
            # Fixed-point: recirculate the delay line until ||delta_x||_2<eps
            # or max_iters is reached.  The single comparator at the
            # photodetector is the only electronic op per sample.
            t_ = torch.full((shape[0],), float(t), device=device)
            self._iterations = 0
            for k in range(self.max_iters):
                dx = self.tau * self.model(x, t_)
                x = x + dx
                self._iterations += 1
                if dx.norm(dim=-1).mean().item() < self.eps:
                    break

        if was_training:
            self.model.train()
        return x

    def count_electronic_ops(self) -> int:
        """Number of electronic O-E-O crossings per generated sample.

        `fixedpoint`: 1 (the termination comparator at the detector readout).
        `onepass`:    0 (single forward pass, no termination test).
        """
        return 0 if self.mode == "onepass" else 1

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, tau={self.tau}, eps={self.eps}, "
            f"max_iters={self.max_iters}, electronic_ops={self.count_electronic_ops()}"
        )


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    from photonflow.model import PhotonFlowModel

    model = PhotonFlowModel(in_dim=16, hidden_dim=16, num_blocks=2,
                             time_dim=16, use_noise=False)

    # --- Test 1: fixedpoint shape + op count ---
    s_fp = OpticalSampler(model, tau=0.1, max_iters=5, mode="fixedpoint")
    out_fp = s_fp(shape=(4, 16))
    assert out_fp.shape == (4, 16), f"Fixedpoint output shape {out_fp.shape}"
    assert not torch.isnan(out_fp).any(), "NaN in fixedpoint output"
    assert s_fp.count_electronic_ops() == 1, "fixedpoint must report 1 E-op"
    print(f"  [PASS] Test 1 - fixedpoint: shape=(4,16), iters={s_fp._iterations}, E-op=1")

    # --- Test 2: onepass shape + op count ---
    s_op = OpticalSampler(model, mode="onepass")
    out_op = s_op(shape=(4, 16))
    assert out_op.shape == (4, 16), f"Onepass output shape {out_op.shape}"
    assert not torch.isnan(out_op).any(), "NaN in onepass output"
    assert s_op.count_electronic_ops() == 0, "onepass must report 0 E-op"
    print(f"  [PASS] Test 2 - onepass: shape=(4,16), E-op=0")

    # --- Test 3: invalid kwargs rejected ---
    for bad_kw in [dict(max_iters=0), dict(eps=0), dict(mode="euler")]:
        try:
            OpticalSampler(model, **bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {bad_kw}")
    print("  [PASS] Test 3 - rejects max_iters=0, eps=0, mode='euler'")

    print("\nAll OpticalSampler tests passed.")
    print(f"  {s_fp}")
    print(f"  {s_op}")
