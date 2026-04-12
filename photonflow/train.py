"""
photonflow/train.py

Training infrastructure for PhotonFlow.

Components:
    CFMLoss      — Conditional Flow Matching loss (Lipman et al. 2023)
    euler_sample — Euler ODE solver for generating samples at inference
    Trainer      — Full training pipeline: config, dataloader, loop, checkpoints

CFM loss (Lipman 2023, Eq. 23, OT path with sigma_min=0):
    L(theta) = E_{t ~ U[0,1], x0 ~ N(0,I), x1 ~ q(data)}
               [ || v_theta(x_t, t) - (x1 - x0) ||^2 ]

    where x_t = (1-t)*x0 + t*x1  (Eq. 22, OT linear interpolation)
    and the target vector field is u_t = x1 - x0 (constant in time).

    The OT path gives straight-line trajectories from noise to data
    (Figure 3 of the paper), leading to faster convergence and more
    efficient sampling vs diffusion paths.

Sampling (Section 6.2):
    1. Draw x_0 ~ N(0, I)
    2. Solve dx/dt = v_theta(x, t) from t=0 to t=1 using Euler method
    3. x_1 ≈ generated sample

References:
    Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.
    - Eq. 22: OT interpolation path psi_t(x) = (1-(1-sigma_min)t)*x0 + t*x1
    - Eq. 23: CFM loss with OT conditional VF
    - Section 6.2: sampling via ODE integration

    spec: CFMLoss — sample t, x0, x1, compute x_t, target=x1-x0, MSE loss.
          Trainer — config YAML load, dataloader, Adam lr=1e-4, training loop,
                    noise toggle, QAT toggle, checkpoint every 5K, sample grid
                    every 5K (Euler ODE, 20 steps).
"""

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from photonflow.model import PhotonFlowModel

__all__ = ["CFMLoss", "euler_sample", "Trainer"]


# ---------------------------------------------------------------------------
# CFMLoss — Conditional Flow Matching (Lipman 2023, Eq. 23)
# ---------------------------------------------------------------------------

class CFMLoss(nn.Module):
    """Conditional Flow Matching loss with Optimal Transport paths.

    Lipman et al. 2023, Eq. 22-23 (with sigma_min = 0):

        x_t    = (1-t) * x0  +  t * x1          (OT interpolation)
        target = x1 - x0                         (constant VF)
        loss   = || v_theta(x_t, t) - target ||^2   (MSE)

    where x0 ~ N(0, I) is noise and x1 is data.

    The OT path produces straight-line trajectories from noise to data,
    which are easier to learn (constant direction in time) and allow
    efficient sampling with few ODE steps (Figure 4 of the paper).

    Args:
        sigma_min (float): Minimum std for the OT path.  Default 0.0.
            With sigma_min=0, the interpolation simplifies to the
            standard linear path x_t = (1-t)*x0 + t*x1.
            Lipman 2023 Eq. 22 uses sigma_min for numerical stability;
            in practice sigma_min=0 works fine for image generation.
    """

    def __init__(self, sigma_min: float = 0.0) -> None:
        super().__init__()
        self.sigma_min = sigma_min

    def forward(
        self,
        model: nn.Module,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CFM loss for a batch of data.

        Args:
            model: v_theta network (PhotonFlowModel or QATWrapper).
                   Must accept (x, t) and return (B, D).
            x1:    (B, D) batch of real data samples.

        Returns:
            Scalar MSE loss.
        """
        B, D = x1.shape
        device = x1.device

        # --- Sample noise x0 ~ N(0, I) ---
        x0 = torch.randn_like(x1)

        # --- Sample time t ~ U[0, 1] ---
        t = torch.rand(B, device=device)

        # --- OT interpolation: x_t = (1 - (1-sigma_min)*t) * x0 + t * x1 ---
        # With sigma_min=0: x_t = (1-t)*x0 + t*x1
        t_expand = t[:, None]  # (B, 1) for broadcasting
        if self.sigma_min == 0.0:
            x_t = (1.0 - t_expand) * x0 + t_expand * x1
        else:
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x0 + t_expand * x1

        # --- Target vector field: x1 - (1-sigma_min)*x0 ---
        # With sigma_min=0: target = x1 - x0
        if self.sigma_min == 0.0:
            target = x1 - x0
        else:
            target = x1 - (1.0 - self.sigma_min) * x0

        # --- Model prediction ---
        v_pred = model(x_t, t)

        # --- MSE loss ---
        loss = F.mse_loss(v_pred, target)
        return loss

    def extra_repr(self) -> str:
        return f"sigma_min={self.sigma_min}"


# ---------------------------------------------------------------------------
# Euler ODE sampler (Section 6.2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model: nn.Module,
    shape: tuple,
    num_steps: int = 20,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate samples using Euler ODE integration.

    Integrates  dx/dt = v_theta(x, t)  from t=0 to t=1.

    Lipman 2023, Section 6.2:
        "draw a random noise sample x_0 ~ N(0,I) then compute phi_1(x_0)
         by solving equation 1 with the trained VF v_t on [0,1]."

    The paper uses adaptive dopri5 solver; we use fixed-step Euler for
    simplicity and photonic hardware compatibility (fixed latency per step).
    With OT paths, Euler gives decent quality even at NFE=20 (Figure 4).

    Args:
        model:     Trained PhotonFlowModel. Switched to eval mode internally.
        shape:     (B, D) — number of samples and dimension.
        num_steps: Number of Euler steps (= NFE). Default 20.
        device:    Torch device.

    Returns:
        (B, D) generated samples.
    """
    was_training = model.training
    model.eval()

    # Start from noise x_0 ~ N(0, I)
    x = torch.randn(shape, device=device)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((shape[0],), t_val, device=device)
        v = model(x, t)
        x = x + dt * v

    # Restore training state
    if was_training:
        model.train()

    return x


# ---------------------------------------------------------------------------
# Trainer — full training pipeline
# ---------------------------------------------------------------------------

class Trainer:
    """PhotonFlow training pipeline.

    Handles:
        - Config dict / YAML loading
        - Dataset setup (MNIST / CIFAR-10 / synthetic)
        - Training loop with CFM loss
        - Noise injection toggle (via model.use_noise config)
        - QAT fine-tuning toggle (wraps model with QATWrapper)
        - Checkpoint saving every N steps
        - Sample grid generation every N steps (Euler ODE)

    Config dict structure::

        model:
            in_dim: 784
            hidden_dim: 256
            num_blocks: 6
            use_noise: true
            sigma_s: 0.02
            sigma_t: 0.01
        data:
            dataset: mnist        # mnist | cifar10
            batch_size: 128
            root: ./data
            num_workers: 2
        training:
            lr: 1e-4
            total_steps: 50000
            checkpoint_every: 5000
            sample_every: 5000
            sample_steps: 20      # Euler ODE steps for generation
            seed: 42
        qat:
            enabled: false
            bits: 4
            checkpoint: null      # path to pre-trained checkpoint to load
        output_dir: outputs

    Args:
        config: dict with the above structure, or path to YAML file.
    """

    def __init__(self, config) -> None:
        if isinstance(config, (str, Path)):
            import yaml
            with open(config, "r") as f:
                config = yaml.safe_load(f)
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Seed
        seed = config.get("training", {}).get("seed", 42)
        torch.manual_seed(seed)

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Load checkpoint if specified (for QAT fine-tuning stage)
        qat_cfg = config.get("qat", {})
        ckpt_path = qat_cfg.get("checkpoint", None)
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])

        # QAT wrapper (if enabled)
        self.qat_enabled = qat_cfg.get("enabled", False)
        if self.qat_enabled:
            from hardware.qat import QATWrapper
            bits = qat_cfg.get("bits", 4)
            self.model = QATWrapper(self.model, bits=bits)

        # Optimizer — Adam lr=1e-4 (CLAUDE.md spec)
        lr = config.get("training", {}).get("lr", 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Loss
        self.criterion = CFMLoss(sigma_min=0.0)

        # Training params
        tcfg = config.get("training", {})
        self.total_steps = tcfg.get("total_steps", 50000)
        self.checkpoint_every = tcfg.get("checkpoint_every", 5000)
        self.sample_every = tcfg.get("sample_every", 5000)
        self.sample_steps = tcfg.get("sample_steps", 20)

        # Output dirs
        self.output_dir = config.get("output_dir", "outputs")
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.fig_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        # Logging state
        self.losses: list = []
        self.global_step: int = 0

    # ---- Model construction ----

    def _build_model(self) -> PhotonFlowModel:
        mcfg = self.config.get("model", {})
        return PhotonFlowModel(
            in_dim=mcfg.get("in_dim", 784),
            hidden_dim=mcfg.get("hidden_dim", 256),
            num_blocks=mcfg.get("num_blocks", 6),
            use_noise=mcfg.get("use_noise", True),
            sigma_s=mcfg.get("sigma_s", 0.02),
            sigma_t=mcfg.get("sigma_t", 0.01),
        )

    # ---- Dataset construction ----

    def _build_dataloader(self) -> DataLoader:
        dcfg = self.config.get("data", {})
        name = dcfg.get("dataset", "mnist").lower()
        bs = dcfg.get("batch_size", 128)
        root = dcfg.get("root", "./data")
        nw = dcfg.get("num_workers", 0)

        import torchvision
        import torchvision.transforms as T

        if name == "mnist":
            transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
            dataset = torchvision.datasets.MNIST(
                root=root, train=True, download=True, transform=transform,
            )
        elif name == "cifar10":
            transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
            dataset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}. Use 'mnist' or 'cifar10'.")

        return DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=True, drop_last=True,
        )

    # ---- Training loop ----

    def train(self, dataloader: DataLoader = None) -> list:
        """Run the full training loop.

        Args:
            dataloader: Optional custom DataLoader. If None, builds from config.

        Returns:
            List of per-step loss values.
        """
        if dataloader is None:
            dataloader = self._build_dataloader()

        self.model.train()
        data_iter = iter(dataloader)

        for step in range(self.global_step, self.total_steps):
            # Cycle through dataloader
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Unpack (x, label) or just x
            x1 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x1 = x1.to(self.device)

            # Forward + backward
            loss = self.criterion(self.model, x1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log
            loss_val = loss.item()
            self.losses.append(loss_val)
            self.global_step = step + 1

            # Print progress every 100 steps
            if self.global_step % 100 == 0:
                avg = sum(self.losses[-100:]) / min(100, len(self.losses))
                print(
                    f"  step {self.global_step:>6d}/{self.total_steps} | "
                    f"loss {loss_val:.4f} | avg100 {avg:.4f}"
                )

            # Checkpoint
            if self.global_step % self.checkpoint_every == 0:
                self.save_checkpoint()

            # Sample
            if self.global_step % self.sample_every == 0:
                self.generate_samples()

        # Final checkpoint
        self.save_checkpoint()
        return self.losses

    # ---- Checkpointing ----

    def save_checkpoint(self) -> str:
        """Save model, optimizer, losses, and config to disk.

        Returns:
            Path to the saved checkpoint file.
        """
        from hardware.qat import QATWrapper
        model_to_save = (
            self.model.model if isinstance(self.model, QATWrapper)
            else self.model
        )
        path = os.path.join(
            self.ckpt_dir, f"checkpoint_step{self.global_step}.pth"
        )
        torch.save(
            {
                "step": self.global_step,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "losses": self.losses,
                "config": self.config,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from disk.

        Args:
            path: Path to checkpoint .pth file.
        """
        from hardware.qat import QATWrapper
        state = torch.load(path, map_location=self.device)
        model_to_load = (
            self.model.model if isinstance(self.model, QATWrapper)
            else self.model
        )
        model_to_load.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.losses = state.get("losses", [])
        self.global_step = state.get("step", 0)

    # ---- Sample generation ----

    @torch.no_grad()
    def generate_samples(self, n_samples: int = 64) -> torch.Tensor:
        """Generate and save a grid of samples using Euler ODE.

        Args:
            n_samples: Number of samples to generate. Default 64 (8×8 grid).

        Returns:
            (n_samples, D) generated samples tensor.
        """
        from hardware.qat import QATWrapper
        model_for_sample = (
            self.model.model if isinstance(self.model, QATWrapper)
            else self.model
        )
        in_dim = model_for_sample.in_dim

        samples = euler_sample(
            model_for_sample,
            shape=(n_samples, in_dim),
            num_steps=self.sample_steps,
            device=self.device,
        )

        # Clamp to [0, 1] for visualization
        samples_viz = samples.clamp(0, 1)

        # Save grid image
        if in_dim == 784:     # MNIST: 1×28×28
            img_h, img_w, ch = 28, 28, 1
        elif in_dim == 3072:  # CIFAR-10: 3×32×32
            img_h, img_w, ch = 32, 32, 3
        else:
            return samples    # Can't make a grid for unknown dims

        try:
            import torchvision
            grid_n = int(math.sqrt(n_samples))
            imgs = samples_viz[: grid_n ** 2].view(-1, ch, img_h, img_w)
            grid = torchvision.utils.make_grid(imgs, nrow=grid_n, padding=2)
            path = os.path.join(
                self.fig_dir, f"samples_step{self.global_step}.png"
            )
            torchvision.utils.save_image(grid, path)
        except Exception:
            pass  # Silently skip if torchvision grid fails

        return samples


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    torch.manual_seed(42)
    print("Testing photonflow/train.py ...\n")

    # --- Test 1: CFMLoss — shape, positive, no NaN ---
    from photonflow.model import PhotonFlowModel

    model = PhotonFlowModel(in_dim=64, hidden_dim=16, num_blocks=2, use_noise=False)
    model.train()

    criterion = CFMLoss(sigma_min=0.0)
    x1 = torch.randn(8, 64)  # fake batch
    loss = criterion(model, x1)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    print(f"  [PASS] Test 1 - CFMLoss: scalar={loss.item():.4f}, positive, no NaN")

    # --- Test 2: CFMLoss gradients flow ---
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Params with no gradient: {no_grad}"
    print(f"  [PASS] Test 2 - CFMLoss gradients flow through all {sum(p.numel() for p in model.parameters()):,} params")

    # --- Test 3: euler_sample — shape, no NaN ---
    model.eval()
    samples = euler_sample(model, shape=(4, 64), num_steps=10, device="cpu")
    assert samples.shape == (4, 64), f"Sample shape: {samples.shape}"
    assert not torch.isnan(samples).any(), "NaN in samples"
    assert not torch.isinf(samples).any(), "Inf in samples"
    print(f"  [PASS] Test 3 - euler_sample: shape=(4,64), 10 steps, no NaN/Inf")

    # --- Test 4: CFMLoss with sigma_min > 0 ---
    criterion_sm = CFMLoss(sigma_min=1e-4)
    model.train()
    loss_sm = criterion_sm(model, x1)
    assert loss_sm.item() > 0, "sigma_min loss should be positive"
    assert not torch.isnan(loss_sm), "sigma_min loss is NaN"
    print(f"  [PASS] Test 4 - CFMLoss(sigma_min=1e-4): loss={loss_sm.item():.4f}")

    # --- Test 5: Trainer with synthetic data — 100 steps, loss decreases ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "model": {
                "in_dim": 64,
                "hidden_dim": 16,
                "num_blocks": 2,
                "use_noise": False,
            },
            "training": {
                "lr": 5e-3,   # Higher lr for fast convergence in test
                "total_steps": 200,
                "checkpoint_every": 100,
                "sample_every": 200,
                "sample_steps": 5,
                "seed": 42,
            },
            "qat": {"enabled": False},
            "output_dir": tmpdir,
        }
        trainer = Trainer(config)

        # Synthetic dataset: random vectors (dim=64)
        synth_x = torch.randn(256, 64)
        synth_ds = TensorDataset(synth_x)
        synth_dl = DataLoader(synth_ds, batch_size=32, shuffle=True)

        print("\n  Training 200 steps on synthetic data (dim=64, 2 blocks)...")
        losses = trainer.train(dataloader=synth_dl)

        # Check loss decreased (compare first 20 vs last 20 for robustness)
        first_20 = sum(losses[:20]) / 20
        last_20 = sum(losses[-20:]) / 20
        assert last_20 < first_20, (
            f"Loss did not decrease: first_20={first_20:.4f}, last_20={last_20:.4f}"
        )
        print(f"  [PASS] Test 5 - Loss decreased: {first_20:.4f} -> {last_20:.4f}")

        # Check no NaN in losses
        assert all(not math.isnan(l) for l in losses), "NaN in loss history"
        print(f"  [PASS] Test 5b - No NaN in {len(losses)} loss values")

        # Check checkpoint saved
        ckpt_files = list(Path(tmpdir, "checkpoints").glob("*.pth"))
        assert len(ckpt_files) >= 1, f"No checkpoints saved (found {len(ckpt_files)})"
        print(f"  [PASS] Test 5c - Checkpoints saved: {[f.name for f in ckpt_files]}")

    # --- Test 6: Trainer with QAT enabled ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config_qat = {
            "model": {
                "in_dim": 64,
                "hidden_dim": 16,
                "num_blocks": 2,
                "use_noise": False,
            },
            "training": {
                "lr": 1e-3,
                "total_steps": 20,
                "checkpoint_every": 20,
                "sample_every": 100,
                "seed": 42,
            },
            "qat": {"enabled": True, "bits": 4},
            "output_dir": tmpdir,
        }
        trainer_qat = Trainer(config_qat)

        synth_dl2 = DataLoader(TensorDataset(torch.randn(64, 64)),
                               batch_size=16, shuffle=True)
        losses_qat = trainer_qat.train(dataloader=synth_dl2)
        assert len(losses_qat) == 20, f"Expected 20 losses, got {len(losses_qat)}"
        assert all(not math.isnan(l) for l in losses_qat), "NaN in QAT losses"
        print(f"  [PASS] Test 6 - QAT training: 20 steps, no NaN, "
              f"loss {losses_qat[0]:.4f} -> {losses_qat[-1]:.4f}")

    # --- Test 7: Checkpoint load/save round-trip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config_rt = {
            "model": {"in_dim": 64, "hidden_dim": 16, "num_blocks": 2, "use_noise": False},
            "training": {"lr": 1e-3, "total_steps": 10, "checkpoint_every": 10,
                         "sample_every": 100, "seed": 42},
            "qat": {"enabled": False},
            "output_dir": tmpdir,
        }
        t1 = Trainer(config_rt)
        dl = DataLoader(TensorDataset(torch.randn(32, 64)), batch_size=8, shuffle=True)
        t1.train(dataloader=dl)
        ckpt = t1.save_checkpoint()

        # Load into new trainer
        t2 = Trainer(config_rt)
        t2.load_checkpoint(ckpt)
        assert t2.global_step == t1.global_step, (
            f"Step mismatch: {t2.global_step} vs {t1.global_step}"
        )
        # Verify model weights match
        for (n1, p1), (n2, p2) in zip(
            t1.model.named_parameters(), t2.model.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6), f"Weight mismatch: {n1}"
        print(f"  [PASS] Test 7 - Checkpoint round-trip: step={t2.global_step}, weights match")

    print("\nAll 7 tests passed.")
    print()
    print("Training pipeline summary:")
    print("  CFMLoss:      L(th) = E[||v_th(xt,t) - (x1-x0)||^2] (Lipman 2023, Eq. 23)")
    print("  Interpolation: xt = (1-t)*x0 + t*x1                 (OT path, Eq. 22)")
    print("  Sampling:      Euler ODE, dx/dt = v_th(x,t), 20 steps (Section 6.2)")
    print("  Optimizer:     Adam lr=1e-4                          (CLAUDE.md spec)")
    print("  Checkpoints:   every 5K steps                        (model + optimizer + losses)")
    print("  Sample grids:  every 5K steps                        (8x8 Euler-generated images)")
