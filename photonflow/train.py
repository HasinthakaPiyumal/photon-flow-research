"""
photonflow/train.py

Training infrastructure for PhotonFlow.

Components:
    CFMLoss      -- Conditional Flow Matching loss (Lipman et al. 2023)
    euler_sample -- Euler ODE solver for generating samples at inference
    Trainer      -- Full training pipeline: config, dataloader, loop, checkpoints

CFM loss (Lipman 2023, Eq. 23, OT path with sigma_min=0):
    L(theta) = E_{t ~ U[0,1], x0 ~ N(0,I), x1 ~ q(data)}
               [ || v_theta(x_t, t) - (x1 - x0) ||^2 ]

    where x_t = (1-t)*x0 + t*x1  (Eq. 22, OT linear interpolation)
    and the target vector field is u_t = x1 - x0 (constant in time).

Sampling (Section 6.2):
    1. Draw x_0 ~ N(0, I)
    2. Solve dx/dt = v_theta(x, t) from t=0 to t=1 using Euler method
    3. x_1 = generated sample

References:
    Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.
    - Eq. 20: OT path  mu_t = t*x1, sigma_t = 1-(1-sigma_min)*t
    - Eq. 22: OT flow  psi_t(x) = (1-(1-sigma_min)*t)*x + t*x1
    - Eq. 23: CFM loss  E[||v_t(psi_t(x0)) - (x1-(1-sigma_min)*x0)||^2]
    - Section 6.2: sampling via ODE integration

    Jacob et al., "Quantization and Training of Neural Networks for Efficient
    Integer-Arithmetic-Only Inference," CVPR 2018.
    - Section 3, Figure 1.1b: QAT inserts fake-quantize nodes in forward pass
    - STE backward: gradient flows through quantize as identity

    Esser et al., "Scaling Rectified Flow Transformers for High-Resolution
    Image Synthesis," ICML 2024.
    - Section 3.1: Logit-normal timestep sampling t = sigmoid(N(m, s))
    - rf/lognorm(0.00, 1.00) consistently outperforms uniform sampling
    - Biases training toward perceptually critical intermediate timesteps

    Karras et al., "Analyzing and Improving the Training Dynamics of
    Diffusion Models," CVPR 2024.
    - EMA (Exponential Moving Average) of model weights for sampling
    - Standard practice: decay=0.9999, use EMA weights for evaluation
"""

import math
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from photonflow.model import PhotonFlowModel

__all__ = ["CFMLoss", "euler_sample", "Trainer", "EMA"]


# ---------------------------------------------------------------------------
# EMA -- Exponential Moving Average (Karras et al. CVPR 2024)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * param

    Use the shadow weights for evaluation/sampling (they produce better
    generation quality than raw training weights).

    Karras et al. "Analyzing and Improving the Training Dynamics of
    Diffusion Models," CVPR 2024: EMA is standard practice for diffusion
    and flow matching models. Typical decay: 0.9999.

    Args:
        model (nn.Module): The model to track.
        decay (float): EMA decay rate. Higher = smoother. Default 0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> dict:
        """Replace model weights with EMA shadow weights.

        Returns the original weights so they can be restored.
        """
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ---------------------------------------------------------------------------
# CFMLoss -- Conditional Flow Matching (Lipman 2023, Eq. 20-23)
# ---------------------------------------------------------------------------

class CFMLoss(nn.Module):
    """Conditional Flow Matching loss with Optimal Transport paths.

    Lipman 2023, Eq. 22-23 (with sigma_min = 0):

        x_t    = (1-t) * x0  +  t * x1          (OT interpolation, Eq. 22)
        target = x1 - x0                         (constant VF)
        loss   = || v_theta(x_t, t) - target ||^2   (MSE, Eq. 23)

    where x0 ~ N(0, I) is noise and x1 is data.

    Logit-normal timestep sampling (Esser et al. "Scaling Rectified Flow
    Transformers for High-Resolution Image Synthesis," ICML 2024):
        Instead of t ~ U[0,1], sample u ~ N(m, s) and set t = sigmoid(u).
        This biases training toward intermediate timesteps (around t=0.5)
        where the velocity prediction task is hardest. The SD3 paper found
        that rf/lognorm(0.00, 1.00) consistently outperforms uniform sampling.

        π_ln(t; m, s) = 1/(s√(2π)) · 1/(t(1-t)) · exp(-(logit(t) - m)²/(2s²))

    Args:
        sigma_min (float): Minimum std for the OT path (Lipman Eq. 20).
            Default 0.0 (standard linear interpolation).
        time_sampling (str): Timestep sampling strategy. One of:
            "uniform"    -- t ~ U[0,1] (Lipman 2023, original)
            "logit_normal" -- t = sigmoid(N(m, s)) (Esser et al. 2024, SD3)
        logit_normal_mean (float): Location parameter m for logit-normal.
            Default 0.0 (centers mass at t=0.5).
        logit_normal_std (float): Scale parameter s for logit-normal.
            Default 1.0 (SD3 recommended: rf/lognorm(0.00, 1.00)).
    """

    def __init__(
        self,
        sigma_min: float = 0.0,
        time_sampling: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.time_sampling = time_sampling
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def forward(
        self,
        model: nn.Module,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CFM loss for a batch of data.

        Args:
            model: v_theta network. Must accept (x, t) and return (B, D).
            x1:    (B, D) batch of real data samples.

        Returns:
            Scalar MSE loss.

        Raises:
            ValueError: if x1 is not 2D or batch is empty.
            RuntimeError: if model output shape mismatches target.
        """
        # --- Input validation ---
        if x1.ndim != 2:
            raise ValueError(
                f"CFMLoss expects 2D input (B, D), got shape {x1.shape}. "
                f"Flatten images before passing to CFMLoss."
            )
        B, D = x1.shape
        if B == 0:
            raise ValueError("Empty batch (B=0) passed to CFMLoss.")

        device = x1.device

        # --- Sample noise x0 ~ N(0, I) ---
        x0 = torch.randn_like(x1)

        # --- Sample time t ---
        if self.time_sampling == "logit_normal":
            # Esser et al. 2024 (SD3): t = sigmoid(N(m, s))
            # Biases toward intermediate timesteps where prediction is hardest
            u = self.logit_normal_mean + self.logit_normal_std * torch.randn(B, device=device)
            t = torch.sigmoid(u)
            # Clamp to avoid numerical issues at exact 0 or 1
            t = t.clamp(1e-5, 1.0 - 1e-5)
        else:
            t = torch.rand(B, device=device)

        # --- OT interpolation (Lipman Eq. 22) ---
        t_expand = t[:, None]  # (B, 1) for broadcasting
        if self.sigma_min == 0.0:
            x_t = (1.0 - t_expand) * x0 + t_expand * x1
        else:
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x0 + t_expand * x1

        # --- Target vector field (Lipman Eq. 23) ---
        if self.sigma_min == 0.0:
            target = x1 - x0
        else:
            target = x1 - (1.0 - self.sigma_min) * x0

        # --- Model prediction ---
        v_pred = model(x_t, t)

        # --- Shape check ---
        if v_pred.shape != target.shape:
            raise RuntimeError(
                f"Model output shape {v_pred.shape} != target shape "
                f"{target.shape}. Check model in_dim matches data dim."
            )

        return F.mse_loss(v_pred, target)

    def extra_repr(self) -> str:
        return f"sigma_min={self.sigma_min}"


# ---------------------------------------------------------------------------
# Euler ODE sampler (Lipman 2023, Section 6.2)
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

    Lipman 2023, Section 6.2: start from x_0 ~ N(0,I), integrate with
    the trained vector field.  OT paths give decent quality at NFE=20
    (Figure 4, right panel).

    Args:
        model:     Trained PhotonFlowModel.
        shape:     (B, D) -- number of samples and dimension.
        num_steps: Euler steps (= NFE). Default 20.
        device:    Torch device (str or torch.device).

    Returns:
        (B, D) generated samples.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    was_training = model.training
    model.eval()

    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((shape[0],), t_val, device=device)
        v = model(x, t)
        x = x + dt * v

    if was_training:
        model.train()

    return x


# ---------------------------------------------------------------------------
# Trainer -- full training pipeline
# ---------------------------------------------------------------------------

class Trainer:
    """PhotonFlow training pipeline.

    Handles config loading, dataset setup, training loop, noise toggle,
    QAT fine-tuning, checkpointing, and sample generation.

    Args:
        config: dict or path to YAML file.
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

        # QAT wrapper (Jacob 2018: insert fake-quantize in forward pass)
        self.qat_enabled = qat_cfg.get("enabled", False)
        if self.qat_enabled:
            from hardware.qat import QATWrapper
            bits = qat_cfg.get("bits", 4)
            self.model = QATWrapper(self.model, bits=bits)

        # Optimizer
        lr = config.get("training", {}).get("lr", 1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Gradient clipping (prevents exploding gradients)
        self.grad_clip = config.get("training", {}).get("grad_clip", 1.0)

        # Loss (with optional logit-normal timestep sampling)
        tcfg = config.get("training", {})
        time_sampling = tcfg.get("time_sampling", "uniform")
        logit_normal_mean = tcfg.get("logit_normal_mean", 0.0)
        logit_normal_std = tcfg.get("logit_normal_std", 1.0)
        self.criterion = CFMLoss(
            sigma_min=0.0,
            time_sampling=time_sampling,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
        )

        # EMA (Karras et al. 2024: use EMA weights for sampling)
        ema_cfg = config.get("ema", {})
        self.use_ema = ema_cfg.get("enabled", False)
        if self.use_ema:
            ema_decay = ema_cfg.get("decay", 0.9999)
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = None

        # Training params
        tcfg = config.get("training", {})
        self.total_steps = tcfg.get("total_steps", 50000)
        self.checkpoint_every = tcfg.get("checkpoint_every", 5000)
        self.sample_every = tcfg.get("sample_every", 5000)
        self.sample_steps = tcfg.get("sample_steps", 20)

        # LR scheduler (optional warmup + cosine decay)
        warmup_steps = tcfg.get("warmup_steps", 0)
        lr_schedule = tcfg.get("lr_schedule", "constant")
        if lr_schedule == "cosine":
            from torch.optim.lr_scheduler import LambdaLR
            total = self.total_steps

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total - warmup_steps, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None

        # Output dirs
        self.output_dir = config.get("output_dir", "outputs")
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.fig_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        # Logging state
        self.losses: list = []
        self.global_step: int = 0

    # ---- Model construction (fix #1: now passes time_dim) ----

    def _build_model(self) -> PhotonFlowModel:
        mcfg = self.config.get("model", {})
        return PhotonFlowModel(
            in_dim=mcfg.get("in_dim", 784),
            hidden_dim=mcfg.get("hidden_dim", 784),
            num_blocks=mcfg.get("num_blocks", 6),
            time_dim=mcfg.get("time_dim", 256),
            use_noise=mcfg.get("use_noise", True),
            sigma_s=mcfg.get("sigma_s", 0.02),
            sigma_t=mcfg.get("sigma_t", 0.01),
            gate_init=mcfg.get("gate_init", 0.0),
            seq_dim=mcfg.get("seq_dim", None),
            feat_dim=mcfg.get("feat_dim", None),
            monarch_init=mcfg.get("monarch_init", "identity"),
        )

    # ---- QAT unwrap helper (fix #2: no import in non-QAT paths) ----

    def _unwrap_model(self) -> PhotonFlowModel:
        """Get the underlying PhotonFlowModel (unwrap QATWrapper if present)."""
        if self.qat_enabled and hasattr(self.model, "model"):
            return self.model.model
        return self.model

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
            # [-1, 1] for CIFAR-10 (torchcfm convention, Lipman 2023)
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                T.Lambda(lambda x: x.view(-1)),
            ])
            dataset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}. Use 'mnist' or 'cifar10'.")

        return DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=True, drop_last=True,
        )

    # ---- Training loop (fixes #3, #7, #8) ----

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

            # Fix #8: auto-flatten >2D input (e.g. if user forgot flatten transform)
            if x1.ndim > 2:
                x1 = x1.view(x1.shape[0], -1)

            # Forward + backward
            loss = self.criterion(self.model, x1)

            self.optimizer.zero_grad()
            loss.backward()

            # Fix #3: gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # EMA update (Karras et al. 2024)
            if self.ema is not None:
                self.ema.update(self.model)

            # Log
            loss_val = loss.item()

            # Fix #7: NaN/Inf loss guard
            if math.isnan(loss_val) or math.isinf(loss_val):
                warnings.warn(
                    f"NaN/Inf loss at step {step+1}. Stopping training. "
                    f"Try lower lr or check model for numerical issues.",
                    RuntimeWarning,
                    stacklevel=1,
                )
                break

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

    # ---- Checkpointing (fix #2: uses _unwrap_model) ----

    def save_checkpoint(self) -> str:
        """Save model, optimizer, losses, and config to disk."""
        model_to_save = self._unwrap_model()
        path = os.path.join(
            self.ckpt_dir, f"checkpoint_step{self.global_step}.pth"
        )
        save_dict = {
            "step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
            "config": self.config,
        }
        if self.ema is not None:
            save_dict["ema_state_dict"] = self.ema.state_dict()
        torch.save(save_dict, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from disk."""
        state = torch.load(path, map_location=self.device)
        model_to_load = self._unwrap_model()
        model_to_load.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.losses = state.get("losses", [])
        self.global_step = state.get("step", 0)
        if self.ema is not None and "ema_state_dict" in state:
            self.ema.load_state_dict(state["ema_state_dict"])

    # ---- Sample generation (fix #2 + #9) ----

    @torch.no_grad()
    def generate_samples(self, n_samples: int = 64) -> torch.Tensor:
        """Generate and save a grid of samples using Euler ODE.

        Uses EMA weights for sampling if EMA is enabled (Karras et al. 2024).
        """
        model_for_sample = self._unwrap_model()
        in_dim = model_for_sample.in_dim

        # Use EMA weights for generation (produces better quality)
        backup = None
        if self.ema is not None:
            backup = self.ema.apply(model_for_sample)

        samples = euler_sample(
            model_for_sample,
            shape=(n_samples, in_dim),
            num_steps=self.sample_steps,
            device=self.device,
        )

        # Restore training weights
        if backup is not None:
            self.ema.restore(model_for_sample, backup)

        samples_viz = samples.clamp(0, 1)

        # Save grid image
        if in_dim == 784:
            img_h, img_w, ch = 28, 28, 1
        elif in_dim == 3072:
            img_h, img_w, ch = 32, 32, 3
        else:
            return samples

        try:
            import torchvision
            grid_n = int(math.sqrt(n_samples))
            imgs = samples_viz[: grid_n ** 2].view(-1, ch, img_h, img_w)
            grid = torchvision.utils.make_grid(imgs, nrow=grid_n, padding=2)
            path = os.path.join(
                self.fig_dir, f"samples_step{self.global_step}.png"
            )
            torchvision.utils.save_image(grid, path)
        except Exception as e:
            # Fix #9: warn instead of silently swallowing
            warnings.warn(
                f"Failed to save sample grid at step {self.global_step}: {e}",
                RuntimeWarning,
                stacklevel=1,
            )

        return samples


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    torch.manual_seed(42)
    print("Testing photonflow/train.py ...\n")

    # --- Test 1: CFMLoss -- shape, positive, no NaN ---
    from photonflow.model import PhotonFlowModel

    model = PhotonFlowModel(in_dim=64, hidden_dim=16, num_blocks=2, use_noise=False)
    model.train()

    criterion = CFMLoss(sigma_min=0.0)
    x1 = torch.randn(8, 64)
    loss = criterion(model, x1)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    print(f"  [PASS] Test 1 -- CFMLoss: scalar={loss.item():.4f}, positive, no NaN")

    # --- Test 2: CFMLoss gradients flow through all params ---
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Params with no gradient: {no_grad}"
    # Test 2b: gradient health -- no NaN/Inf in any gradient tensor
    for n, p in model.named_parameters():
        assert not torch.isnan(p.grad).any(), f"NaN grad in {n}"
        assert not torch.isinf(p.grad).any(), f"Inf grad in {n}"
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  [PASS] Test 2 -- Gradients: all {total_p:,} params, no NaN/Inf grads")

    # --- Test 3: euler_sample -- shape, no NaN ---
    model.eval()
    samples = euler_sample(model, shape=(4, 64), num_steps=10, device="cpu")
    assert samples.shape == (4, 64), f"Sample shape: {samples.shape}"
    assert not torch.isnan(samples).any(), "NaN in samples"
    assert not torch.isinf(samples).any(), "Inf in samples"
    print(f"  [PASS] Test 3 -- euler_sample: shape=(4,64), 10 steps, no NaN/Inf")

    # --- Test 3b: input validation ---
    # CFMLoss rejects 4D input
    try:
        criterion(model, torch.randn(4, 1, 8, 8))
        assert False, "Should have raised ValueError for 4D input"
    except ValueError:
        pass
    # CFMLoss rejects empty batch
    try:
        criterion(model, torch.randn(0, 64))
        assert False, "Should have raised ValueError for B=0"
    except ValueError:
        pass
    # euler_sample rejects num_steps=0
    try:
        euler_sample(model, shape=(4, 64), num_steps=0)
        assert False, "Should have raised ValueError for num_steps=0"
    except ValueError:
        pass
    print(f"  [PASS] Test 3b -- Validation: rejects 4D, empty batch, num_steps=0")

    # --- Test 4: CFMLoss with sigma_min > 0 (Lipman Eq. 20) ---
    criterion_sm = CFMLoss(sigma_min=1e-4)
    model.train()
    loss_sm = criterion_sm(model, x1)
    assert loss_sm.item() > 0, "sigma_min loss should be positive"
    assert not torch.isnan(loss_sm), "sigma_min loss is NaN"
    print(f"  [PASS] Test 4 -- CFMLoss(sigma_min=1e-4): loss={loss_sm.item():.4f}")

    # --- Test 5: Trainer -- 200 steps, loss decreases, grad_clip works ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "model": {
                "in_dim": 64, "hidden_dim": 16, "num_blocks": 2,
                "time_dim": 16, "use_noise": False,
            },
            "training": {
                "lr": 5e-3, "total_steps": 200,
                "checkpoint_every": 100, "sample_every": 200,
                "sample_steps": 5, "seed": 42, "grad_clip": 1.0,
            },
            "qat": {"enabled": False},
            "output_dir": tmpdir,
        }
        trainer = Trainer(config)

        synth_x = torch.randn(256, 64)
        synth_dl = DataLoader(TensorDataset(synth_x), batch_size=32, shuffle=True)

        print("\n  Training 200 steps on synthetic data (dim=64, 2 blocks)...")
        losses = trainer.train(dataloader=synth_dl)

        first_20 = sum(losses[:20]) / 20
        last_20 = sum(losses[-20:]) / 20
        assert last_20 < first_20, (
            f"Loss did not decrease: {first_20:.4f} -> {last_20:.4f}"
        )
        assert all(not math.isnan(l) for l in losses), "NaN in loss history"

        ckpt_files = list(Path(tmpdir, "checkpoints").glob("*.pth"))
        assert len(ckpt_files) >= 1, "No checkpoints saved"
        print(f"  [PASS] Test 5 -- Trainer: loss {first_20:.4f}->{last_20:.4f}, "
              f"no NaN, {len(ckpt_files)} ckpts, grad_clip=1.0")

    # --- Test 6: Trainer with QAT enabled (Jacob 2018) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config_qat = {
            "model": {
                "in_dim": 64, "hidden_dim": 16, "num_blocks": 2,
                "time_dim": 16, "use_noise": False,
            },
            "training": {
                "lr": 1e-3, "total_steps": 20,
                "checkpoint_every": 20, "sample_every": 100, "seed": 42,
            },
            "qat": {"enabled": True, "bits": 4},
            "output_dir": tmpdir,
        }
        trainer_qat = Trainer(config_qat)

        synth_dl2 = DataLoader(TensorDataset(torch.randn(64, 64)),
                               batch_size=16, shuffle=True)
        losses_qat = trainer_qat.train(dataloader=synth_dl2)
        assert len(losses_qat) == 20
        assert all(not math.isnan(l) for l in losses_qat), "NaN in QAT losses"
        print(f"  [PASS] Test 6 -- QAT training: 20 steps, no NaN, "
              f"loss {losses_qat[0]:.4f} -> {losses_qat[-1]:.4f}")

    # --- Test 7: Checkpoint round-trip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        config_rt = {
            "model": {"in_dim": 64, "hidden_dim": 16, "num_blocks": 2,
                      "time_dim": 16, "use_noise": False},
            "training": {"lr": 1e-3, "total_steps": 10, "checkpoint_every": 10,
                         "sample_every": 100, "seed": 42},
            "qat": {"enabled": False},
            "output_dir": tmpdir,
        }
        t1 = Trainer(config_rt)
        dl = DataLoader(TensorDataset(torch.randn(32, 64)), batch_size=8, shuffle=True)
        t1.train(dataloader=dl)
        ckpt = t1.save_checkpoint()

        t2 = Trainer(config_rt)
        t2.load_checkpoint(ckpt)
        assert t2.global_step == t1.global_step
        for (n1, p1), (n2, p2) in zip(
            t1.model.named_parameters(), t2.model.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6), f"Weight mismatch: {n1}"
        print(f"  [PASS] Test 7 -- Checkpoint round-trip: step={t2.global_step}, weights match")

    print("\nAll tests passed.")
    print()
    print("Fixes applied:")
    print("  [1] _build_model passes time_dim from config")
    print("  [2] _unwrap_model() replaces QATWrapper import in save/load/generate")
    print("  [3] Gradient clipping (training.grad_clip, default 1.0)")
    print("  [4] CFMLoss input validation (ndim != 2, B == 0)")
    print("  [5] CFMLoss model output shape check")
    print("  [6] euler_sample validates num_steps > 0")
    print("  [7] NaN/Inf loss guard stops training early")
    print("  [8] Auto-flatten >2D input in Trainer.train()")
    print("  [9] generate_samples warns on error instead of silent pass")
