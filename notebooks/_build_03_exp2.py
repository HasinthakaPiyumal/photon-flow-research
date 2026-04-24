"""Build notebooks/03_exp2_photonflow_mnist.ipynb.

Generates the Experiment 2 notebook with:
  - S_scale_4M architecture (5M params, num_blocks=5, adaLN-scale ON,
    cond_bias_hidden=576, use_noise=False, factors=3)
  - 200K iterations (K_bs256 recipe: bs=256, lr=3e-3, warmup=300, Adam)
  - Strict photon-native forward graph (0 nn.Linear / nn.SiLU / nn.Sigmoid /
    nn.ReLU / nn.GELU at inference time).
  - Same W&B integration pattern as the prior v17 notebook:
    environment detection -> dependency install -> repo clone -> imports ->
    wandb.login (from Colab/Kaggle/local secrets) -> wandb.init with
    project/name/config/tags/notes -> per-step train/eval logging ->
    sample grids + loss curves + final summary + wandb.finish
  - OpticalSampler (photon-native fixedpoint sampler) replaces legacy
    euler_sample (deleted in the zero-OEO refactor).

Run this script to regenerate the notebook:
    python notebooks/_build_03_exp2.py
"""
import json
import pathlib

CELLS = []


def code(src, cell_id=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src.splitlines()],
    }
    if cell_id:
        cell["id"] = cell_id
    CELLS.append(cell)


def md(src, cell_id=None):
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in src.splitlines()],
    }
    if cell_id:
        cell["id"] = cell_id
    CELLS.append(cell)


# ---------------------------------------------------------------------------
# Cell 1 -- intro markdown
# ---------------------------------------------------------------------------
md(r"""# Experiment 2 — PhotonFlow on MNIST (S_scale_4M, adaLN-scale, 200 K)

**Status**: this is the **strict-photon-native champion at baseline parity**
from `kaggle/photonflow-scale-sweep` (see `photonflow_experiments_report.md`
§26 for the full scale-sweep, and §24-§25 for the preceding adaLN-scale
break-through).

**Architecture** (`S_scale_4M`):

| Knob | Value | Role |
|---|---|---|
| `hidden_dim` | 784 | MNIST-flat input = 28² |
| `num_blocks` | 5 | depth (scale-sweep sweet-spot at baseline parity) |
| `num_monarch_factors` | 3 | stacked Cayley-unitary MZI meshes per Monarch |
| `time_dim` | 576 | time-embed width; perfect square (24²) |
| `cond_bias_hidden` | 576 | adaLN bottleneck hidden width |
| `use_adaln_scale` | **True** | multiplicative time modulation via electro-optic MZM |
| `adaln_init_std` | 0.5 | aggressive time-embed init (E_cb576 combo-v2 winner) |
| `learnable_absorber_alpha` | True | SA α is trainable |
| `absorber_leaky_slope` | 0.05 | 5 % linear bypass past tanh-saturation |
| `use_noise` | False | no photonic noise at train (noise regulariser = exp3) |

**Recipe** (K_bs256 winning configuration):

| Knob | Value |
|---|---|
| optimiser | `Adam` |
| `lr` | 3.0 × 10⁻³ |
| `warmup` | 300 steps |
| `batch_size` | 256 |
| schedule | cosine decay to 0 over `total_steps` |
| `grad_clip` | 1.0 |
| `seed` | 42 |
| loss | plain uniform-t `CFMLoss()` (no logit-normal, no direction-loss, no time-weighting) |

**Result at 2 K steps** (kaggle `photonflow-scale-sweep`, S_scale_4M row):

| Model | Params | ×baseline | Eval @ 2 K | Gap |
|---|---:|---:|---:|---:|
| Baseline (DiT attention + GELU + LayerNorm) | 4,886,544 | 1.00× | 0.1734 | 0.0000 |
| **PhotonFlow S_scale_4M ⭐** | **4,867,082** | **1.00×** | **0.2363** | **+0.0629** |

**This notebook scales the run to 200 K iterations** to see how much further
the strict-photon-native gap closes at long horizon.  Training is still
descending at step 2 K (`loss=0.24 avg50`), so we expect meaningful gap
closure by step 50-100 K.

**Zero-OEO guarantee**: every forward-graph op maps to a published on-chip
photonic primitive.  Module-tree audit: **0 `nn.Linear` / 0 `nn.SiLU` /
0 `nn.Sigmoid` / 0 `nn.ReLU` / 0 `nn.GELU`** (verified in every kernel).

References: `photonflow_experiments_report.md` §24-§26, `references.bib`,
Shen 2017, Clements 2016, Peebles 2023, Dao 2022, Lipman 2023.
""", cell_id="cell-md-01")


# ---------------------------------------------------------------------------
# Cell 2 -- environment + deps
# ---------------------------------------------------------------------------
code(r'''# -- 1. Environment detection + dependency install -------------------------
import sys, subprocess, os

IN_COLAB  = False
IN_KAGGLE = False
IN_LOCAL  = False

# Check Kaggle FIRST -- Kaggle can also import google.colab.
if os.path.exists('/kaggle/input') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    IN_KAGGLE = True
    ENV_NAME  = "Kaggle"
else:
    try:
        import google.colab  # noqa: F401
        IN_COLAB = True
        ENV_NAME = "Colab"
    except ImportError:
        IN_LOCAL = True
        ENV_NAME = "Local"

print(f"Environment: {ENV_NAME}")

def _pip_install(*pkgs):
    """Install packages one by one; warn rather than crash."""
    for pkg in pkgs:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  [ok] {pkg}")
        except subprocess.CalledProcessError:
            print(f"  [warn] {pkg} -- install failed (may need Internet)")
            if IN_KAGGLE:
                print("         Kaggle: Settings > Internet > toggle ON")

if IN_COLAB or IN_KAGGLE:
    print("Installing dependencies...")
    _pip_install("tqdm", "pyyaml", "wandb")

# Kaggle pre-installed torch (cu128) drops sm_60; force cu121 for P100 support.
if IN_KAGGLE:
    try:
        import torch as _torch_probe
        cap = _torch_probe.cuda.get_device_capability(0)
        if cap[0] < 7:
            print(f"[setup] GPU compute {cap[0]}.{cap[1]} < 7.0 -- reinstalling torch 2.5.1+cu121 ...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "--index-url", "https://download.pytorch.org/whl/cu121",
                "torch==2.5.1", "torchvision==0.20.1",
            ])
            print("[setup] torch 2.5.1+cu121 installed -- may need RESTART + re-run all cells")
        del _torch_probe
    except Exception as e:
        print(f"[setup] could not probe GPU/torch: {e}")

import torch
assert torch.cuda.is_available(), (
    "GPU not found.\n"
    "  Colab : Runtime > Change runtime type > T4 GPU\n"
    "  Kaggle: Settings > Accelerator > GPU T4 x2 (or P100)"
)
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
''', cell_id="cell-01-setup")


# ---------------------------------------------------------------------------
# Cell 3 -- repo clone / sys.path
# ---------------------------------------------------------------------------
code(r'''# -- 2. Repo / sys.path setup ----------------------------------------------
REPO_URL    = "https://github.com/HasinthakaPiyumal/photon-flow-research.git"
REPO_BRANCH = "h/phase1"

if IN_COLAB:
    REPO_DIR = "/content/photon-flow-research"
elif IN_KAGGLE:
    REPO_DIR = "/kaggle/working/photon-flow-research"
else:
    REPO_DIR = ".."  # notebook lives in notebooks/, repo root is one up

if (IN_COLAB or IN_KAGGLE) and not os.path.exists(REPO_DIR):
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR
    ])
    print(f"Cloned repo (branch={REPO_BRANCH}) to {REPO_DIR}")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print(f"Repo  : {os.path.abspath(REPO_DIR)}")
print(f"Branch: {REPO_BRANCH}")
print(f"HEAD  : ", end="")
try:
    print(subprocess.check_output(
        ["git", "-C", REPO_DIR, "log", "-1", "--oneline"]
    ).decode().strip())
except Exception:
    print("(not a git checkout)")
''', cell_id="cell-02-repo")


# ---------------------------------------------------------------------------
# Cell 4 -- imports
# ---------------------------------------------------------------------------
code(r'''# -- 3. Imports -------------------------------------------------------------
import math, json, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# PhotonFlow core
# - PhotonFlowModel: strict photon-native forward graph, now with use_adaln_scale
#   kwarg (electro-optic MZM multiplicative time modulation, commit 6701f5e).
# - CFMLoss: Lipman 2023 conditional flow matching with optional min-SNR / EDM /
#   SD3-shift / loss_weight_gamma levers (all off here for baseline comparability).
# - OpticalSampler: fixed-point photonic sampler (replaces the deleted
#   legacy euler_sample; 1 electronic op per generated sample = termination
#   comparator at the photodetector).
from photonflow.model import PhotonFlowModel
from photonflow.train import CFMLoss
from photonflow.sampler import OpticalSampler

# FID evaluation (Heusel 2017); optional -- skips gracefully if deps are missing.
try:
    from eval.fid import FIDCalculator
    FID_AVAILABLE = True
except Exception as _e:
    FID_AVAILABLE = False
    print(f"[warn] FID not available ({_e}); cell 11 will skip FID")

device = torch.device("cuda")
print(f"Device: {device}")
print("Imports OK")
''', cell_id="cell-03-imports")


# ---------------------------------------------------------------------------
# Cell 5 -- W&B init (get API key from secrets)
# ---------------------------------------------------------------------------
code(r'''# -- W&B init (get API key from secrets) -----------------------------------
import wandb

# Retrieve WANDB_API_KEY from environment secrets:
#  Colab : Add-ons > Secrets > "WANDB_API_KEY"
#  Kaggle: Add-ons > Secrets > "WANDB_API_KEY" (toggle Internet ON)
#  Local : export WANDB_API_KEY=... in your shell
WANDB_API_KEY = None

if IN_COLAB:
    try:
        from google.colab import userdata
        WANDB_API_KEY = userdata.get("WANDB_API_KEY")
    except Exception as e:
        print(f"[warn] Could not load WANDB_API_KEY from Colab secrets: {e}")
elif IN_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        WANDB_API_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as e:
        print(f"[warn] Could not load WANDB_API_KEY from Kaggle secrets: {e}")
else:
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    print("W&B login: OK")
else:
    print("[warn] WANDB_API_KEY not found -- running wandb in offline mode")
    os.environ["WANDB_MODE"] = "offline"

wandb_run = None  # initialised in cell 5 once CFG is built
print("W&B cell loaded -- run will be initialised after config is ready")
''', cell_id="cell-wandb-init")


# ---------------------------------------------------------------------------
# Cell 6 -- config + W&B init
# ---------------------------------------------------------------------------
code(r'''# -- 4. Experiment config + W&B run init -----------------------------------
import yaml

# Long-horizon S_scale_4M config (5M params, adaLN-scale, 200K iterations).
config_path = os.path.join(REPO_DIR, "configs", "exp2_mnist_5m_200k.yaml")
with open(config_path) as f:
    yaml_cfg = yaml.safe_load(f)

print(f"Loaded config: {config_path}")

m = yaml_cfg["model"]
t = yaml_cfg["training"]
d = yaml_cfg["data"]
s = yaml_cfg.get("sampler", {})

CFG = {
    # ---------- model (S_scale_4M, strict photon-native) ----------
    "in_dim":                   m.get("in_dim", 784),
    "hidden_dim":               m.get("hidden_dim", 784),
    "num_blocks":               m.get("num_blocks", 5),
    "time_dim":                 m.get("time_dim", 576),
    "num_monarch_factors":      m.get("num_monarch_factors", 3),
    "monarch_init":             m.get("monarch_init", "random"),
    "adaln_init_std":           m.get("adaln_init_std", 0.5),
    "absorber_alpha":           m.get("absorber_alpha", 0.8),
    "absorber_leaky_slope":     m.get("absorber_leaky_slope", 0.05),
    "learnable_absorber_alpha": m.get("learnable_absorber_alpha", True),
    "use_noise":                m.get("use_noise", False),
    "sigma_s":                  m.get("sigma_s", 0.0),
    "sigma_t":                  m.get("sigma_t", 0.0),
    "shot_signal_dependent":    m.get("shot_signal_dependent", False),
    "mean_center_norm":         m.get("mean_center_norm", False),
    "phase_noise_sigma":        m.get("phase_noise_sigma", 0.0),
    "cumulative_loss_db_per_stage": m.get("cumulative_loss_db_per_stage", 0.0003),
    "cond_bias_hidden":         m.get("cond_bias_hidden", 576),
    "use_adaln_scale":          m.get("use_adaln_scale", True),
    "seq_dim":                  m.get("seq_dim", None),
    "feat_dim":                 m.get("feat_dim", None),
    # ---------- data ----------
    "dataset":                  d["dataset"],
    "batch_size":               d.get("batch_size", 256),
    "data_root":                os.path.join(REPO_DIR, d.get("root", "./data")),
    "num_workers":              d.get("num_workers", 2),
    # ---------- training (K_bs256 winning recipe) ----------
    "lr":                       t.get("lr", 3.0e-3),
    "total_steps":              t.get("total_steps", 200_000),
    "warmup_steps":             t.get("warmup_steps", 300),
    "lr_schedule":              t.get("lr_schedule", "cosine"),
    "grad_clip":                t.get("grad_clip", 1.0),
    "checkpoint_every":         t.get("checkpoint_every", 10000),
    "sample_every":             t.get("sample_every", 5000),
    "sample_steps":             t.get("sample_steps", 20),
    "seed":                     t.get("seed", 42),
    "time_sampling":            t.get("time_sampling", "uniform"),
    "direction_loss_weight":    t.get("direction_loss_weight", 0.0),
    "loss_weight_gamma":        t.get("loss_weight_gamma", 0.0),
    "loss_weight_gamma_max":    t.get("loss_weight_gamma_max", 10.0),
    "loss_weight_mode":         t.get("loss_weight_mode", "none"),
    "sigma_data":               t.get("sigma_data", 0.5),
    "logit_normal_mean":        t.get("logit_normal_mean", 0.0),
    "logit_normal_std":         t.get("logit_normal_std", 1.0),
    "logit_normal_time_shift":  t.get("logit_normal_time_shift", 1.0),
    "curriculum_transition_frac": t.get("curriculum_transition_frac", 0.6),
    # ---------- sampler ----------
    "sampler_mode":             s.get("mode", "fixedpoint"),
    "sampler_tau":              s.get("tau", None),
    "sampler_eps":              s.get("eps", 1.0e-3),
    "sampler_max_iters":        s.get("max_iters", 20),
    # ---------- io ----------
    "output_dir":               os.path.join(REPO_DIR, yaml_cfg.get("output_dir", "outputs/exp2_photonflow_mnist_5m_200k")),
}

# Windows multiprocessing can't pickle transforms across workers cleanly in some
# setups; force num_workers=0 locally on Windows.
if IN_LOCAL and sys.platform.startswith("win"):
    CFG["num_workers"] = 0

IN_DIM = CFG["in_dim"]

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

OUTPUT_DIR = CFG["output_dir"]
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
for _d in [CKPT_DIR, FIG_DIR]:
    os.makedirs(_d, exist_ok=True)

_ts_k = max(1, CFG["total_steps"] // 1000)
wandb_run = wandb.init(
    project="photonflow",
    name=f"exp2-photonflow-S_scale_4M-{_ts_k}K-{ENV_NAME.lower()}",
    config=CFG,
    tags=["exp2", "photonflow", "monarch", "S_scale_4M",
          "adaln-scale", "mnist", "200K", ENV_NAME.lower()],
    notes=yaml_cfg["experiment"]["description"],
    reinit=True,
)

print()
print("=" * 64)
print("PhotonFlow S_scale_4M (5M params, adaLN-scale, 200K) config")
print("=" * 64)
print(f"  hidden_dim={CFG['hidden_dim']}  num_blocks={CFG['num_blocks']}  time_dim={CFG['time_dim']}")
print(f"  num_monarch_factors={CFG['num_monarch_factors']}  monarch_init={CFG['monarch_init']!r}")
print(f"  use_adaln_scale={CFG['use_adaln_scale']}  adaln_init_std={CFG['adaln_init_std']}")
print(f"  cond_bias_hidden={CFG['cond_bias_hidden']}")
print(f"  absorber_alpha={CFG['absorber_alpha']}  leaky_slope={CFG['absorber_leaky_slope']}")
print(f"  learnable_absorber_alpha={CFG['learnable_absorber_alpha']}")
print(f"  use_noise={CFG['use_noise']}  mean_center_norm={CFG['mean_center_norm']}")
print()
print(f"  lr={CFG['lr']}  warmup={CFG['warmup_steps']}/{CFG['total_steps']:,}  schedule={CFG['lr_schedule']}")
print(f"  batch_size={CFG['batch_size']}  num_workers={CFG['num_workers']}  seed={CFG['seed']}")
print(f"  loss=CFMLoss(time_sampling={CFG['time_sampling']!r}, mode={CFG['loss_weight_mode']!r})")
print()
print(f"  W&B run -> {wandb_run.url}")
''', cell_id="cell-04-config")


# ---------------------------------------------------------------------------
# Cell 7 -- MNIST DataLoader
# ---------------------------------------------------------------------------
code(r'''# -- 5. MNIST DataLoader ---------------------------------------------------
# Flatten (1,28,28) -> (784,) is done INLINE in the loop (NOT via a Lambda
# transform) because Windows multiprocessing DataLoaders can't pickle local
# lambdas.  Semantically identical to the baseline notebook's pipeline.
_tfm = transforms.ToTensor()

train_ds = datasets.MNIST(CFG["data_root"], train=True,  download=True, transform=_tfm)
test_ds  = datasets.MNIST(CFG["data_root"], train=False, download=True, transform=_tfm)

train_loader = DataLoader(
    train_ds, batch_size=CFG["batch_size"],
    shuffle=True, num_workers=CFG["num_workers"],
    pin_memory=True, drop_last=True,
)

eval_loader = DataLoader(
    train_ds, batch_size=CFG["batch_size"],
    shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    generator=torch.Generator().manual_seed(CFG["seed"] + 1),
)

steps_per_epoch = len(train_loader)
total_epochs    = CFG["total_steps"] / steps_per_epoch
print(f"Train: {len(train_ds):,}  Test: {len(test_ds):,}")
print(f"Batch size: {CFG['batch_size']}  Steps/epoch: {steps_per_epoch}")
print(f"Total epochs: {total_epochs:.1f}  ({CFG['total_steps']:,} steps)")
''', cell_id="cell-05-data")


# ---------------------------------------------------------------------------
# Cell 8 -- Model + optimiser + scheduler + audit
# ---------------------------------------------------------------------------
code(r'''# -- 6. PhotonFlowModel (S_scale_4M) + strict zero-OEO audit --------------
def _build_model():
    return PhotonFlowModel(
        in_dim                    = CFG["in_dim"],
        hidden_dim                = CFG["hidden_dim"],
        num_blocks                = CFG["num_blocks"],
        time_dim                  = CFG["time_dim"],
        use_noise                 = CFG["use_noise"],
        sigma_s                   = CFG["sigma_s"],
        sigma_t                   = CFG["sigma_t"],
        shot_signal_dependent     = CFG["shot_signal_dependent"],
        monarch_init              = CFG["monarch_init"],
        adaln_init_std            = CFG["adaln_init_std"],
        num_monarch_factors       = CFG["num_monarch_factors"],
        absorber_alpha            = CFG["absorber_alpha"],
        absorber_leaky_slope      = CFG["absorber_leaky_slope"],
        learnable_absorber_alpha  = CFG["learnable_absorber_alpha"],
        mean_center_norm          = CFG["mean_center_norm"],
        phase_noise_sigma         = CFG["phase_noise_sigma"],
        cumulative_loss_db_per_stage = CFG["cumulative_loss_db_per_stage"],
        cond_bias_hidden          = CFG["cond_bias_hidden"],
        use_adaln_scale           = CFG["use_adaln_scale"],
        seq_dim                   = CFG["seq_dim"],
        feat_dim                  = CFG["feat_dim"],
    ).to(device)

def _build_lr_scheduler(opt, total_steps, warmup_steps, schedule):
    """Linear warmup then cosine decay to 0; or constant if schedule != cosine."""
    from torch.optim.lr_scheduler import LambdaLR
    if schedule == "cosine":
        def _fn(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * p))
        return LambdaLR(opt, _fn)
    return None

def _audit_zero_oeo(model):
    """Fail fast if any electronic op snuck into the forward graph."""
    n_lin  = sum(1 for mod in model.modules() if isinstance(mod, nn.Linear))
    n_silu = sum(1 for mod in model.modules() if isinstance(mod, nn.SiLU))
    n_sig  = sum(1 for mod in model.modules() if isinstance(mod, nn.Sigmoid))
    n_relu = sum(1 for mod in model.modules() if isinstance(mod, nn.ReLU))
    n_gelu = sum(1 for mod in model.modules() if isinstance(mod, nn.GELU))
    assert n_lin == 0 and n_silu == 0 and n_sig == 0 and n_relu == 0 and n_gelu == 0, (
        f"electronic op in forward graph: "
        f"Linear={n_lin}, SiLU={n_silu}, Sigmoid={n_sig}, ReLU={n_relu}, GELU={n_gelu}"
    )
    return dict(n_lin=n_lin, n_silu=n_silu, n_sig=n_sig, n_relu=n_relu, n_gelu=n_gelu)

model     = _build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
scheduler = _build_lr_scheduler(optimizer, CFG["total_steps"], CFG["warmup_steps"], CFG["lr_schedule"])

# Two CFMLoss instances:
#   loss_fn      -- training (whatever's in CFG; defaults to plain uniform-t)
#   eval_loss_fn -- always plain uniform-t, for fair comparison with baseline.
loss_fn = CFMLoss(
    sigma_min=0.0,
    time_sampling            = CFG["time_sampling"],
    logit_normal_mean        = CFG["logit_normal_mean"],
    logit_normal_std         = CFG["logit_normal_std"],
    curriculum_transition_frac = CFG["curriculum_transition_frac"],
    direction_loss_weight    = CFG["direction_loss_weight"],
    loss_weight_gamma        = CFG["loss_weight_gamma"],
    loss_weight_gamma_max    = CFG["loss_weight_gamma_max"],
    loss_weight_mode         = CFG["loss_weight_mode"],
    sigma_data               = CFG["sigma_data"],
    logit_normal_time_shift  = CFG["logit_normal_time_shift"],
)
eval_loss_fn = CFMLoss()  # uniform-t plain MSE

total_params = model.count_parameters()
print(f"Model: PhotonFlowModel (S_scale_4M, adaLN-scale ON)")
print(f"  params = {total_params:,}")
print(f"  baseline reference = 4,886,544 (DiT/attention 4-block)")
ratio = total_params / 4_886_544
print(f"  -> {ratio:.2f}x baseline")

# Strict zero-OEO module-tree audit (hard-fails if any electronic op snuck in)
audit = _audit_zero_oeo(model)
print(f"  [audit] 0 nn.Linear / 0 nn.SiLU / 0 nn.Sigmoid / 0 nn.ReLU / 0 nn.GELU -- OK")

wandb.log({
    "model/total_params":       total_params,
    "model/params_vs_baseline": ratio,
    "audit/n_linear":           audit["n_lin"],
    "audit/n_silu":             audit["n_silu"],
    "audit/n_sigmoid":          audit["n_sig"],
    "audit/n_relu":             audit["n_relu"],
    "audit/n_gelu":             audit["n_gelu"],
})

# Sanity: forward pass + backward, no NaN, all params receive gradient.
with torch.no_grad():
    _x = torch.randn(4, CFG["in_dim"], device=device)
    _t = torch.rand(4, device=device)
    _y = model(_x, _t)
    _max = _y.abs().max().item()
print(f"Init forward OK: out shape {_y.shape}, max abs {_max:.4f}")

_loss = loss_fn(model, _x.detach())
_loss.backward()
_no_grad = [n for n, p in model.named_parameters() if p.grad is None]
assert not _no_grad, f"Params with no gradient: {_no_grad}"
optimizer.zero_grad(set_to_none=True)
print(f"Init backward OK: loss = {_loss.item():.4f}, all {sum(1 for _ in model.parameters())} params have grad")
''', cell_id="cell-06-model")


# ---------------------------------------------------------------------------
# Cell 9 -- verification run (500 steps)
# ---------------------------------------------------------------------------
code(r'''# -- 7. VERIFY FIRST 500 STEPS (quick sanity check) -----------------------
VERIFY_STEPS = min(500, CFG["total_steps"] // 4)
print(f"Verification run: {VERIFY_STEPS} steps ...")

# Build a fresh OpticalSampler for verification sample grid.
sampler = OpticalSampler(
    model,
    tau       = CFG["sampler_tau"],
    eps       = CFG["sampler_eps"],
    max_iters = CFG["sampler_max_iters"],
    mode      = CFG["sampler_mode"],
)

verify_losses = []
data_iter_v = iter(train_loader)
model.train()

for step in range(VERIFY_STEPS):
    try: x1, _ = next(data_iter_v)
    except StopIteration:
        data_iter_v = iter(train_loader); x1, _ = next(data_iter_v)
    x1 = x1.view(x1.size(0), -1).to(device, non_blocking=True)
    loss = loss_fn(model, x1)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if CFG["grad_clip"] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    verify_losses.append(loss.item())
    if (step + 1) % 100 == 0:
        avg = float(np.mean(verify_losses[-100:]))
        print(f"  step {step+1:>4d}/{VERIFY_STEPS}  loss={avg:.4f}")

first_100 = float(np.mean(verify_losses[:100]))
last_100  = float(np.mean(verify_losses[-100:]))
any_nan   = any(math.isnan(l) for l in verify_losses)
decreased = last_100 < first_100

print()
print(f"  Loss first 100: {first_100:.4f}")
print(f"  Loss last  100: {last_100:.4f}")
print(f"  Decreased:      {decreased}")
print(f"  Any NaN:        {any_nan}")

# Quick sample grid (16 images) via OpticalSampler (photon-native).
with torch.no_grad():
    quick_samp = sampler(shape=(16, IN_DIM), device=device)
quick_samp = quick_samp.clamp(0.0, 1.0).cpu().view(16, 1, 28, 28)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(quick_samp[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.suptitle(f"Exp2 verification ({VERIFY_STEPS} steps, loss={last_100:.4f})", fontsize=11)
plt.tight_layout()
wandb.log({"verify/samples": wandb.Image(fig)})
plt.show(); plt.close(fig)

if decreased and not any_nan:
    print("\n" + "=" * 50)
    print("  VERDICT: GO -- proceed to full training (Cell 8)")
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print("  VERDICT: NO-GO -- debug before full training")
    print("=" * 50)

# Reset for the real run: reseed, rebuild model+opt+sched, rebuild sampler.
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
model     = _build_model()
_audit_zero_oeo(model)  # audit again on the fresh model
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
scheduler = _build_lr_scheduler(optimizer, CFG["total_steps"], CFG["warmup_steps"], CFG["lr_schedule"])
sampler = OpticalSampler(
    model, tau=CFG["sampler_tau"], eps=CFG["sampler_eps"],
    max_iters=CFG["sampler_max_iters"], mode=CFG["sampler_mode"],
)
print(f"\nModel reset for full {CFG['total_steps']:,}-step training.")
''', cell_id="cell-07-verify")


# ---------------------------------------------------------------------------
# Cell 10 -- main training loop
# ---------------------------------------------------------------------------
code(r'''# -- 8. Training loop (200K steps) ----------------------------------------
# W&B logs:
#   train/loss              per-step training loss
#   train/loss_avg100       100-step rolling average
#   eval/uniform_t_loss     plain-CFM uniform-t eval (every CFG['sample_every']
#                           steps), 8-batch average -- DIRECTLY COMPARABLE
#                           with baseline notebook's reported loss
#   eval/best               running min of eval/uniform_t_loss
#   cb_stats/block_i        mean/std of cond_bias_proj final-layer bias
#                           (photon-native diagnostic; replaces v17's gate_mean)
#   lr/value                current optimiser learning rate
#   samples/step_N          sample grid at step N
@torch.no_grad()
def _compute_eval_loss(n_batches=8):
    """Uniform-t, plain-MSE CFM loss averaged over n_batches eval batches."""
    model.eval()
    it = iter(eval_loader)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        try: xe, _ = next(it)
        except StopIteration: break
        xe = xe.view(xe.size(0), -1).to(device, non_blocking=True)
        tot += eval_loss_fn(model, xe).item()
        n += 1
    model.train()
    return tot / max(1, n)

def _cond_bias_stats(block, dim):
    """Photon-native diagnostic: mean/std of cond_bias_proj final-layer bias.

    Replaces the v17 notebook's `gate_mean/block_i` (the new model has no
    `adaLN_proj[...].bias[2d:3d]` gate slot -- instead we track the output
    of the MonarchLinear that emits (scale, shift) per sub-layer).
    """
    proj = block.cond_bias_proj
    # cond_bias_proj is either a single MonarchLinear or a
    # Sequential(MonarchLinear, PPLNSigmoid, MonarchLinear).  Grab the FINAL
    # MonarchLinear's bias either way.
    final = proj[-1] if isinstance(proj, nn.Sequential) else proj
    b = final.bias.detach() if (hasattr(final, "bias") and final.bias is not None) else None
    if b is None:
        return {}
    # When use_adaln_scale=True the output is 4*dim = (scale1, shift1, scale2, shift2).
    # When it's False the output is 2*dim = (bias1, bias2).
    if CFG["use_adaln_scale"]:
        scale1 = b[0:dim]; shift1 = b[dim:2*dim]
        scale2 = b[2*dim:3*dim]; shift2 = b[3*dim:4*dim]
        return dict(scale1_mean=scale1.mean().item(), scale1_std=scale1.std().item(),
                    shift1_mean=shift1.mean().item(), shift1_std=shift1.std().item(),
                    scale2_mean=scale2.mean().item(), scale2_std=scale2.std().item(),
                    shift2_mean=shift2.mean().item(), shift2_std=shift2.std().item())
    else:
        bias1 = b[0:dim]; bias2 = b[dim:2*dim]
        return dict(bias1_mean=bias1.mean().item(), bias1_std=bias1.std().item(),
                    bias2_mean=bias2.mean().item(), bias2_std=bias2.std().item())

losses        = []   # per-step training loss
step_log      = []   # (step, avg100) for the loss-curve plot
eval_log      = []   # (step, eval_loss) every sample_every steps
data_iter     = iter(train_loader)
model.train()
t_start       = time.time()
best_eval     = float("inf")

pbar = tqdm(range(CFG["total_steps"]), desc="exp2 S_scale_4M", dynamic_ncols=True)

for step in pbar:
    try:
        x1, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x1, _ = next(data_iter)
    x1 = x1.view(x1.size(0), -1).to(device, non_blocking=True)

    loss = loss_fn(model, x1, step=step, total_steps=CFG["total_steps"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if CFG["grad_clip"] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    losses.append(loss.item())
    wandb.log({"train/loss": loss.item(), "lr/value": optimizer.param_groups[0]["lr"]}, step=step)

    if step % 100 == 0:
        avg = float(np.mean(losses[-100:]))
        step_log.append((step, avg))
        pbar.set_postfix(loss=f"{avg:.4f}")
        wandb.log({"train/loss_avg100": avg}, step=step)

    # Cond-bias-proj diagnostics every 10 % of the run (or every 5K, whichever larger)
    diag_every = max(5000, CFG["total_steps"] // 10)
    if step % diag_every == 0:
        stats = {}
        for i, blk in enumerate(model.blocks):
            for key, val in _cond_bias_stats(blk, blk.dim).items():
                stats[f"cb_stats/block_{i}/{key}"] = val
        if stats:
            wandb.log(stats, step=step)

    # uniform-t eval loss (baseline-comparable) + sample grid every sample_every
    if (step + 1) % CFG["sample_every"] == 0:
        ev = _compute_eval_loss(n_batches=8)
        eval_log.append((step + 1, ev))
        if ev < best_eval: best_eval = ev
        wandb.log({"eval/uniform_t_loss": ev, "eval/best": best_eval}, step=step)
        elapsed = time.time() - t_start
        print(f"  [step {step+1:>6d}] eval={ev:.4f}  best={best_eval:.4f}  elapsed={elapsed:.0f}s")

        # Sample grid via OpticalSampler (photon-native)
        with torch.no_grad():
            samp = sampler(shape=(64, IN_DIM), device=device)
        samp = samp.clamp(0.0, 1.0).cpu().view(64, 1, 28, 28)
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samp[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.suptitle(f"Exp2 S_scale_4M  step={step+1:,}  eval={ev:.4f}", fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, f"samples_step{step+1:07d}.png")
        plt.savefig(fig_path, dpi=100, bbox_inches="tight")
        wandb.log({f"samples/step_{step+1}": wandb.Image(fig)}, step=step)
        plt.show(); plt.close(fig)

    # Checkpoint
    if (step + 1) % CFG["checkpoint_every"] == 0:
        ckpt_path = os.path.join(CKPT_DIR, f"step_{step+1:07d}.pt")
        torch.save({
            "step": step + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "config": CFG,
        }, ckpt_path)

pbar.close()
total_time = time.time() - t_start
print(f"\nTraining complete: {len(losses):,} steps in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  final train loss (avg of last 100): {np.mean(losses[-100:]):.4f}")
print(f"  best uniform-t eval loss:           {best_eval:.4f}")
BASELINE_2K_REF = 0.1734
gap = best_eval - BASELINE_2K_REF
print(f"  baseline 2K reference eval:         {BASELINE_2K_REF:.4f}")
print(f"  gap-to-baseline (best_eval - ref):  {gap:+.4f}")
wandb.log({"summary/baseline_ref": BASELINE_2K_REF, "summary/gap_vs_baseline": gap})
''', cell_id="cell-08-train")


# ---------------------------------------------------------------------------
# Cell 11 -- loss curves
# ---------------------------------------------------------------------------
code(r'''# -- 9. Loss curves (training + uniform-t eval) ---------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))

if step_log:
    s, l = zip(*step_log)
    ax.plot(s, l, lw=1.2, alpha=0.8, color="tab:orange",
            label="train loss (avg of 100 steps)")

if eval_log:
    es, el = zip(*eval_log)
    ax.plot(es, el, "-o", lw=1.6, ms=4.5, color="tab:blue",
            label="eval loss (uniform-t, 8-batch avg)")

ax.axhline(0.1734, ls="--", color="gray", alpha=0.6, label="baseline @ 2K (0.1734)")
ax.axhline(0.1734 + 0.05, ls=":", color="green", alpha=0.5, label="target gap = 0.05")

ax.set_xlabel("Training step", fontsize=12)
ax.set_ylabel("CFM loss (MSE)", fontsize=12)
_ts_k = max(1, CFG["total_steps"] // 1000)
ax.set_title(
    f"Exp2 PhotonFlow S_scale_4M -- {_ts_k}K steps  "
    f"({total_params:,} params, {total_params/4886544:.2f}x baseline, adaLN-scale)",
    fontsize=12,
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
curve_path = os.path.join(FIG_DIR, "loss_curve.png")
plt.savefig(curve_path, dpi=150)
wandb.log({"charts/loss_curve": wandb.Image(fig)})
plt.show(); plt.close(fig)
print(f"Loss curve saved: {curve_path}")
''', cell_id="cell-09-loss")


# ---------------------------------------------------------------------------
# Cell 12 -- final sample grid
# ---------------------------------------------------------------------------
code(r'''# -- 10. Final 10x10 sample grid ------------------------------------------
with torch.no_grad():
    final_samp = sampler(shape=(100, IN_DIM), device=device)
final_samp = final_samp.clamp(0.0, 1.0).cpu().view(100, 1, 28, 28)

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(final_samp[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.suptitle(f"Exp2 S_scale_4M final samples  (step {CFG['total_steps']:,})", fontsize=13)
plt.tight_layout()
samp_path = os.path.join(FIG_DIR, "final_samples.png")
plt.savefig(samp_path, dpi=150, bbox_inches="tight")
wandb.log({"samples/final_10x10": wandb.Image(fig)})
plt.show(); plt.close(fig)
print(f"Final samples saved: {samp_path}")
''', cell_id="cell-10-samples")


# ---------------------------------------------------------------------------
# Cell 13 -- FID
# ---------------------------------------------------------------------------
code(r'''# -- 11. FID computation (optional) ---------------------------------------
fid_score = float("nan")
if not FID_AVAILABLE:
    print("[skip] eval.fid not importable; skipping FID")
else:
    N_FID    = 10_000
    fid_calc = FIDCalculator(device=device)

    real_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    real_imgs = []
    for imgs, _ in real_loader:
        real_imgs.append(imgs.view(-1, 1, 28, 28))
    real_imgs = torch.cat(real_imgs, dim=0)[:N_FID]
    print(f"Real images: {real_imgs.shape}")

    model.eval()
    gen_batches = []
    GEN_BATCH   = 256
    n_generated = 0
    with torch.no_grad():
        pbar_fid = tqdm(total=N_FID, desc="Generating for FID", unit="img")
        while n_generated < N_FID:
            bs   = min(GEN_BATCH, N_FID - n_generated)
            samp = sampler(shape=(bs, IN_DIM), device=device)
            gen_batches.append(samp.clamp(0.0, 1.0).cpu().view(bs, 1, 28, 28))
            n_generated += bs
            pbar_fid.update(bs)
        pbar_fid.close()
    gen_imgs = torch.cat(gen_batches, dim=0)[:N_FID]

    print("Extracting features...")
    real_feats = fid_calc.extract_features(real_imgs, batch_size=256)
    gen_feats  = fid_calc.extract_features(gen_imgs,  batch_size=256)
    real_stats = fid_calc.compute_statistics(real_feats)
    gen_stats  = fid_calc.compute_statistics(gen_feats)
    fid_score  = fid_calc.compute_fid(real_stats, gen_stats)
    _ts_k = max(1, CFG["total_steps"] // 1000)
    print(f"\nFID (PhotonFlow S_scale_4M, {_ts_k}K steps): {fid_score:.2f}")
    wandb.log({"eval/fid": fid_score})

    # Compare to exp1 baseline if its results.json exists
    exp1_path = os.path.join(REPO_DIR, "outputs", "exp1_baseline", "results_exp1.json")
    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            exp1 = json.load(f)
        exp1_fid  = exp1.get("fid", float("nan"))
        delta_pct = ((fid_score - exp1_fid) / exp1_fid) * 100 if exp1_fid > 0 else float("nan")
        target    = exp1_fid * 1.10
        passed    = fid_score <= target
        print(f"Exp1 baseline FID: {exp1_fid:.2f}")
        print(f"FID delta:         {delta_pct:+.1f}%   (target: within +10%)")
        print(f"Verdict:           {'PASS' if passed else 'FAIL'}  (target FID <= {target:.2f})")
        wandb.log({"eval/exp1_fid": exp1_fid, "eval/fid_delta_pct": delta_pct})
    else:
        print("(exp1 results not found -- run notebook 02 first)")
''', cell_id="cell-11-fid")


# ---------------------------------------------------------------------------
# Cell 14 -- results summary + wandb.finish
# ---------------------------------------------------------------------------
code(r'''# -- 12. Results summary + W&B finish -------------------------------------
results = {
    "experiment":  "exp2_photonflow_mnist_S_scale_4M_200k",
    "environment": ENV_NAME,
    "description": "PhotonFlow S_scale_4M (5M, adaLN-scale, strict zero-OEO) @ 200K steps",
    "total_steps": CFG["total_steps"],
    "fid":         round(float(fid_score), 4) if not math.isnan(fid_score) else None,
    "final_loss_avg100":          round(float(np.mean(losses[-100:])), 6),
    "best_uniform_t_eval_loss":   round(float(best_eval), 6),
    "baseline_2k_reference_eval": 0.1734,
    "gap_vs_baseline":            round(float(best_eval) - 0.1734, 6),
    "model_params":               total_params,
    "params_vs_baseline":         round(total_params / 4_886_544, 4),
    "architecture": {
        "type":                       "photonflow",
        "hidden_dim":                 CFG["hidden_dim"],
        "num_blocks":                 CFG["num_blocks"],
        "time_dim":                   CFG["time_dim"],
        "num_monarch_factors":        CFG["num_monarch_factors"],
        "monarch_init":               CFG["monarch_init"],
        "adaln_init_std":             CFG["adaln_init_std"],
        "absorber_alpha":             CFG["absorber_alpha"],
        "absorber_leaky_slope":       CFG["absorber_leaky_slope"],
        "learnable_absorber_alpha":   CFG["learnable_absorber_alpha"],
        "cond_bias_hidden":           CFG["cond_bias_hidden"],
        "use_adaln_scale":            CFG["use_adaln_scale"],
        "use_noise":                  CFG["use_noise"],
    },
    "training": {
        "optimizer":             "Adam",
        "lr":                    CFG["lr"],
        "warmup_steps":          CFG["warmup_steps"],
        "lr_schedule":           CFG["lr_schedule"],
        "grad_clip":             CFG["grad_clip"],
        "batch_size":            CFG["batch_size"],
        "dataset":               CFG["dataset"],
        "seed":                  CFG["seed"],
        "loss":                  "CFMLoss(uniform-t, plain MSE)",
    },
    "sampler": {
        "type":       "OpticalSampler",
        "mode":       CFG["sampler_mode"],
        "tau":        CFG["sampler_tau"],
        "eps":        CFG["sampler_eps"],
        "max_iters":  CFG["sampler_max_iters"],
    },
    "audit": {
        "electronic_forward_ops": 0,
        "strict_photon_native":   True,
        "primitives": [
            "MonarchLayer -> Cayley-unitary MZI mesh (Shen 2017, Clements 2016)",
            "DivisivePowerNorm -> microring + photodetector + fixed SOA gain",
            "SaturableAbsorber -> graphene waveguide insert",
            "WavelengthCodedTime -> AWGR time encoder (Moss 2022)",
            "PPLNSigmoid -> PPLN chi^2 nonlinearity (eLight 2026)",
            "adaLN-scale (1+scale)*x + shift -> electro-optic MZM bank (Shen 2017 II / Clements 2016 III)",
            "coherent-add residual -> tunable directional coupler",
        ],
    },
    "sources": [
        "Dao et al. 2022 -- Monarch matrices M=PLP^TR (ICML, Def 3.1)",
        "Lipman et al. 2023 -- Flow Matching CFM loss (ICLR, Eq. 23)",
        "Peebles & Xie 2023 -- adaLN-Zero conditioning (DiT, ICCV)",
        "Shen et al. 2017 -- Saturable absorber on MZI mesh (Nature Photonics)",
        "Clements et al. 2016 -- MZI mesh unitary decomposition (Optica)",
        "Moss 2022 -- AWGR wavelength-demux time encoder",
        "Internal report -- photonflow_experiments_report.md (sections 24-26)",
    ],
}

results_path = os.path.join(OUTPUT_DIR, "results_exp2.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, "losses_exp2.npy"), np.array(losses))
if eval_log:
    np.save(os.path.join(OUTPUT_DIR, "eval_log_exp2.npy"), np.array(eval_log))

final_ckpt = os.path.join(CKPT_DIR, "exp2_final.pt")
torch.save({
    "step":                  CFG["total_steps"],
    "model_state_dict":      model.state_dict(),
    "optimizer_state_dict":  optimizer.state_dict(),
    "results":               results,
    "config":                CFG,
}, final_ckpt)

wandb.log({
    "summary/fid":               fid_score if not math.isnan(fid_score) else 0.0,
    "summary/final_train_loss":  float(np.mean(losses[-100:])),
    "summary/best_eval":         best_eval,
    "summary/gap_vs_baseline":   best_eval - 0.1734,
    "summary/total_params":      total_params,
})
wandb.save(results_path)
wandb.finish()

SEP = "=" * 64
print(SEP)
print("EXP2 PHOTONFLOW S_scale_4M (200K) RESULTS")
print(SEP)
print(f"  Model parameters:         {total_params:,}  ({total_params/4886544:.2f}x baseline)")
print(f"  Training steps:           {CFG['total_steps']:,}")
print(f"  Final train loss (100):   {float(np.mean(losses[-100:])):.4f}")
print(f"  Best uniform-t eval:      {best_eval:.4f}")
print(f"  Baseline 2K reference:    0.1734")
print(f"  Gap vs baseline:          {best_eval - 0.1734:+.4f}")
if not math.isnan(fid_score):
    print(f"  FID (10K samples):        {fid_score:.2f}")
print(f"  Environment:              {ENV_NAME}")
print(f"  Strict zero-OEO forward:  YES (0 nn.Linear / SiLU / Sigmoid / ReLU / GELU)")
print(SEP)
print(f"  Results    -> {results_path}")
print(f"  Checkpoint -> {final_ckpt}")
print(f"  W&B run    -> {wandb_run.url}")
print(SEP)
print()
print("Next: notebooks/04_exp3_noise_regularized.ipynb")
print("      Add shot noise (sigma_s=0.001) + thermal crosstalk (sigma_t=0.005)")
''', cell_id="cell-12-results")


# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "03_exp2_photonflow_mnist.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out}  ({len(CELLS)} cells)")
