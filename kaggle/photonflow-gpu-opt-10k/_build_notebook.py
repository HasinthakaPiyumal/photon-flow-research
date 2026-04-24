"""Kaggle kernel `photonflow-gpu-opt-10k` -- GPU-speed optimisation @ 10K steps.

Measures speed + quality of three variants (single kernel, 10K steps each):

  - baseline          DiT attention (reference for gap)                eager
  - photonflow_eager  S_scale_4M strict photon-native                  eager
  - photonflow_opt    S_scale_4M strict photon-native                  torch.compile + bf16

Optimisations in photonflow_opt preserve:
  - Model math: torch.compile is JIT kernel fusion; bf16 autocast is
    precision-level, maps to the 4-6 bit photonic MZI effective precision.
  - Zero-OEO contract: neither wrapper adds any nn.Linear/SiLU/Sigmoid/
    ReLU/GELU to the module tree.  audit_module_tree is run on the
    UN-compiled model before torch.compile wraps it (compile turns
    .modules() into an opaque graph).

At end of run, each variant saves a final 8x8 sample grid PNG:
  /kaggle/working/logs/samples_<tag>_10k.png
These get downloaded locally to outputs/exp_gpu_opt_compare/ after
kernel COMPLETE.
"""
import json
import pathlib

CELLS = []


def code(src):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src.splitlines()],
    })


def md(src):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in src.splitlines()],
    })


md(r'''# PhotonFlow GPU-opt 10K (baseline + eager + optimised)

Measures three variants at 10K MNIST CFM steps:

| Tag | Model | Mode |
|---|---|---|
| `baseline`          | DiT attention + GELU + LayerNorm (~4.89 M) | eager |
| `photonflow_eager`  | S_scale_4M strict photon-native (~4.87 M)  | eager |
| `photonflow_opt`    | S_scale_4M strict photon-native (~4.87 M)  | **torch.compile + bf16** |

All three share hyperparams: bs=256, lr=3e-3, warmup=300, Adam, cosine decay, grad_clip=1.0, seed=42.

Target: photonflow_opt is >=1.67x faster than photonflow_eager at 10K
with eval delta <= 0.01.  Module-tree audit: 0 electronic ops in
forward graph for both photonflow variants (the `torch.compile`
wrapper does not add nn.Modules -- it JITs the existing graph).

Final sample grids (8x8, 64 samples, OpticalSampler fixedpoint) are
saved as PNGs for each variant at step 10K.
''')


code(r'''# 0. Torch compatibility
import subprocess, sys
probe = subprocess.run(
    [sys.executable, '-c',
     'import torch; c = torch.cuda.get_device_capability(0); print(f"cap={c[0]}.{c[1]} ok={c[0]>=7}")'],
    capture_output=True, text=True
)
print(probe.stdout.strip())
if 'ok=False' in probe.stdout:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--quiet',
        '--index-url', 'https://download.pytorch.org/whl/cu121',
        'torch==2.5.1', 'torchvision==0.20.1',
    ])
    print("torch 2.5.1+cu121 installed")''')


code(r'''# 1. Clone repo (adaLN-scale + paper-informed CFMLoss on h/phase1)
import os, subprocess, sys
REPO_URL    = "https://github.com/HasinthakaPiyumal/photon-flow-research.git"
REPO_BRANCH = "h/phase1"
REPO_DIR    = "/kaggle/working/photon-flow-research"

if not os.path.exists(REPO_DIR):
    subprocess.check_call([
        "git","clone","--depth","1","--branch",REPO_BRANCH, REPO_URL, REPO_DIR
    ])
sys.path.insert(0, REPO_DIR)
print("HEAD:", subprocess.check_output(["git","-C",REPO_DIR,"log","-1","--oneline"]).decode().strip())
from photonflow import PhotonFlowModel, OpticalSampler
from photonflow.train import CFMLoss
_m = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=2, time_dim=16,
                     use_noise=False, use_adaln_scale=True)
assert _m.blocks[0].use_adaln_scale, "use_adaln_scale kwarg missing"
print("photon-native API + adaLN-scale verified")''')


code(r'''# 2. GPU check
import torch
assert torch.cuda.is_available(), "need GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
print(f"BF16 support: {torch.cuda.is_bf16_supported()}")''')


code(r'''# 3. Shared base arch (S_scale_4M) + audit helper
import torch, torch.nn as nn, math
from photonflow.model import PhotonFlowModel
from photonflow.train import CFMLoss
from photonflow.sampler import OpticalSampler

# S_scale_4M arch (from configs/exp2_mnist_5m_200k.yaml; scale-sweep winner
# at baseline-parity params, gap +0.0629 at 2K).
BASE_ARCH = dict(
    in_dim=784, hidden_dim=784,
    num_blocks=5, time_dim=576,
    use_noise=False, sigma_s=0.0, sigma_t=0.0,
    shot_signal_dependent=False,
    adaln_init_std=0.5,
    num_monarch_factors=3,
    absorber_alpha=0.8, absorber_leaky_slope=0.05,
    learnable_absorber_alpha=True,
    mean_center_norm=False,
    phase_noise_sigma=0.0,
    cumulative_loss_db_per_stage=0.0003,
    cond_bias_hidden=576,
    monarch_init='random',
    use_adaln_scale=True,
)

def audit_module_tree(model, tag):
    """Run on the UN-compiled model; torch.compile makes .modules() opaque."""
    n_lin  = sum(1 for mod in model.modules() if isinstance(mod, nn.Linear))
    n_silu = sum(1 for mod in model.modules() if isinstance(mod, nn.SiLU))
    n_sig  = sum(1 for mod in model.modules() if isinstance(mod, nn.Sigmoid))
    n_relu = sum(1 for mod in model.modules() if isinstance(mod, nn.ReLU))
    n_gelu = sum(1 for mod in model.modules() if isinstance(mod, nn.GELU))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_lin == 0 and n_silu == 0 and n_sig == 0 and n_relu == 0 and n_gelu == 0, \
        f"{tag}: electronic op in tree (Linear={n_lin}, SiLU={n_silu}, Sigmoid={n_sig}, ReLU={n_relu}, GELU={n_gelu})"
    print(f"  [audit] {tag}: 0 electronic ops, {n_params:,} params")
    return n_params''')


code(r'''# 4. Training harness -- supports optional torch.compile + bf16 autocast
import logging, time, pathlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

OUT = pathlib.Path("/kaggle/working/logs"); OUT.mkdir(exist_ok=True)
ROOT = pathlib.Path("/kaggle/working")
DATA_ROOT = str(ROOT / "data")

def run_experiment(tag, build_model, *,
                   total_steps=10_000, batch_size=256, lr=3e-3,
                   warmup_steps=300, grad_clip=1.0,
                   eval_every=2000, eval_batches=8, seed=42,
                   is_photonflow=True, use_compile=False, use_bf16=False):
    log_path = OUT / f"{tag}.log"
    try: log_path.unlink()
    except FileNotFoundError: pass
    logger = logging.getLogger(tag); logger.handlers.clear(); logger.propagate = False
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    sh = logging.StreamHandler(); sh.setFormatter(fh.formatter)
    logger.addHandler(fh); logger.addHandler(sh)
    def log(msg): logger.info(msg)
    def logkv(**kv): logger.info(" ".join(f"{k}={v}" for k, v in kv.items()))

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    log("="*70); log(f"EXP: {tag}")
    log(f"total_steps={total_steps} batch_size={batch_size} lr={lr} warmup={warmup_steps}")
    log(f"is_photonflow={is_photonflow} use_compile={use_compile} use_bf16={use_bf16}")

    tfm = transforms.ToTensor()
    ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    # 1. Build the raw model FIRST (so audit sees original modules).
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params={n_params:,}")
    if is_photonflow:
        audit_module_tree(model, tag)  # BEFORE torch.compile

    # 2. Wrap with torch.compile (JIT fusion, no new nn.Modules).
    if use_compile:
        log("wrapping with torch.compile(mode='reduce-overhead') ...")
        model = torch.compile(model, mode='reduce-overhead')

    # 3. Build the training loss.
    criterion = CFMLoss()
    eval_criterion = CFMLoss()  # plain uniform-t eval
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    eval_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1))

    def compute_eval():
        model.eval(); it = iter(eval_loader); tot, n = 0.0, 0
        with torch.no_grad():
            for _ in range(eval_batches):
                try: xe, _ = next(it)
                except StopIteration: break
                xe = xe.view(xe.size(0), -1).to(device, non_blocking=True)
                # Eval always in FP32 for fair comparison
                tot += eval_criterion(model, xe).item(); n += 1
        model.train()
        return tot / max(1, n)

    model.train(); it = iter(loader); losses = []; eval_hist = []
    best_eval = float("inf"); t_start = time.time(); t_last = t_start
    step_times = []  # wall-clock per-step in ms, rolling
    log("--- training start ---")
    # Track time in chunks of 500 steps (skip first chunk: compile warmup)
    COMPILE_WARMUP_STEPS = 100 if use_compile else 0

    for step in range(1, total_steps + 1):
        try: x, _ = next(it)
        except StopIteration: it = iter(loader); x, _ = next(it)
        x = x.view(x.size(0), -1).to(device, non_blocking=True)

        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = criterion(model, x)
        else:
            loss = criterion(model, x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step(); scheduler.step()
        losses.append(loss.item())

        if step % 500 == 0 or step == 1:
            torch.cuda.synchronize()  # make ms/step accurate
            now = time.time()
            chunk_ms = 1000 * (now - t_last) / 500 if step > 1 else 0.0
            t_last = now
            if step > COMPILE_WARMUP_STEPS:
                step_times.append(chunk_ms)
            a50 = sum(losses[-50:])/min(50, len(losses))
            logkv(step=f"{step:5d}/{total_steps}",
                  loss=f"{loss.item():.4f}", avg50=f"{a50:.4f}",
                  lr=f"{scheduler.get_last_lr()[0]:.2e}", gnorm=f"{gnorm:.3f}",
                  ms=f"{chunk_ms:.1f}", elap=f"{now-t_start:.0f}s")

        if step % eval_every == 0:
            ev = compute_eval(); eval_hist.append((step, ev))
            best_eval = min(best_eval, ev)
            log(f"[EVAL] step={step} uniform_t_loss={ev:.4f} best={best_eval:.4f}")

    torch.cuda.synchronize()
    wall = time.time() - t_start
    mean_ms_per_step = (sum(step_times) / len(step_times)) if step_times else float('nan')
    final_eval = eval_hist[-1][1] if eval_hist else float("nan")
    log(f"[SUMMARY] tag={tag} params={n_params:,}")
    log(f"[SUMMARY] final_eval={final_eval:.4f} best_eval={best_eval:.4f}")
    log(f"[SUMMARY] wall={wall:.1f}s mean_ms_per_step={mean_ms_per_step:.1f}")
    return dict(
        tag=tag, final_eval=final_eval, best_eval=best_eval, n_params=n_params,
        eval_hist=eval_hist, wall=wall, mean_ms_per_step=mean_ms_per_step,
        model=model, is_photonflow=is_photonflow,
    )''')


code(r'''# 5. Sample-grid helper (saves PNG at /kaggle/working/logs/samples_<tag>_10k.png)
def save_sample_grid(model, tag, is_photonflow, device, n=64):
    import matplotlib.pyplot as plt
    with torch.no_grad():
        model.eval()
        if is_photonflow:
            sampler = OpticalSampler(model, tau=None, eps=1e-3, max_iters=20, mode='fixedpoint')
            samp = sampler(shape=(n, 784), device=device)
        else:
            # baseline uses Euler-like integration: x = x0 + sum(v_theta dt)
            B = n; D = 784
            x = torch.randn(B, D, device=device)
            num_steps = 20
            for k in range(num_steps):
                t = torch.full((B,), (k + 0.5) / num_steps, device=device)
                v = model(x, t)
                x = x + v / num_steps
            samp = x
        model.train()
    samp = samp.clamp(0.0, 1.0).cpu().view(n, 1, 28, 28)
    side = int(n ** 0.5)
    fig, axes = plt.subplots(side, side, figsize=(side, side))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samp[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"{tag} @ step 10,000", fontsize=11)
    plt.tight_layout()
    png_path = OUT / f"samples_{tag}_10k.png"
    plt.savefig(str(png_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {png_path}")
    return str(png_path)''')


code(r'''# 6. Baseline (DiT attention) -- reference for gap
import torch.nn as nn, math, traceback

def _sinu(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half-1))
    a = t[:,None].float() * f[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)

def _mod(x, s, sc): return x * (1.0 + sc.unsqueeze(1)) + s.unsqueeze(1)

class BaselineBlock(nn.Module):
    def __init__(self, d, h, td, r=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.a  = nn.MultiheadAttention(d, h, batch_first=True, dropout=0.0)
        m = int(d*r)
        self.mlp = nn.Sequential(nn.Linear(d,m), nn.GELU(), nn.Linear(m,d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 6*d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
    def forward(self, x, te):
        s1,sc1,g1,s2,sc2,g2 = self.ada(te).chunk(6, dim=-1)
        h = _mod(self.n1(x), s1, sc1); h,_ = self.a(h,h,h)
        x = x + g1.unsqueeze(1) * h
        h = _mod(self.n2(x), s2, sc2); h = self.mlp(h)
        x = x + g2.unsqueeze(1) * h
        return x

class BaselineCFM(nn.Module):
    def __init__(self, d=256, heads=4, layers=4, td=256, r=4.0, ps=4):
        super().__init__()
        self.ps=ps; self.pd=ps*ps; self.np=(28//ps)**2
        self.pp = nn.Linear(self.pd, d)
        self.pe = nn.Parameter(torch.randn(1, self.np, d)*0.02)
        self.tm = nn.Sequential(nn.Linear(d, td), nn.SiLU(), nn.Linear(td, td))
        self.blocks = nn.ModuleList([BaselineBlock(d, heads, td, r) for _ in range(layers)])
        self.n = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.op = nn.Linear(d, self.pd)
        nn.init.zeros_(self.op.weight); nn.init.zeros_(self.op.bias)
        self.d = d
    def forward(self, x, t):
        B = x.size(0)
        patches = x.view(B, self.np, self.pd)
        h = self.pp(patches) + self.pe
        te = _sinu(t, self.d); te = self.tm(te)
        for blk in self.blocks: h = blk(h, te)
        h = self.n(h); o = self.op(h)
        return o.view(B, -1)

RESULTS = {}

# baseline: DiT attention, eager, 10K steps, lr=1e-4, wu=0 (matches prior runs)
print("\n" + "="*78 + "\n== baseline: DiT attention + GELU + LayerNorm (eager)\n" + "="*78)
try:
    RESULTS['baseline'] = run_experiment(
        tag='baseline',
        build_model=lambda: BaselineCFM(),
        total_steps=10_000,
        lr=1e-4, warmup_steps=0, batch_size=256,
        is_photonflow=False, use_compile=False, use_bf16=False,
    )
    # Final sample grid
    _m = RESULTS['baseline']['model']
    RESULTS['baseline']['sample_png'] = save_sample_grid(_m, 'baseline', False, torch.device('cuda'))
except Exception as e:
    print("[baseline] FAILED"); traceback.print_exc()
    RESULTS['baseline'] = dict(tag='baseline', final_eval=float('nan'),
                               best_eval=float('nan'), n_params=0, wall=0.0,
                               mean_ms_per_step=float('nan'), eval_hist=[], error=str(e))''')


code(r'''# 7. photonflow_eager: S_scale_4M strict photon-native, eager mode
print("\n" + "="*78 + "\n== photonflow_eager: S_scale_4M, eager\n" + "="*78)
try:
    def _build():
        torch.manual_seed(42)
        return PhotonFlowModel(**BASE_ARCH)
    RESULTS['photonflow_eager'] = run_experiment(
        tag='photonflow_eager',
        build_model=_build,
        total_steps=10_000,
        lr=3e-3, warmup_steps=300, batch_size=256,
        is_photonflow=True, use_compile=False, use_bf16=False,
    )
    _m = RESULTS['photonflow_eager']['model']
    RESULTS['photonflow_eager']['sample_png'] = save_sample_grid(_m, 'photonflow_eager', True, torch.device('cuda'))
except Exception as e:
    print("[photonflow_eager] FAILED"); traceback.print_exc()
    RESULTS['photonflow_eager'] = dict(tag='photonflow_eager', final_eval=float('nan'),
                                       best_eval=float('nan'), n_params=0, wall=0.0,
                                       mean_ms_per_step=float('nan'), eval_hist=[], error=str(e))''')


code(r'''# 8. photonflow_opt: S_scale_4M + torch.compile + bf16 autocast
print("\n" + "="*78 + "\n== photonflow_opt: S_scale_4M + torch.compile + bf16 autocast\n" + "="*78)
try:
    def _build():
        torch.manual_seed(42)
        return PhotonFlowModel(**BASE_ARCH)
    RESULTS['photonflow_opt'] = run_experiment(
        tag='photonflow_opt',
        build_model=_build,
        total_steps=10_000,
        lr=3e-3, warmup_steps=300, batch_size=256,
        is_photonflow=True, use_compile=True, use_bf16=True,
    )
    _m = RESULTS['photonflow_opt']['model']
    # For sampling, unwrap torch.compile to avoid graph edge cases
    base_model = _m._orig_mod if hasattr(_m, '_orig_mod') else _m
    RESULTS['photonflow_opt']['sample_png'] = save_sample_grid(base_model, 'photonflow_opt', True, torch.device('cuda'))
except Exception as e:
    print("[photonflow_opt] FAILED"); traceback.print_exc()
    RESULTS['photonflow_opt'] = dict(tag='photonflow_opt', final_eval=float('nan'),
                                     best_eval=float('nan'), n_params=0, wall=0.0,
                                     mean_ms_per_step=float('nan'), eval_hist=[], error=str(e))''')


code(r'''# 9. Summary table + speed-up computation
print("\n" + "="*96)
print("GPU-OPT 10K RESULTS  (uniform-t CFM eval @ 10K steps)")
print("="*96)
hdr = f"{'tag':<20} {'params':>12} {'best_eval':>10} {'gap':>9} {'wall(s)':>9} {'ms/step':>9}"
print(hdr); print("-"*96)
base = RESULTS.get('baseline', {}).get('best_eval', float('nan'))
for tag in ('baseline', 'photonflow_eager', 'photonflow_opt'):
    r = RESULTS.get(tag, {})
    be = r.get('best_eval', float('nan'))
    gap = be - base if base == base else float('nan')
    wall = r.get('wall', float('nan'))
    ms = r.get('mean_ms_per_step', float('nan'))
    n = r.get('n_params', 0)
    print(f"{tag:<20} {n:>12,} {be:>10.4f} {gap:>+9.4f} {wall:>9.1f} {ms:>9.1f}")
print("-"*96)

# Speed-up
eag = RESULTS.get('photonflow_eager', {}).get('mean_ms_per_step', float('nan'))
opt = RESULTS.get('photonflow_opt', {}).get('mean_ms_per_step', float('nan'))
if eag == eag and opt == opt and opt > 0:
    speedup = eag / opt
    print(f"\nSPEED-UP (photonflow_opt vs eager): {speedup:.2f}x  ({eag:.1f} -> {opt:.1f} ms/step)")

# Quality delta
eag_eval = RESULTS.get('photonflow_eager', {}).get('best_eval', float('nan'))
opt_eval = RESULTS.get('photonflow_opt', {}).get('best_eval', float('nan'))
if eag_eval == eag_eval and opt_eval == opt_eval:
    print(f"QUALITY DELTA (opt - eager): {opt_eval - eag_eval:+.4f} (target <= 0.01)")''')


code(r'''# 10. Persist summary.txt + results.json (UTF-8 safe)
import json
def serialisable(r):
    return {k: v for k, v in r.items() if not hasattr(v, 'parameters')}  # drop 'model' nn.Module

with open(OUT / "results.json", 'w', encoding='utf-8') as f:
    j = {}
    for k, r in RESULTS.items():
        j[k] = serialisable(r)
        if 'eval_hist' in j[k]:
            j[k]['eval_hist'] = [list(x) if isinstance(x, tuple) else x for x in j[k]['eval_hist']]
    json.dump(j, f, indent=2)

with open(OUT / "summary.txt", "w", encoding='utf-8') as f:
    base = RESULTS.get('baseline', {}).get('best_eval', float('nan'))
    f.write(f"baseline_best_eval={base:.4f}\n")
    for tag in ('baseline', 'photonflow_eager', 'photonflow_opt'):
        r = RESULTS.get(tag, {})
        be = r.get('best_eval', float('nan'))
        gap = be - base
        wall = r.get('wall', float('nan'))
        ms = r.get('mean_ms_per_step', float('nan'))
        n = r.get('n_params', 0)
        f.write(f"{tag}_best_eval={be:.4f} gap_vs_baseline={gap:+.4f} "
                f"wall={wall:.1f} ms_per_step={ms:.1f} params={n}\n")

with open(OUT / 'summary.txt', 'r', encoding='utf-8') as f:
    print(f.read())''')


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "photonflow_gpu_opt_10k.ipynb"
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out}")
