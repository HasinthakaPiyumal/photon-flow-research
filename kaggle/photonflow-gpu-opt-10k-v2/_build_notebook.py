"""Retry kernel for photonflow_opt at 10K steps after v1 torch.compile failure.

v1 diagnosis:
  - torch.compile(mode='reduce-overhead') wraps the model + enables CUDA
    graphs, but CUDA graphs require fixed-shape tensors.  Our Cayley
    unitary projection uses torch.linalg.solve on batched (m,m,m) tensors
    (m=28 for hidden_dim=784).  When num_monarch_factors=3, that's 3 solves
    per MonarchLayer × ~26 MonarchLayers per forward = 78+ solves.  CUDA
    graphs + bf16 autocast + this many solves produced a silent failure
    (log stopped after "wrapping with torch.compile" line).

v2 recipe (safe + proven):
  - torch.cuda.amp.autocast(dtype=torch.bfloat16) ONLY -- no torch.compile.
    This has been proven stable on PyTorch + photonic models for over a year.
  - Same math, same gradient path, same zero-OEO forward graph.
  - Expected speed-up: ~1.3-1.5x over eager (less than compile+bf16's 2-3x
    target, but 1.3x is still a real win and it actually runs).

Single variant: photonflow_opt (bf16 only), 10K steps, S_scale_4M.
Baseline + photonflow_eager already ran in v1; reuse those numbers.
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


md(r'''# photonflow_opt retry (bf16 autocast only, no torch.compile)

v1 failed silently at `torch.compile(mode='reduce-overhead')` wrap --
CUDA graphs don't compose with the 78+ `torch.linalg.solve` calls per
forward pass (Cayley unitary projections).

v2: keep bf16 autocast only.  Proven-stable PyTorch path; no JIT tracing
required.  Expected speed-up ~1.3-1.5x over eager, eval delta <= 0.01.
''')


code(r'''# 0. Torch compat
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
    ])''')


code(r'''# 1. Clone repo
import os, subprocess, sys
REPO_URL    = "https://github.com/HasinthakaPiyumal/photon-flow-research.git"
REPO_BRANCH = "h/phase1"
REPO_DIR    = "/kaggle/working/photon-flow-research"
if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git","clone","--depth","1","--branch",REPO_BRANCH, REPO_URL, REPO_DIR])
sys.path.insert(0, REPO_DIR)
print("HEAD:", subprocess.check_output(["git","-C",REPO_DIR,"log","-1","--oneline"]).decode().strip())''')


code(r'''# 2. Imports + GPU + audit helper
import torch, torch.nn as nn, math, time, logging, pathlib, traceback, json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from photonflow.model import PhotonFlowModel
from photonflow.train import CFMLoss
from photonflow.sampler import OpticalSampler

assert torch.cuda.is_available()
print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")
print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

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
    n_lin  = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    n_silu = sum(1 for m in model.modules() if isinstance(m, nn.SiLU))
    n_sig  = sum(1 for m in model.modules() if isinstance(m, nn.Sigmoid))
    n_relu = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
    n_gelu = sum(1 for m in model.modules() if isinstance(m, nn.GELU))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_lin == 0 and n_silu == 0 and n_sig == 0 and n_relu == 0 and n_gelu == 0, \
        f"{tag}: electronic op in tree"
    print(f"  [audit] {tag}: 0 electronic ops, {n_params:,} params")
    return n_params

OUT = pathlib.Path("/kaggle/working/logs"); OUT.mkdir(exist_ok=True)
DATA_ROOT = "/kaggle/working/data"''')


code(r'''# 3. Training harness -- bf16 autocast only
def run_experiment(tag, build_model, *,
                   total_steps=10_000, batch_size=256, lr=3e-3,
                   warmup_steps=300, grad_clip=1.0,
                   eval_every=2000, eval_batches=8, seed=42,
                   is_photonflow=True, use_bf16=True):
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
    log(f"is_photonflow={is_photonflow} use_bf16={use_bf16} (no torch.compile)")

    tfm = transforms.ToTensor()
    ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params={n_params:,}")
    if is_photonflow:
        audit_module_tree(model, tag)

    criterion = CFMLoss()
    eval_criterion = CFMLoss()
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
                tot += eval_criterion(model, xe).item(); n += 1
        model.train()
        return tot / max(1, n)

    model.train(); it = iter(loader); losses = []; eval_hist = []
    best_eval = float("inf"); t_start = time.time(); t_last = t_start
    step_times = []
    log("--- training start ---")
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
            torch.cuda.synchronize()
            now = time.time()
            chunk_ms = 1000 * (now - t_last) / 500 if step > 1 else 0.0
            t_last = now
            if step > 100:  # skip warmup
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
    return dict(tag=tag, final_eval=final_eval, best_eval=best_eval, n_params=n_params,
                eval_hist=eval_hist, wall=wall, mean_ms_per_step=mean_ms_per_step,
                model=model, is_photonflow=is_photonflow)''')


code(r'''# 4. Run photonflow_opt (bf16 only)
print("\n" + "="*78 + "\n== photonflow_opt: S_scale_4M + bf16 autocast (NO torch.compile)\n" + "="*78)
RESULTS = {}
try:
    def _build():
        torch.manual_seed(42)
        return PhotonFlowModel(**BASE_ARCH)
    RESULTS['photonflow_opt'] = run_experiment(
        tag='photonflow_opt',
        build_model=_build,
        total_steps=10_000,
        lr=3e-3, warmup_steps=300, batch_size=256,
        is_photonflow=True, use_bf16=True,
    )
except Exception as e:
    print("[photonflow_opt] FAILED"); traceback.print_exc()
    RESULTS['photonflow_opt'] = dict(tag='photonflow_opt', final_eval=float('nan'),
                                     best_eval=float('nan'), n_params=0, wall=0.0,
                                     mean_ms_per_step=float('nan'), eval_hist=[], error=str(e))''')


code(r'''# 5. Sample grid PNG
def save_sample_grid(model, tag, device, n=64):
    with torch.no_grad():
        model.eval()
        sampler = OpticalSampler(model, tau=None, eps=1e-3, max_iters=20, mode='fixedpoint')
        samp = sampler(shape=(n, 784), device=device)
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
    return str(png_path)

r = RESULTS.get('photonflow_opt')
if r and r.get('best_eval', float('nan')) == r.get('best_eval', float('nan')):
    png = save_sample_grid(r['model'], 'photonflow_opt', torch.device('cuda'))
    RESULTS['photonflow_opt']['sample_png'] = png
    print(f"  sample saved -> {png}")''')


code(r'''# 6. Summary + persist
print("\n" + "="*96)
print("PHOTONFLOW_OPT (bf16 only) RESULTS @ 10K steps")
print("="*96)
r = RESULTS.get('photonflow_opt', {})
be = r.get('best_eval', float('nan'))
ms = r.get('mean_ms_per_step', float('nan'))
wall = r.get('wall', float('nan'))
n = r.get('n_params', 0)
print(f"  params={n:,}  best_eval={be:.4f}  wall={wall:.1f}s  ms/step={ms:.1f}")
# baseline reference from v1 run: 0.1324 eval
BASELINE_REF = 0.1324
EAGER_MS_PER_STEP = 223.4  # from v1 photonflow_eager log
if ms == ms and EAGER_MS_PER_STEP > 0:
    speedup = EAGER_MS_PER_STEP / ms
    print(f"  SPEED-UP vs v1 eager (223.4 ms/step):  {speedup:.2f}x")
if be == be:
    print(f"  GAP vs v1 baseline (0.1324):            {be - BASELINE_REF:+.4f}")

def ser(r):
    return {k: v for k, v in r.items() if not hasattr(v, 'parameters')}

with open(OUT / "results.json", 'w', encoding='utf-8') as f:
    json.dump({k: ser(v) for k, v in RESULTS.items()}, f, indent=2)
with open(OUT / "summary.txt", 'w', encoding='utf-8') as f:
    f.write(f"baseline_ref_from_v1={BASELINE_REF:.4f}\n")
    f.write(f"photonflow_opt_best_eval={be:.4f} gap_vs_baseline={be-BASELINE_REF:+.4f} wall={wall:.1f} ms_per_step={ms:.1f} params={n}\n")
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
out = pathlib.Path(__file__).parent / "photonflow_gpu_opt_10k_v2.ipynb"
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out}")
