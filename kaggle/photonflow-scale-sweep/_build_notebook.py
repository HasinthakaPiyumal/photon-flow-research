"""Kaggle kernel `photonflow-scale-sweep` — scale the best architecture.

Takes the K_bs256 winning config (adaLN-scale + bs=256 + lr=3e-3 +
warmup=300 + Adam, which hit gap +0.0435 at 6.69 M params) and varies
`num_blocks` to land on THREE parameter budgets — all strict photon-native
(0 electronic ops in forward graph), all 2K MNIST CFM steps.

  - S_scale_4M   num_blocks=5   ->   4,867,082 params  (baseline parity)
  - T_scale_9M   num_blocks=10  ->   9,414,292 params
  - U_scale_15M  num_blocks=16  ->  14,870,944 params

NO baseline in this kernel (we already have reference from prior sweeps:
baseline DiT attention = 0.1734 uniform-t CFM at 2K steps).

Each variant passes `audit_module_tree` before training:
0 nn.Linear / 0 nn.SiLU / 0 nn.Sigmoid / 0 nn.ReLU / 0 nn.GELU.
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


md(r'''# PhotonFlow scale sweep (best-arch + 3 param budgets)

The ≤ 0.05 target was hit by K_bs256 (Kernel 2) at **+0.0435 gap**
on the K_bs256 winning recipe: adaLN-scale + bs=256 + lr=3e-3 + wu=300.
That winner was 6.69 M params (1.37× baseline).  This kernel scales
the same architecture to three param budgets by varying `num_blocks`:

| Variant | `num_blocks` | Params | ×baseline | Hypothesis |
|---|---:|---:|---:|---|
| `S_scale_4M`  | 5  |  4.87 M | 1.00× | does baseline-parity preserve target? |
| `T_scale_9M`  | 10 |  9.41 M | 1.93× | does more depth close the gap further? |
| `U_scale_15M` | 16 | 14.87 M | 3.04× | stretch: diminishing returns? |

Baseline (4.89 M DiT attention) = 0.1734 CFM uniform-t eval (from prior
kernels; NOT re-trained here to save GPU budget).

All three variants share the Kernel-2 winning config:
  `num_monarch_factors=3, time_dim=576, hidden_dim=784,
   cond_bias_hidden=576, use_adaln_scale=True, adaln_init_std=0.5,
   learnable_absorber_alpha=True, absorber_leaky_slope=0.05,
   use_noise=False, monarch_init='random'`
Training: bs=256, lr=3e-3, warmup=300, Adam, 2000 steps, grad_clip=1.0.
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


code(r'''# 1. Clone repo (h/phase1 -- adaLN-scale commit 6701f5e is live)
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
from photonflow import PhotonFlowModel
from photonflow.train import CFMLoss
_m = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=2, time_dim=16,
                     use_noise=False, use_adaln_scale=True)
assert _m.blocks[0].use_adaln_scale, "use_adaln_scale kwarg missing"
print("photon-native API + adaLN-scale verified")''')


code(r'''# 2. GPU check
import torch
assert torch.cuda.is_available(), "need GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}")''')


code(r'''# 3. Shared base architecture + module-tree audit
import torch, torch.nn as nn, math
from photonflow.model import PhotonFlowModel
from photonflow.train import CFMLoss

# K_bs256 winning arch template (Kernel 2 winner), num_blocks varies per variant
BASE_ARCH = dict(
    in_dim=784, hidden_dim=784, time_dim=576,
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

def audit_module_tree(model, tag: str):
    n_lin  = sum(1 for mod in model.modules() if isinstance(mod, nn.Linear))
    n_silu = sum(1 for mod in model.modules() if isinstance(mod, nn.SiLU))
    n_sig  = sum(1 for mod in model.modules() if isinstance(mod, nn.Sigmoid))
    n_relu = sum(1 for mod in model.modules() if isinstance(mod, nn.ReLU))
    n_gelu = sum(1 for mod in model.modules() if isinstance(mod, nn.GELU))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_lin == 0 and n_silu == 0 and n_sig == 0 and n_relu == 0 and n_gelu == 0, \
        f"{tag}: electronic op in forward graph"
    print(f"  [audit] {tag}: 0 electronic ops, {n_params:,} params")
    return n_params''')


code(r'''# 4. Training harness (K_bs256 recipe: Adam, bs=256, lr=3e-3, wu=300)
import logging, time, pathlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT = pathlib.Path("/kaggle/working/logs"); OUT.mkdir(exist_ok=True)
ROOT = pathlib.Path("/kaggle/working")
DATA_ROOT = str(ROOT / "data")

def run_experiment(tag, build_model, *,
                   total_steps=2000, batch_size=256, lr=3e-3,
                   warmup_steps=300, grad_clip=1.0,
                   eval_every=500, eval_batches=8, seed=42):
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

    tfm = transforms.ToTensor()
    ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params={n_params:,}")
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
    best_eval = float("inf"); t0 = time.time(); t_last = t0
    log("--- training start ---")
    for step in range(1, total_steps + 1):
        try: x, _ = next(it)
        except StopIteration: it = iter(loader); x, _ = next(it)
        x = x.view(x.size(0), -1).to(device, non_blocking=True)
        loss = criterion(model, x)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step(); scheduler.step()
        losses.append(loss.item())
        if step % 200 == 0 or step == 1:
            a50 = sum(losses[-50:])/min(50,len(losses))
            now = time.time(); ms = 1000*(now - t_last)/200 if step>1 else 0.0
            t_last = now
            logkv(step=f"{step:4d}/{total_steps}",
                  loss=f"{loss.item():.4f}", avg50=f"{a50:.4f}",
                  lr=f"{scheduler.get_last_lr()[0]:.2e}", gnorm=f"{gnorm:.3f}",
                  ms=f"{ms:.1f}", elap=f"{now-t0:.0f}s")
        if step % eval_every == 0:
            ev = compute_eval(); eval_hist.append((step, ev))
            best_eval = min(best_eval, ev)
            log(f"[EVAL] step={step} uniform_t_loss={ev:.4f} best={best_eval:.4f}")

    final_eval = eval_hist[-1][1] if eval_hist else float("nan")
    log(f"[SUMMARY] tag={tag} params={n_params:,}")
    log(f"[SUMMARY] final_eval_uniform_t={final_eval:.4f} best_eval={best_eval:.4f}")
    log(f"[SUMMARY] wall={(time.time()-t0):.1f}s")
    return dict(tag=tag, final_eval=final_eval, best_eval=best_eval,
                n_params=n_params, eval_hist=eval_hist)''')


code(r'''# 5. Scale sweep: 3 param budgets (no baseline -- prior reference = 0.1734)
import traceback
RESULTS = {}
BASELINE_EVAL = 0.1734  # reference from notebook4c567217a1 and all prior kernels

VARIANTS = [
    ('S_scale_4M',  dict(num_blocks=5),  'baseline-parity (4.87M params, 1.00x baseline)'),
    ('T_scale_9M',  dict(num_blocks=10), '9M (9.41M params, 1.93x baseline)'),
    ('U_scale_15M', dict(num_blocks=16), '15M (14.87M params, 3.04x baseline)'),
]

for tag, arch_overrides, desc in VARIANTS:
    print(f"\n{'='*78}\n== {tag}: {desc}\n{'='*78}")
    mk = dict(BASE_ARCH)
    for k, v in arch_overrides.items():
        mk[k] = v

    def make_model(mk=mk):
        torch.manual_seed(42)
        return PhotonFlowModel(**mk)

    m = make_model()
    audit_module_tree(m, tag)
    del m

    print(f"  [audit] Arch overrides: {arch_overrides}")
    try:
        RESULTS[tag] = run_experiment(
            tag=tag,
            build_model=make_model,
            total_steps=2000,
            batch_size=256,
            lr=3e-3,
            warmup_steps=300,
        )
    except Exception as e:
        print(f"[{tag}] FAILED"); traceback.print_exc()
        RESULTS[tag] = dict(tag=tag, final_eval=float('nan'),
                            best_eval=float('nan'), n_params=0, eval_hist=[], error=str(e))''')


code(r'''# 6. Summary
base = BASELINE_EVAL
print(f"\n{'='*84}")
print(f"SCALE SWEEP RESULTS  (baseline reference = {base:.4f} from prior kernels)")
print('='*84)
print(f"{'run':<15} {'params':>12} {'best_eval':>10} {'gap_vs_baseline':>18}  note")
print('-'*84)

variant_results = []
for tag, _, desc in VARIANTS:
    r = RESULTS[tag]
    gap = r['best_eval'] - base
    variant_results.append((tag, r['n_params'], r['best_eval'], gap, desc))
    print(f"{tag:<15} {r['n_params']:>12,} {r['best_eval']:>10.4f} {gap:+18.4f}  {desc}")
print('-'*84)

completed = [v for v in variant_results if v[2] == v[2] and v[2] != float('inf')]
if completed:
    winner = min(completed, key=lambda v: v[3])
    print(f"WINNER: {winner[0]} -- gap {winner[3]:+.4f}")
    target = 0.05
    below_target = [v for v in variant_results if v[3] <= target]
    if below_target:
        names = ", ".join(v[0] for v in below_target)
        print(f"TARGET_HIT on: {names} (gap <= {target})")
    else:
        print(f"NO_TARGET_HIT (best {winner[3]:+.4f})")
else:
    print("RESULT: ALL_VARIANTS_FAILED")''')


code(r'''# 7. Persist (UTF-8 safe)
import json
def serialisable(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, list):
            out[k] = [list(x) if isinstance(x, tuple) else x for x in v]
        else:
            out[k] = v
    return out

with open(OUT / "results.json", 'w', encoding='utf-8') as f:
    json.dump({k: serialisable(v) for k, v in RESULTS.items()}, f, indent=2)

with open(OUT / "summary.txt", "w", encoding='utf-8') as f:
    f.write(f"baseline_reference_eval={BASELINE_EVAL:.4f}\n")
    for tag, _, _ in VARIANTS:
        r = RESULTS.get(tag, {})
        be = r.get('best_eval', float('nan'))
        gap = be - BASELINE_EVAL
        f.write(f"{tag}_best_eval={be:.4f} gap_vs_baseline={gap:+.4f} params={r.get('n_params',0)}\n")
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

out = pathlib.Path(__file__).parent / "photonflow_scale_sweep.ipynb"
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out}")
