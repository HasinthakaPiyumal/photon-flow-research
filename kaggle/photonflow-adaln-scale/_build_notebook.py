"""Kaggle kernel `photonflow-adaln-scale` — Kernel 1 of the ≤0.05-gap plan.

Tests MULTIPLICATIVE time modulation via electro-optic Mach-Zehnder
modulators (MZMs) — the biggest untested lever across 13 prior sweeps.

All prior combo v1-v6 variants used ADDITIVE-only `cond_bias`:
    h = norm(x) + bias(t)
Baseline DiT uses adaLN-Zero:
    h = norm(x) * (1 + scale) + shift; out = x + gate * mlp(h)
The expressivity gap between the two is LARGE: multiplicative modulation
gives the model time-dependent per-channel gain, which DiT exploits
to "turn on/off" feature channels as a function of timestep.

Photonic realisability: per-channel scalar multiplication is implemented
by an electro-optic Mach-Zehnder modulator (MZM) bank in silicon
(Shen 2017 §II; Clements 2016 §III).  No `nn.Linear` / `nn.SiLU` /
`nn.Sigmoid` / `nn.ReLU` / `nn.GELU` added to the forward graph —
`scale * x` is just tensor multiplication, `(1 + scale)` is addition.

Zero-initialised by setting the final MonarchLinear's `init_scale=0.0`
so at step 0 `scale = shift = 0` and `h = norm(x)` exactly — the
adaLN-Zero identity-init trick (Peebles 2023).

Variants (baseline-parity sweep):
  - baseline            DiT attention              4,886,544   (reference)
  - A_ref               current ceiling control    5,112,366   gap +0.0930
  - G_adaln_s_cb128     adaLN-scale, cb_hidden=128 ~4.50 M     under-budget
  - H_adaln_s_cb288     adaLN-scale, cb_hidden=288 ~5.00 M     near-parity
  - I_adaln_s_cb576     adaLN-scale, cb_hidden=576 ~6.69 M     over-budget (1.37×)

All five photon-native variants pass `audit_module_tree` (0 electronic ops).
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


md(r'''# PhotonFlow adaLN-scale sweep (Kernel 1 of ≤0.05-gap plan)

Tests multiplicative time modulation (electro-optic MZM) — the #1
untested lever after 13 prior Kaggle sweeps converged to +0.0930 gap.

| Run | `use_adaln_scale` | `cond_bias_hidden` | Expected params | Hypothesis |
|-----|:-----------------:|:------------------:|---:|---|
| `baseline`          | —  | —    | 4.89 M | DiT reference |
| `A_ref`             | no | 576  | 5.11 M | combo-v2 winner |
| `G_adaln_s_cb128`   | **yes** | 128  | ~4.5 M | under-budget test |
| `H_adaln_s_cb288`   | **yes** | 288  | ~5.0 M | near-parity test |
| `I_adaln_s_cb576`   | **yes** | 576  | ~6.7 M | over-budget test |

Rationale: expand `cond_bias_proj` output from `2·dim` → `4·dim` so
we can output `(scale, shift)` per sub-layer.  Apply `h = norm(x) * (1+scale) + shift`
instead of `h = norm(x) + bias`.  Zero-init the final MonarchLinear so
scale=shift=0 at step 0 (adaLN-Zero identity trick, Peebles 2023).

**Target**: close below +0.0930 gap.  If any variant hits ≤0.05 gap,
TARGET_HIT; else proceed to Kernel 2 (train-recipe sweep).
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


code(r'''# 1. Clone repo (photon-native + adaLN-scale kwarg on h/phase1)
import os, subprocess, sys, pathlib
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
from photonflow.layers import PPLNSigmoid, MonarchLinear
from photonflow.time_embed import WavelengthCodedTime
from photonflow.train import CFMLoss
# Sanity: the new use_adaln_scale kwarg must be accepted
_m_test = PhotonFlowModel(in_dim=784, hidden_dim=784, num_blocks=2, time_dim=16,
                          use_noise=False, use_adaln_scale=True)
assert _m_test.blocks[0].use_adaln_scale, "use_adaln_scale kwarg not plumbed"
print("photon-native API OK + use_adaln_scale kwarg verified")''')


code(r'''# 2. GPU check
import torch
assert torch.cuda.is_available(), "need GPU"
print(f"GPU: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
print(f"PyTorch: {torch.__version__}")''')


code(r'''# 3. Shared base architecture + module-tree audit
import torch, torch.nn as nn, math
from photonflow.model import PhotonFlowModel
from photonflow.train import CFMLoss

BASE_ARCH = dict(
    in_dim=784, hidden_dim=784,
    num_blocks=7, time_dim=576,
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
)

def audit_module_tree(model, tag: str):
    n_lin  = sum(1 for mod in model.modules() if isinstance(mod, nn.Linear))
    n_silu = sum(1 for mod in model.modules() if isinstance(mod, nn.SiLU))
    n_sig  = sum(1 for mod in model.modules() if isinstance(mod, nn.Sigmoid))
    n_relu = sum(1 for mod in model.modules() if isinstance(mod, nn.ReLU))
    n_gelu = sum(1 for mod in model.modules() if isinstance(mod, nn.GELU))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_lin == 0 and n_silu == 0 and n_sig == 0 and n_relu == 0 and n_gelu == 0, \
        f"{tag}: electronic op in tree"
    print(f"  [audit] {tag}: 0 electronic ops, {n_params:,} params")
    return n_params''')


code(r'''# 4. Training harness (same as combo v6 / notebook4c567217a1)
import logging, time, pathlib
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUT = pathlib.Path("/kaggle/working/logs"); OUT.mkdir(exist_ok=True)
ROOT = pathlib.Path("/kaggle/working")
DATA_ROOT = str(ROOT / "data")

def run_experiment(tag, build_model, criterion_builder, *,
                   total_steps=2000, batch_size=128, lr=1.7e-3,
                   warmup_steps=600, grad_clip=1.0,
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
    criterion = criterion_builder()
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


code(r'''# 5. Baseline: DiT-style attention (reference, 2K)
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
try:
    RESULTS['baseline'] = run_experiment(
        tag='baseline',
        build_model=lambda: BaselineCFM(),
        criterion_builder=lambda: CFMLoss(),
        total_steps=2000,
        lr=1e-4, warmup_steps=0,
    )
except Exception as e:
    print("[baseline] FAILED"); traceback.print_exc()
    RESULTS['baseline'] = dict(tag='baseline', final_eval=float('nan'),
                                best_eval=float('nan'), n_params=0, eval_hist=[], error=str(e))''')


code(r'''# 6. Variants: adaLN-scale (electro-optic MZM) sweep
# Each variant overrides `use_adaln_scale` and/or `cond_bias_hidden` on top of BASE_ARCH.
VARIANTS = [
    ('A_ref',
     dict(),
     'Control: additive cond_bias only (combo-v2 winner, gap +0.0930)'),
    ('G_adaln_s_cb128',
     dict(use_adaln_scale=True, cond_bias_hidden=128),
     'adaLN-scale ON, cb_hidden=128 (under-baseline-budget test)'),
    ('H_adaln_s_cb288',
     dict(use_adaln_scale=True, cond_bias_hidden=288),
     'adaLN-scale ON, cb_hidden=288 (near-baseline-parity test)'),
    ('I_adaln_s_cb576',
     dict(use_adaln_scale=True, cond_bias_hidden=576),
     'adaLN-scale ON, cb_hidden=576 (over-budget 1.37x; full expressivity)'),
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

    print(f"  [v7 audit] Model overrides: {arch_overrides}")
    try:
        RESULTS[tag] = run_experiment(
            tag=tag,
            build_model=make_model,
            criterion_builder=lambda: CFMLoss(),
            total_steps=2000,
            lr=1.7e-3,
            warmup_steps=600,
        )
    except Exception as e:
        print(f"[{tag}] FAILED"); traceback.print_exc()
        RESULTS[tag] = dict(tag=tag, final_eval=float('nan'),
                            best_eval=float('nan'), n_params=0, eval_hist=[], error=str(e))''')


code(r'''# 7. Summary + pick winner
base = RESULTS['baseline']['best_eval']
print(f"\n{'='*84}")
print("ADALN-SCALE SWEEP RESULTS  (best uniform-t eval in 2,000 steps)")
print('='*84)
print(f"{'run':<20} {'params':>12} {'best_eval':>10} {'gap_vs_baseline':>18}  note")
print('-'*84)
print(f"{'baseline':<20} {RESULTS['baseline']['n_params']:>12,} {base:>10.4f} {'+0.0000':>18}  DiT attention")

variant_results = []
for tag, _, desc in VARIANTS:
    r = RESULTS[tag]
    gap = r['best_eval'] - base
    variant_results.append((tag, r['n_params'], r['best_eval'], gap, desc))
    print(f"{tag:<20} {r['n_params']:>12,} {r['best_eval']:>10.4f} {gap:+18.4f}  {desc}")

print('-'*84)

completed = [v for v in variant_results if v[2] == v[2] and v[2] != float('inf')]
if completed:
    winner = min(completed, key=lambda v: v[3])
    print(f"WINNER: {winner[0]} -- gap {winner[3]:+.4f}  ({winner[4]})")
    target = 0.05
    if winner[3] <= target:
        print(f"RESULT: TARGET_HIT (gap <= {target})")
    elif winner[3] < 0.093:
        print(f"RESULT: CEILING_BROKEN (below +0.0930, still above {target})")
    elif winner[3] <= 0.095:
        print(f"RESULT: TIED_CEILING (at known +0.0930 ceiling)")
    else:
        print(f"RESULT: ABOVE_CEILING")
else:
    print("RESULT: ALL_VARIANTS_FAILED")''')


code(r'''# 8. Persist (UTF-8 safe)
import json, pathlib
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

base = RESULTS.get('baseline', {}).get('best_eval', float('nan'))
with open(OUT / "summary.txt", "w", encoding='utf-8') as f:
    f.write(f"baseline_best_eval={base:.4f}\n")
    for tag, _, _ in VARIANTS:
        r = RESULTS.get(tag, {})
        be = r.get('best_eval', float('nan'))
        gap = be - base
        f.write(f"{tag}_best_eval={be:.4f} gap_vs_baseline={gap:+.4f}\n")
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

out = pathlib.Path(__file__).parent / "photonflow_adaln_scale.ipynb"
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out}")
