# PhotonFlow — Implementation Sprint Plan

> **Team:** Hasinthaka (H) + Senumi (S) — **equal responsibility, both cover everything**  
> **Start:** Friday April 11, 2026 — 7:00 PM  
> **End:** Sunday April 13, 2026 — 10:00 AM  
> **Sleep:** 4:00 AM → 8:00 AM both nights (4h × 2 = 8h)  
> **Productive:** ~31h per person | **Total:** ~62 person-hours  
> **Platform:** Google Colab (GPU) + local dev  
> **Approach:** `.py` modules + Colab notebooks  
> **Paper:** NOT in scope — code + experiments only

---

## Schedule

| Block | Time | Hours | Focus |
|-------|------|-------|-------|
| **PHASE 1** | Fri 7:00 PM → Sat 4:00 AM | **9h** | All .py modules + start exp1/exp2 |
| **SLEEP 1** | Sat 4:00 AM → 8:00 AM | 4h | GPU training runs on Colab |
| **PHASE 2** | Sat 8:00 AM → Sun 4:00 AM | **20h** | Experiments + hardware sim + eval + figures |
| **SLEEP 2** | Sun 4:00 AM → 8:00 AM | 4h | Rest |
| **PHASE 3** | Sun 8:00 AM → 10:00 AM | **2h** | Final checks + package |

---

## PHASE 1 — Friday 7:00 PM → Saturday 4:00 AM (9 hours)

*Goal: All .py modules done, exp1 + exp2 training on Colab*

| Time | Hasinthaka | Senumi |
|------|-----------|--------|
| **7:00-7:15** | `pip install`, `__init__.py` files (**photonflow/**, **hardware/**, **eval/**), verify GPU | `requirements.txt`, all 5 config YAMLs, dataset download trigger |
| **7:15-7:45** | **activation.py** — `SaturableAbsorber(nn.Module)`: `tanh(alpha*x)/alpha`, alpha=0.8. ✅ Write test: shape, range, gradients, f(0)=0 | **fid.py Part 1** — `FIDCalculator`: InceptionV3 load, pool3 feature extraction (2048-dim) |
| **7:45-8:15** | **normalization.py** — `DivisivePowerNorm(nn.Module)`: `x / (‖x‖₂ + eps)`. ✅ Write test: unit norm output, zero-input safety, gradients | **fid.py Part 2** — `compute_fid()`: mean/cov stats, Frechet distance formula. ✅ Write test: same-vs-same≈0, real-vs-noise=high |
| **8:15-9:00** | **noise.py** — `PhotonicNoise`: shot noise σ=0.02, thermal crosstalk σ=0.01 (correlated Gaussian). ✅ Write test: train adds noise, eval passes through, noise std matches σ | **metrics.py** — `PhotonicMetrics`: latency = MZI_layers × 10ps × steps, energy = shifters × 10fJ. + `compute_inception_score()` + `compute_precision_recall()`. ✅ Write test: positive values, sensible ranges |
| | | |
| **9:00-9:15** | ☕ *Tea + swap code, quick review* | |
| | | |
| **9:15-10:15** | **model.py: MonarchLayer** — block-diag R/L, permutation P (reshape, transpose, flatten) | **mzi_profiler.py Part 1** — SVD → MZI phase angles (θ,φ), `quantize_phases(bits=4)` |
| **10:15-11:15** | **model.py: PhotonFlowBlock** — MonarchL → MonarchR → Absorber → PowerNorm → +time_embed, zero-init α | **mzi_profiler.py Part 2** — optical loss (0.1dB/stage cumulative), detector noise, thermal crosstalk, full `simulate()` pipeline |
| **11:15-11:45** | **model.py: PhotonFlowModel** — sinusoidal time embed + MLP, stack 6 blocks, final projection | **qat.py** — `FakeQuantize(autograd.Function)`: clamp→round→scale, straight-through backward, `QATWrapper` |
| **11:45-12:00** | ✅ **Test model.py**: random batch=4 dim=784, verify output shape, no NaN, gradients flow through all params, param count check | ✅ **Test hardware**: mzi_profiler — decompose→reconstruct→shapes. qat — fake_quantize ≤16 levels, STE gradient flows |
| | | |
| **12:00-12:30** | 🍜 *Food break + REVIEW: swap model.py ↔ mzi/qat, read each other's code* | |
| | | |
| **12:30-1:15** | **train.py: CFMLoss** — sample t, x0, x1, compute x_t, target=x1-x0, MSE loss | **Notebook 02: exp1** — baseline CFM + **self-attention model** (2-4 heads, 2-4 layers), MNIST, **200K steps** |
| **1:15-2:00** | **train.py: Trainer** — config YAML load, dataloader, Adam lr=1e-4, training loop, noise toggle, QAT toggle, checkpoint every 5K, sample grid every 5K | **Notebook 02 continued** — verify runs, start exp1 on Colab GPU (runs ~5-6h, finishes during SLEEP 1) |
| **2:00-2:30** | ✅ **Test train.py** — 50-100 steps MNIST, verify loss decreases, no NaN, checkpoint saves | **Notebook 03: exp2** — `from photonflow.model import PhotonFlowModel`, PhotonFlow MNIST, **100K steps** |
| **2:30-3:00** | **Notebook 03 continued** — verify first 1K steps, start exp2 on Colab (runs ~3h, finishes during SLEEP 1) | **Review + improve** train.py — edge cases, noise shapes, gradient checks |
| **3:00-3:30** | **Notebook 04: exp3 prep** — noise-injected training ready (σ_s=0.02, σ_t=0.01), **100K steps** | **Notebook 05: exp4 prep** — QAT fine-tune ready (4-bit, load exp3 checkpoint, **10K steps**) |
| **3:30-3:45** | **Git push** — all code to GitHub, verify Colab clone works | **Verify Colab clone** — `!git clone`, install, test imports |
| **3:45-4:00** | **Final check** — exp1 + exp2 running stable on Colab? | Same check |

### 🛌 SLEEP 1 — Saturday 4:00 AM → 8:00 AM
*exp1 (200K steps, ~5-6h) + exp2 (100K steps, ~3h) training on Colab GPU — both should finish by 8 AM*

---

## PHASE 2 — Saturday 8:00 AM → Sunday 4:00 AM (20 hours)

*Goal: Run exp3/4/6, evaluate everything, generate all figures + results*

| Time | Hasinthaka | Senumi |
|------|-----------|--------|
| **8:00-8:15** | Check exp1 + exp2 — finished? Loss curves? | Same — verify both Colab instances |
| **8:15-8:45** | **Evaluate exp2** — generate 10K samples, FID vs exp1 | **Evaluate exp1** — generate 10K samples, FID (baseline number) |
| **8:45-9:15** | **Start exp3** — noise-regularized training, Colab, **100K steps** (~3-4h) | **Notebook 06: exp6** — set up hardware simulation pipeline, ready for exp4 checkpoint |
| **9:15-9:30** | 🍳 *Sync: share FID numbers, discuss — is PhotonFlow working?* | |
| | | |
| **9:30-10:00** | Debug/improve model if exp2 FID bad — adjust blocks (4→8), dims, lr | Debug/improve mzi_profiler if simulation output looks wrong |
| **10:00-10:30** | **Notebook 01: setup** — write proper Colab setup notebook (clone + install + verify) | ✅ **Test qat.py on Colab** — load model, apply QAT wrapper, verify forward/backward, quantized output has ≤16 unique levels |
| | | |
| **10:30-10:45** | ☕ *Tea break* | |
| | | |
| **10:45-11:30** | Monitor exp3 — loss curve, any divergence? | **Improve fid.py** — batch processing, caching, speed optimization |
| **11:30-12:00** | **Generate comparison**: exp1 vs exp2 sample grids side by side | ✅ **Test metrics.py** — compute latency/energy for PhotonFlowModel, verify <1ns latency, <1pJ energy, positive values |
| **12:00-12:30** | Check exp3 — halfway done? Looking good? | Prepare exp4 notebook — double check QAT config, lr schedule |
| | | |
| **12:30-1:00** | 🍜 *Lunch break* | |
| | | |
| **1:00-1:30** | **Evaluate exp3** — FID with noise, compare exp2 vs exp3. **Plot exp2 vs exp3 training curve overlay** | **Start exp4** — load exp3 checkpoint, 4-bit QAT, **10K steps** (~1h) |
| **1:30-2:00** | **Analysis**: exp2 vs exp3 — noise hurt quality? By how much? Worth it? | **Monitor exp4** — QAT tricky, watch for divergence, try lr=1e-5 if needed |
| **2:00-2:30** | Generate exp3 samples grid, loss comparison plot (exp2 vs exp3) | **Evaluate exp4** — FID + **precision** after QAT, compare exp3 vs exp4 |
| **2:30-3:00** | **Run exp6** — load exp4 → MZI sim: decompose→quantize→loss→noise→FID | **Analysis**: exp3 vs exp4 — QAT impact, 4-bit precision acceptable? |
| | | |
| **3:00-3:15** | ☕ *Tea break + sync: ALL experiment FID numbers so far* | |
| | | |
| **3:15-3:45** | **exp6 metrics** — photonic latency (ns/step), energy (fJ/MAC, pJ/sample) | Generate exp4 samples grid, QAT before/after comparison |
| **3:45-4:15** | **Fill Table I** — all FID scores in one table | **Fill Table II** — all hardware metrics in one table |
| **4:15-4:45** | **Figure: loss curves** — all 4 experiments on same plot | **Figure: FID bar chart** — all experiments comparison |
| **4:45-5:15** | **Figure: generated samples grid** — 8×8 best model + 8×8 real images | **Paper Figure 1** — 3-panel: (1) S1→S5 pipeline, (2) PhotonFlowBlock detail, (3) training + regularization diagram |
| **5:15-5:45** | **Figure: Monarch↔MZI mapping** — computation graph comparison (supplementary) | **Figure: noise impact** — exp2 vs exp3 samples + **train curve overlay** |
| | | |
| **5:45-6:15** | 🍕 *Dinner break* | |
| | | |
| **6:15-7:00** | **Notebook 07: results** — collect all plots, tables, sample grids in one notebook | **exp5 CelebA-64 (stretch)** — if Colab GPU available, start with best config |
| **7:00-7:30** | **results_summary.csv** — all metrics one CSV file | Review + run all notebooks top-to-bottom, fix broken cells |
| **7:30-8:00** | **Code cleanup**: docstrings photonflow/*.py, type hints, clean imports | **Code cleanup**: docstrings hardware/*.py + eval/*.py |
| **8:00-8:30** | **Seed reproducibility** — torch.manual_seed(42) everywhere, verify | **README.md update** — install instructions, how to run, results summary |
| | | |
| **8:30-8:45** | ☕ *Tea break* | |
| | | |
| **8:45-9:30** | Review S's code (hardware/, eval/) — any bugs? improvements? | Review H's code (photonflow/) — any bugs? improvements? |
| **9:30-10:00** | Fix review issues in photonflow/ | Fix review issues in hardware/ + eval/ |
| **10:00-10:30** | **Verify**: fresh Colab clone → install → run exp2 100 steps → works? | **Verify**: same from different Colab account |
| **10:30-11:00** | Re-run any failed/incomplete experiments | Re-run any failed/incomplete experiments |
| **11:00-11:30** | **Git push** — all notebooks, figures, results, code cleanup | Verify push — everything present? |
| **11:30-12:00** | Check exp5 CelebA (if started) — any results? | Generate exp5 samples if available |
| | | |
| **12:00-12:30** | 🌙 *Midnight snack + final sync* | |
| | | |
| **12:30-1:00** | **Polish figures** — consistent fonts, axis labels, colors | **Polish notebooks** — markdown cells, clear outputs, rerun clean |
| **1:00-1:30** | **Final results check** — all numbers in Table I/II correct? | **Final code check** — all imports clean, no hardcoded paths |
| **1:30-2:00** | Extra experiments if needed (different block counts, noise levels) | Extra debug/testing if any issues found |
| **2:00-3:00** | **Buffer** — handle anything unexpected | **Buffer** — handle anything unexpected |
| **3:00-3:30** | **Git tag v1.0** — final commit, clean repo | **Verify** — clone, install, test |
| **3:30-4:00** | Final status check — all deliverables complete? | Same |

### 🛌 SLEEP 2 — Sunday 4:00 AM → 8:00 AM

---

## PHASE 3 — Sunday 8:00 AM → 10:00 AM (2 hours)

| Time | Hasinthaka | Senumi |
|------|-----------|--------|
| **8:00-8:30** | End-to-end verify: fresh clone → install → run notebook 03 → output | Same from separate machine |
| **8:30-9:00** | Create submission ZIP: code + notebooks + figures + results | Final README read-through |
| **9:00-9:30** | Any last-minute fixes | Any last-minute fixes |
| **9:30-10:00** | **Final push + done** 🎉 | **Final push + done** 🎉 |

---

## Deliverables Checklist

### .py Modules (12 files)

| # | File | Who Writes | Who Reviews | Status |
|---|------|-----------|-------------|--------|
| 1 | `photonflow/__init__.py` | H | S | [ ] |
| 2 | `photonflow/activation.py` | H | S | [ ] |
| 3 | `photonflow/normalization.py` | H | S | [ ] |
| 4 | `photonflow/noise.py` | H | S | [ ] |
| 5 | `photonflow/model.py` | H+S | Both | [ ] |
| 6 | `photonflow/train.py` | H+S | Both | [ ] |
| 7 | `hardware/__init__.py` | H | S | [ ] |
| 8 | `hardware/mzi_profiler.py` | S+H | Both | [ ] |
| 9 | `hardware/qat.py` | S | H | [ ] |
| 10 | `eval/__init__.py` | H | S | [ ] |
| 11 | `eval/fid.py` | S | H | [ ] |
| 12 | `eval/metrics.py` — includes IS + precision/recall | S | H | [ ] |

### Configs + Notebooks (12 files, renumbered)

| # | File | Who | Status |
|---|------|-----|--------|
| 13 | `configs/exp1_baseline.yaml` (steps=200K) | S | [ ] |
| 14 | `configs/exp2_mnist.yaml` (steps=100K) | S | [ ] |
| 15 | `configs/exp3_noise.yaml` (steps=100K) | S | [ ] |
| 16 | `configs/exp4_qat.yaml` (steps=10K) | S | [ ] |
| 17 | `configs/exp6_hardware.yaml` | S | [ ] |
| 18 | `notebooks/01_setup_and_verify.ipynb` | H | [ ] |
| 19 | `notebooks/02_exp1_baseline.ipynb` (attention model, 200K) | S | [ ] |
| 20 | `notebooks/03_exp2_photonflow_mnist.ipynb` (100K) | H | [ ] |
| 21 | `notebooks/04_exp3_noise_regularized.ipynb` (100K) | H | [ ] |
| 22 | `notebooks/05_exp4_qat_finetune.ipynb` (10K) | S | [ ] |
| 23 | `notebooks/06_exp6_hardware_simulation.ipynb` | S | [ ] |
| 24 | `notebooks/07_results_and_figures.ipynb` | H | [ ] |

### Experiment Results (5 evaluations)

| # | Experiment | Metrics (per Paper Table I) | Who | Status |
|---|-----------|---------------------------|-----|--------|
| 25 | exp1 baseline (200K steps) | FID (reference) | S | [ ] |
| 26 | exp2 PhotonFlow (100K steps) | FID delta vs exp1 | H | [ ] |
| 27 | exp3 +noise (100K steps) | FID + **training curve overlay** vs exp2 | H | [ ] |
| 28 | exp4 +QAT (10K steps) | FID + **precision** | S | [ ] |
| 29 | exp6 photonic sim | ns/step, fJ/MAC | H+S | [ ] |

### Outputs (7 items)

| # | Item | Who | Status |
|---|------|-----|--------|
| 30 | `outputs/figures/loss_curves.png` — all experiments | H | [ ] |
| 31 | `outputs/figures/fid_comparison.png` — bar chart | S | [ ] |
| 32 | `outputs/figures/generated_samples.png` — 8×8 grid | H | [ ] |
| 33 | `outputs/figures/train_curve_exp2_vs_exp3.png` — overlay | H | [ ] |
| 34 | `outputs/figures/figure1_architecture.png` — paper Fig 1 (pipeline + block + training) | S | [ ] |
| 35 | `outputs/results/results_summary.csv` | H | [ ] |
| 36 | `README.md` update | S | [ ] |

**Total: 36 deliverables | H:18 | S:18**

---

## Dependency Chain

```
requirements.txt ──→ pip install ──→ ALL CODE
                                       │
activation.py ──┐                      │
normalization.py┤──→ model.py ──→ train.py ──→ NB03(exp2) ──→ NB04(exp3) ──→ NB05(exp4)
noise.py ───────┘                                                              │
                                                                               │
configs/*.yaml ──→ train.py                                                    │
                                                                               │
fid.py ──────────→ evaluate ALL experiments                                    │
                                                                               │
mzi_profiler.py ─┐                                                             │
qat.py ──────────┤──→ NB06(exp6) ←── exp4 checkpoint ←────────────────────────┘
metrics.py ──────┘         │
                           ↓
                  Results Tables + Figures
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| torchcfm/torchonn install fails | Implement CFM loss manually (MSE), custom MZI sim |
| Training doesn't converge | Lower lr (1e-5), reduce blocks (6→4), MNIST only |
| Colab disconnects | Checkpoint every 5K steps, resume from last |
| Colab GPU quota exhausted | Kaggle notebooks (free GPU alternative) |
| QAT diverges | Skip exp4, report exp3, note QAT as future work |
| FID gap > 10% | Train longer, try block counts 4/6/8, adjust noise σ |
| One person stuck | Sync every 2-3h, pair debug, swap tasks |
| **exp1 200K not done by 8AM** | Check at wake-up; if <80% done, restart on fresh Colab; reduce to 150K if needed |
| **exp5 CelebA-64 not feasible** | Mark as "future work" in paper, remove row from Table I, focus on MNIST results |
| **exp2 100K too slow** | MNIST is small — should be ~3h. If slower, reduce to 50K and note in paper |

---

## Success Criteria

| Metric | Target | Verify With |
|--------|--------|-------------|
| FID | Within 10% of exp1 (200K baseline) | `eval/fid.py` |
| IS (Inception Score) | Report for exp5 CelebA if run | `eval/metrics.py` |
| Precision | Report for exp4 (QAT impact) | `eval/metrics.py` |
| Latency | < 1 ns/step | `eval/metrics.py` |
| Energy | < 1 pJ/sample | `eval/metrics.py` |
| Zero O-E-O | All ops photonic-native | No softmax/LayerNorm/ReLU in model.py |
| Train curves | exp2 vs exp3 overlay shows noise impact | `outputs/figures/train_curve_exp2_vs_exp3.png` |
| Notebooks | All 7 run clean | Verify Sun 8 AM |

---

---

## How to Write Test Cases

**Rule: Never move to the next module until the current one passes all tests.**

Each `.py` module gets a test cell in the same Colab notebook (or a scratch notebook). Tests take 1-2 minutes to run. Total testing overhead: ~30 minutes across the whole sprint.

### Workflow

1. Write the `.py` module
2. Add a test cell in the notebook — import, create dummy input, run asserts
3. Run the cell — if ✅ move on, if ❌ fix immediately
4. Once 2-3 modules are done, test them **together** (integration)

### What to Check in Every Module

| Check | How | Why |
|-------|-----|-----|
| **Output shape** | `assert y.shape == x.shape` or expected shape | Catches dimension bugs early |
| **No NaN / Inf** | `assert not torch.isnan(y).any()` | Catches exploding/vanishing values |
| **Gradients flow** | `y.sum().backward()` then `assert x.grad is not None` | If grads don't flow, training won't work |
| **Edge cases** | Zero input, very large input, single-item batch | Catches divide-by-zero, overflow |
| **Train vs eval mode** | `.train()` vs `.eval()` behave differently where expected | Noise should only apply in training |

### Per-Module Test Checklist

| Module | Key Tests |
|--------|-----------|
| **activation.py** | Output shape = input shape. Output bounded by 1/α. f(0) = 0 (odd function). Gradients flow. |
| **normalization.py** | Output has unit L2 norm per sample. Zero input doesn't crash (eps protects). Gradients flow. |
| **noise.py** | Train mode: output ≠ input. Eval mode: output = input. Noise std ≈ √(σ_s² + σ_t²) ≈ 0.0224. |
| **model.py** | Forward pass shape (batch, 784) → (batch, 784). No NaN. Gradients reach all named params. Param count in sensible range (10K-5M). |
| **train.py** | Run 50-100 steps. Loss decreases. No NaN loss. Checkpoint file created. |
| **fid.py** | Same images → FID ≈ 0. Real vs random noise → FID > 50. |
| **metrics.py** | Latency and energy are positive numbers. Latency < 1 ns for 6 blocks. |
| **mzi_profiler.py** | Weight matrix → phase decomposition → reconstruction. Output shape matches input. |
| **qat.py** | 4-bit fake quantize produces ≤ 16 unique values. Straight-through estimator: gradients flow despite quantization. |

### Integration Tests (after blocks of modules are done)

| When | Integration Test |
|------|-----------------|
| **After activation + norm + noise** | Chain all three: x → absorber → powernorm → noise → check shape + no NaN |
| **After model.py** | Full forward pass with time embedding: model(x, t) → correct shape |
| **After train.py** | 100 steps on MNIST dataloader → loss curve goes down |
| **After mzi + qat** | Load trained model → apply QAT → run MZI sim → get FID, latency, energy numbers |
| **After ALL code** | Fresh Colab: clone → install → import all → run 100 train steps → generate samples → FID |

### Quick Test Template (copy-paste into any notebook cell)

```
# === TEST: [module_name] ===
# 1. Import
# 2. Create dummy input: x = torch.randn(batch, dim)
# 3. Forward pass: y = module(x)
# 4. Assert shape: assert y.shape == expected
# 5. Assert no NaN: assert not torch.isnan(y).any()
# 6. Assert gradients: x.requires_grad_(True); module(x).sum().backward(); assert x.grad is not None
# 7. Print "module_name passed all tests"
```

---

> **Fri 7PM → Sun 10AM | Sleep 4AM-8AM × 2 | 36 deliverables | Steps match Paper Table I | Pure code + experiments** 🚀
