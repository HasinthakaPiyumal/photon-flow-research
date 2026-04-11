# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PhotonFlow** is a generative model co-designed for silicon photonic hardware. The goal is to run flow matching inference on Mach-Zehnder interferometer (MZI) meshes without falling back to electronics for any operation.

The core problem: existing flow matching architectures (CFM, DiT) use softmax attention and LayerNorm. Neither of those runs natively on a photonic chip, so they have to be offloaded to slow electronic circuits, which destroys the speed and energy advantage of optical computing.

PhotonFlow replaces every non-photonic operation in the vector field network with one that maps directly to an MZI primitive:

| Standard component | PhotonFlow replacement | Maps to |
|---|---|---|
| Softmax attention | Monarch (butterfly) linear layers | MZI mesh array |
| LayerNorm | Divisive power normalization `x / (\|\|x\|\|_2 + eps)` | Microring resonator + photodetector feedback |
| ReLU / GELU | Saturable absorber `sigma(x) = tanh(alpha*x) / alpha`, alpha=0.8 | Graphene waveguide insert |

The training objective stays as conditional flow matching:

```
L(theta) = E_{t, x0, x1} [ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ]
```

with two extra regularizers injected after each Monarch layer to bridge the simulation-to-hardware gap:

- Shot noise, sigma_s = 0.02
- Thermal crosstalk, sigma_t = 0.01

After standard training, the model is fine-tuned with 4-bit quantization-aware training to match the 4-to-6-bit effective precision of analog photonic systems.

## Project Status

- **Literature review:** Complete. 9 reference papers analyzed in depth. Obsidian vault at `paper/lit-review/photonflow/`. Comprehensive Sinhala/English explanation in `CLAUDE_READING.md`.
- **Novelty confirmed:** Google Scholar search verified — no prior work combines flow matching + Monarch matrices + photonic hardware. "Monarch matrix MZI photonic" returns only 1 result (unrelated to generation). PhotonFlow's co-design approach is unique vs competitor accelerator approaches (PhotoGAN, DiffLight by Suresh/Afifi/Pasricha group).
- **Implementation:** Sprint in progress (Fri Apr 11 7PM → Sun Apr 13 10AM). See `IMPLEMENTATION_PLAN.md` for detailed timeline.
- **Codebase:** `.py` modules (core logic) + Google Colab notebooks (experiments, training, eval). All source modules currently skeleton files awaiting implementation.

## Known Competitors (from Google Scholar search)

| Paper | Approach | Difference from PhotonFlow |
|---|---|---|
| Suresh et al. "PhotoGAN" (ISQED 2025) | GAN photonic **accelerator** | Accelerates existing GAN arch, still needs O-E-O |
| Suresh et al. "DiffLight" (IEEE D&T 2026) | Diffusion model photonic **accelerator** | Accelerates existing DM arch, still needs O-E-O |
| Suresh et al. "Sustainable Acceleration" (ICCD 2025) | Combined GAN+DM photonic accelerator | Generic accelerator, not co-designed |
| Jiang/Zhu 2026 (already in references) | Optical GAN, 8×8 MNIST | Real chip demo but tiny scale, GAN instability |

**PhotonFlow's unique advantage:** Co-designed architecture (not accelerator) → zero O-E-O conversions.

## Methodology (5 stages)

The paper organizes the work as a five-stage pipeline. Mirror this structure in code and configs:

1. **MZI hardware enumeration** - list the primitives available on chip (butterfly linear, optical power detection, saturable-absorber nonlinearity). Anything outside this list gets excluded.
2. **Architecture co-design** - build the `PhotonFlowBlock` from Monarch L and R, optical activation, divisive power normalization, and time embedding. 6 to 8 blocks form `v_theta(x_t, t)`.
3. **Training** - CFM loss plus shot-noise and thermal-crosstalk regularization, then 4-bit QAT fine-tune.
4. **Photonic simulation** - profile weights in `torchonn`, modeling MZI phase quantization, optical loss (0.1 dB per beamsplitter), and detector noise.
5. **Evaluation** - FID, photonic latency (ns/step), and energy (fJ/MAC, pJ/sample).

## Datasets and targets

- MNIST for sanity checks
- CIFAR-10 as the primary benchmark
- CelebA-64 for higher resolution (stretch goal)

Success criteria from the paper:

- FID within 10% of standard CFM with attention on GPU
- < 1 ns per ODE step in photonic simulation
- < 1 pJ per generated sample

## Baselines to compare against

- **Primary:** standard CFM with softmax attention on GPU
- Optical GAN of Zhu et al. (Frontiers of Optoelectronics, 2026)
- Ablated PhotonFlow without noise regularization
- **New competitors to cite:** PhotoGAN, DiffLight (Suresh/Afifi/Pasricha 2025-2026) — accelerator-based approaches

## Build and development

**Platform:** Google Colab (free GPU for training) + local development for `.py` modules.

**Approach:** Hybrid — `.py` modules contain core logic (importable classes/functions), Colab notebooks run experiments (import from `.py`, train on Colab GPU, visualize inline).

**Dependencies** (`requirements.txt`):

```
torch
torchcfm          # Conditional flow matching objective
torchonn           # MZI mesh profiling and photonic simulation
photontorch        # Optical circuit modeling
torchvision        # MNIST, CIFAR-10 datasets
numpy
scipy
matplotlib
pyyaml
tqdm
jupyter
```

**Colab setup cell** (use in every notebook):

```python
!pip install torchcfm torchonn photontorch
import torch
assert torch.cuda.is_available(), "GPU required"
```

**Running experiments:**

```bash
# Local: test modules
python -c "from photonflow.model import PhotonFlowModel; print('OK')"

# Colab: open notebooks/03_exp2_photonflow_mnist.ipynb
# → Run All cells → training starts on Colab GPU
```

## Architecture (codebase layout)

```
photonflow-research/
├── photonflow/                    ← .py modules (core logic)
│   ├── __init__.py                ← Package init, convenience imports
│   ├── model.py                   ← MonarchLayer, PhotonFlowBlock, PhotonFlowModel
│   ├── activation.py              ← SaturableAbsorber (tanh(αx)/α)
│   ├── normalization.py           ← DivisivePowerNorm (x/‖x‖₂+ε)
│   ├── noise.py                   ← PhotonicNoise (shot σ_s=0.02, thermal σ_t=0.01)
│   └── train.py                   ← CFMLoss, Trainer class, config loading
├── hardware/
│   ├── mzi_profiler.py            ← MZI simulation: SVD→phases, quantize, optical loss
│   └── qat.py                     ← FakeQuantize, StraightThroughEstimator, QATWrapper
├── eval/
│   ├── fid.py                     ← FIDCalculator (InceptionV3 features + Frechet distance)
│   └── metrics.py                 ← PhotonicLatency, PhotonicEnergy estimation
├── configs/
│   ├── exp1_baseline.yaml         ← GPU CFM+attention baseline hyperparams
│   ├── exp2_mnist.yaml            ← PhotonFlow MNIST config
│   ├── exp3_noise.yaml            ← + noise injection params (σ_s, σ_t)
│   ├── exp4_qat.yaml              ← + 4-bit QAT params
│   └── exp6_hardware.yaml         ← Photonic simulation params
├── notebooks/                     ← Colab notebooks (experiments + visualization)
│   ├── 01_setup_and_verify.ipynb          ← Install deps, verify GPU, test imports
│   ├── 02_exp1_baseline.ipynb             ← Train baseline CFM+attention on MNIST
│   ├── 03_exp2_photonflow_mnist.ipynb     ← Train PhotonFlow on MNIST
│   ├── 04_exp3_noise_regularized.ipynb    ← Train with noise injection
│   ├── 05_exp4_qat_finetune.ipynb         ← 4-bit QAT fine-tuning
│   ├── 06_exp6_hardware_simulation.ipynb  ← MZI profiling + photonic metrics
│   └── 07_results_and_figures.ipynb       ← All plots, tables, sample grids for paper
├── paper/
│   ├── PAPER_DRAFT.md             ← Research paper draft
│   └── lit-review/                ← Obsidian vault with 9 reference paper notes
│       ├── photonflow/            ← Markdown notes per paper
│       └── pdfs/                  ← Reference PDFs
├── data/                          ← Datasets (not committed, auto-downloaded)
├── outputs/
│   ├── checkpoints/               ← Model checkpoints (.pth)
│   ├── figures/                   ← Generated plots, sample grids (.png/.pdf)
│   └── results/                   ← Metrics CSV, results summary
├── requirements.txt
├── CLAUDE.md                      ← This file
├── CLAUDE_READING.md              ← Comprehensive Sinhala/English project explanation
├── IMPLEMENTATION_PLAN.md         ← 35h sprint timeline with task assignments
└── README.md
```

## Experiments (Paper Table I)

| Exp | Configuration | Steps | Dataset | Purpose |
|---|---|---|---|---|
| exp1 | Standard CFM + attention on GPU | 50K | MNIST | Baseline FID reference |
| exp2 | PhotonFlow (Monarch layers) | 50K | MNIST | Sanity check — "do Monarch layers work for generation?" |
| exp3 | exp2 + shot noise + thermal crosstalk | 50K | MNIST | Noise robustness — "does noise-aware training help?" |
| exp4 | exp3 + 4-bit QAT fine-tune | 5-10K | MNIST | Hardware precision — "does 4-bit QAT preserve quality?" |
| exp5 | Best config on CelebA-64 | TBD | CelebA-64 | Scaling (stretch goal) |
| exp6 | Photonic hardware simulation via torchonn | — | — | Performance — "what are photonic latency/energy numbers?" |

Experiments run via Colab notebooks (`notebooks/02-06`). Results collected in `notebooks/07_results_and_figures.ipynb`.

## Key domain concepts

- **MZI (Mach-Zehnder Interferometer):** the core photonic computing element. Light splits via beamsplitter, one path gets phase-shifted, then recombines. Mathematically a 2×2 unitary matrix. A cascade of MZI beamsplitters performs a sequence of two-by-two unitary rotations at the speed of light. This is exactly the computational graph of a Monarch matrix, which is why PhotonFlow uses Monarch layers.
- **Monarch matrix:** structured matrix of the form `M = P L P^T R`, where L and R are block-diagonal and P is a fixed permutation (stride/perfect shuffle). From Dao et al. 2022. In PhotonFlow each Monarch layer pair replaces self-attention. Parameters: O(n√n) vs O(n²) for dense. FLOPs: O(n^{3/2}). Key insight: Monarch computation graph = MZI mesh computation graph (block-diagonal = MZI column, permutation = waveguide routing = free).
- **Saturable absorber:** the photonic analog of a nonlinear activation. A graphene waveguide absorbs low-intensity light but becomes transparent at high intensity. Acts like `tanh(alpha*x)/alpha`. This is the only nonlinearity allowed in the architecture.
- **Divisive power normalization:** photonic analog of LayerNorm. A photodetector measures total optical power (L2 norm), a microring resonator feedback loop divides by it. `x / (||x||_2 + eps)`. No mean/variance computation needed.
- **Shot noise:** quantum noise on photon counts at the detector. Photons arrive as discrete particles with inherent randomness (like raindrops on a window). Modeled as additive Gaussian with sigma_s = 0.02 during training.
- **Thermal crosstalk:** unwanted heat coupling between adjacent phase shifters. When one heater warms up, neighbors shift too. Modeled as correlated additive Gaussian with sigma_t = 0.01.
- **QAT (Quantization-Aware Training):** training technique that accounts for limited precision in photonic hardware. MZI phase shifters have 4-6 bit effective precision (16-64 discrete angles). QAT inserts fake quantization nodes in forward pass, uses straight-through estimator for gradients. PhotonFlow targets 4-bit weights. Two-stage strategy: (1) float32 + noise training → convergence, (2) 4-bit QAT fine-tune 5-10K steps.
- **CFM (Conditional Flow Matching):** the training objective from Lipman et al. 2023. Learns a vector field `v_theta(x_t, t)` that transports noise to data along straight (optimal transport) paths. Loss = MSE regression, architecture-agnostic, stable training, few ODE steps at inference.
- **FID (Frechet Inception Distance):** standard metric for generative image quality. Compares InceptionV3 feature distributions between real and generated images. Lower is better. Computed via `eval/fid.py`.
- **O-E-O conversion:** Opto-Electronic-Opto — converting light to electricity and back. The #1 bottleneck in photonic computing. Each conversion adds nanoseconds of latency (vs picosecond MZI computation). PhotonFlow eliminates all O-E-O by using only photonic-native operations.
- **Optical loss:** light power lost when passing through beamsplitters. ~0.1 dB per stage (~2.3% power loss). Accumulates through MZI cascade.

## Implementation notes

- **Model architecture:** `PhotonFlowModel` = 6 `PhotonFlowBlock` stacked. Each block: MonarchL → MonarchR → SaturableAbsorber → DivisivePowerNorm → +TimeEmbed. Time embedding: sinusoidal + 2-layer MLP. Zero-initialized skip connections (α=0 trick from DiT).
- **Training:** CFM loss via `torchcfm` or manual MSE implementation. Adam optimizer lr=1e-4. Noise injection toggle from config. Sample generation every 5K steps (Euler ODE solver, 20 steps).
- **Hardware simulation:** SVD/Clements decomposition → MZI phase angles → phase quantization (4-bit) → optical loss injection (0.1 dB/stage cumulative) → detector noise → thermal crosstalk (correlated Gaussian).
- **Evaluation:** FID via InceptionV3 pool3 features (2048-dim), Frechet distance formula. Latency = MZI_layers × propagation_delay × ODE_steps. Energy = phase_shifters × energy_per_shifter + detector_energy.

## Team and sprint info

Undergraduate research by **Hasinthaka Piyumal** (University of Kelaniya, Sri Lanka) and **Senumi Costa** (University of Plymouth, UK).

**Sprint:** Fri Apr 11 7PM → Sun Apr 13 10AM (35h productive, 4h sleep). Equal workload, both members work on all areas. See `IMPLEMENTATION_PLAN.md` for detailed schedule.

Built on the open-source `torchcfm`, `torchonn`, and `photontorch` projects.
