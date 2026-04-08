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
- CelebA-64 for higher resolution

Success criteria from the paper:

- FID within 10% of standard CFM with attention on GPU
- < 1 ns per ODE step in photonic simulation
- < 1 pJ per generated sample

## Baselines to compare against

- **Primary:** standard CFM with softmax attention on GPU
- Optical GAN of Zhu et al. (Frontiers of Optoelectronics, 2026)
- Ablated PhotonFlow without noise regularization

## Build and development

No build system is configured yet. The project is pure Python and will be PyTorch-based. All source modules under `photonflow/`, `hardware/`, and `eval/` are currently skeleton files awaiting implementation.

Expected dependencies (from the paper acknowledgments):

- `torch`
- `torchcfm` for the conditional flow matching objective
- `torchonn` for MZI mesh profiling and photonic simulation
- `photontorch` for optical circuit modeling

When a build system is added, update this section with install, test, and lint commands.

## Architecture (codebase layout)

- **`photonflow/`** Core package: model definitions (`model.py`), training loop (`train.py`), custom optical activation functions (`activation.py`), photonic noise modeling (`noise.py`), normalization layers (`normalization.py`)
- **`hardware/`** Hardware simulation: MZI profiling (`mzi_profiler.py`), quantization-aware training for hardware constraints (`qat.py`)
- **`eval/`** Evaluation: FID scoring (`fid.py`), other metrics (`metrics.py`)
- **`experiments/`** Numbered experiment directories. The paper has six experiments in Table I:
  - exp1: standard CFM + attention on GPU baseline (200K steps)
  - exp2: PhotonFlow on MNIST (100K steps)
  - exp3: exp2 + shot noise + thermal crosstalk
  - exp4: exp3 + 4-bit QAT (10K fine-tune)
  - exp5: best config on CelebA-64
  - exp6: photonic hardware simulation via `torchonn`
- **`configs/`** Experiment configuration files
- **`data/`** Datasets (not committed)
- **`outputs/`** Checkpoints, figures, and results (not committed)
- **`paper/`** Research paper files. The literature review is kept as an Obsidian vault at `paper/lit-review/photonflow/`. See `README.md` for how to open it.
- **`notebooks/`** Jupyter notebooks for exploration

## Key domain concepts

- **MZI (Mach-Zehnder Interferometer):** the core photonic computing element. A cascade of MZI beamsplitters performs a sequence of two-by-two unitary rotations at the speed of light. This is exactly the computational graph of a Monarch matrix, which is why PhotonFlow uses Monarch layers.
- **Monarch matrix:** structured matrix of the form `M = P L P^T R`, where L and R are block-diagonal and P is a fixed permutation. From Dao et al. 2022. In PhotonFlow each Monarch layer pair replaces self-attention.
- **Saturable absorber:** the photonic analog of a nonlinear activation. A graphene waveguide acts like `tanh(alpha*x)/alpha`. This is the only nonlinearity allowed in the architecture.
- **Divisive power normalization:** photonic analog of LayerNorm. A photodetector measures total power and a microring resonator divides by it.
- **Shot noise:** quantum noise on photon counts at the detector. Modeled as additive Gaussian with sigma_s = 0.02 during training.
- **Thermal crosstalk:** unwanted heat coupling between adjacent phase shifters. Modeled as additive Gaussian with sigma_t = 0.01.
- **QAT (Quantization-Aware Training):** training technique that accounts for limited precision in photonic hardware. PhotonFlow targets 4-bit weights.
- **CFM (Conditional Flow Matching):** the training objective from Lipman et al. 2023. Learns a vector field `v_theta(x_t, t)` that transports noise to data along straight paths.
- **FID (Frechet Inception Distance):** standard metric for generative image quality. Lower is better.

## Authors and acknowledgments

Undergraduate research by Hasinthaka Piyumal (University of Kelaniya, Sri Lanka) and Senumi Costa (University of Plymouth, UK). Built on the open-source `torchcfm`, `torchonn`, and `photontorch` projects.
