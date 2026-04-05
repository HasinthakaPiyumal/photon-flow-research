# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhotonFlow is a research project for photonic neural networks — deep learning models designed for deployment on photonic integrated circuits. Key research areas: optical hardware modeling (Mach-Zehnder Interferometers), hardware-aware training (quantization-aware training), noise characterization in photonic systems, and evaluation on vision tasks.

## Build & Development

No build system is configured yet. The project is pure Python (likely PyTorch-based). All source modules under `photonflow/`, `hardware/`, and `eval/` are currently skeleton files awaiting implementation.

When a build system is added, update this section with install, test, and lint commands.

## Architecture

- **`photonflow/`** — Core package: model definitions (`model.py`), training loop (`train.py`), custom optical activation functions (`activation.py`), photonic noise modeling (`noise.py`), normalization layers (`normalization.py`)
- **`hardware/`** — Hardware simulation: MZI (Mach-Zehnder Interferometer) profiling (`mzi_profiler.py`), quantization-aware training for hardware constraints (`qat.py`)
- **`eval/`** — Evaluation: FID scoring (`fid.py`), other metrics (`metrics.py`)
- **`experiments/`** — Numbered experiment directories (exp1 baseline through exp6 hardware sim)
- **`configs/`** — Experiment configuration files
- **`data/`** — Datasets (not committed)
- **`outputs/`** — Checkpoints, figures, and results (not committed)
- **`paper/`** — Research paper files
- **`notebooks/`** — Jupyter notebooks for exploration

## Key Domain Concepts

- **MZI (Mach-Zehnder Interferometer)**: Core photonic computing element that performs matrix operations optically
- **QAT (Quantization-Aware Training)**: Training technique that accounts for limited precision in photonic hardware
- **FID (Fréchet Inception Distance)**: Metric for evaluating generative model output quality
- Photonic systems have inherent noise characteristics that must be modeled during training
