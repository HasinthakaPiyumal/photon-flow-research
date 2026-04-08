# PhotonFlow

A generative model designed to run on silicon photonic hardware instead of GPUs.

## What this is

PhotonFlow is a research project on photonic neural networks. We are trying to build a flow matching model whose every operation can run directly on a Mach-Zehnder interferometer (MZI) chip, with no electronic offload.

The short version: today's flow matching models use softmax attention and LayerNorm, which photonic chips cannot do natively. We replace those with butterfly-structured linear layers, a saturable-absorber nonlinearity, and a divisive power normalization. All three of these map one-to-one to real photonic components.

We then train with the standard conditional flow matching loss, plus extra noise injection to match what a real chip would see, and finish with 4-bit quantization-aware training.

For the full technical context, read `CLAUDE.md` and the paper draft in `paper/`.

## Folder structure

```
photon-flow-research/
├── photonflow/        Core package (model, training, activation, noise, normalization)
├── hardware/          MZI profiling and quantization-aware training
├── eval/              FID and other metrics
├── experiments/       Numbered experiments (exp1 baseline through exp6 hardware sim)
├── configs/           Experiment configs
├── data/              Datasets (not committed)
├── outputs/           Checkpoints, figures, results (not committed)
├── notebooks/         Jupyter notebooks for exploration
├── paper/             Paper draft and the lit-review Obsidian vault
├── CLAUDE.md          Detailed project context for Claude Code
├── README.md          This file
└── .gitignore
```

Most source files under `photonflow/`, `hardware/`, and `eval/` are still skeletons. We are filling them in as the experiments come together.

## Setup

The project is pure Python and will be PyTorch-based. There is no `requirements.txt` yet because the code is still being written. When you start working on it, expect to need:

- Python 3.10 or newer
- PyTorch
- `torchcfm` for the conditional flow matching loss
- `torchonn` for MZI mesh profiling
- `photontorch` for optical circuit simulation

A rough setup that should work once dependencies are pinned:

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

pip install torch
pip install torchcfm torchonn photontorch
```

We will add a proper `pyproject.toml` or `requirements.txt` once the first experiment runs end to end.

## Setting up Obsidian for the lit review

We keep our literature notes as an Obsidian vault inside the repo at `paper/lit-review/photonflow/`. This way the notes are version controlled with the code.

To open it:

1. Download Obsidian from [obsidian.md](https://obsidian.md) and install it.
2. Open Obsidian. On the start screen, click **Open folder as vault**.
3. Pick the folder `paper/lit-review/photonflow/` from this repo.
4. Obsidian will load the existing notes. Start with `Index.md`.

The vault already has a starting set of notes covering the four papers our work builds on:

- Shen et al. 2017, the first silicon-photonic neural network on real MZI hardware
- Lipman et al. 2023, the original flow matching paper
- Peebles and Xie 2023, the DiT architecture we cannot use directly
- Dao et al. 2022, the Monarch structured matrix that replaces attention in our model

The `.obsidian/` config folder is committed so plugin and appearance settings are shared across the team. Your local UI state (`workspace.json`) is gitignored, so opening or rearranging panes will not show up in `git status`.

## Authors

Undergraduate research by:

- Hasinthaka Piyumal, Faculty of Science, University of Kelaniya, Sri Lanka
- Senumi Costa, Faculty of Computing, University of Plymouth, UK

Built on top of `torchcfm`, `torchonn`, and `photontorch`. Thanks to the maintainers of those projects.
