# Hu et al. 2024 -- Lateralization MLP (L-MLP)

**Full title:** Lateralization MLP: A Simple Brain-inspired Architecture for Diffusion
**Authors:** Z. Hu, H. Chen, J. Lu
**Venue:** arXiv 2024
**Link:** https://arxiv.org/abs/2405.16098

## What this paper does

Proposes L-MLP, a brain-inspired MLP architecture for diffusion models. Each L-MLP block:
1. Permutes data dimensions
2. Processes each dimension in parallel (left/right "hemispheres")
3. Merges them
4. Passes through a joint MLP

## Key results

- First fully-MLP architecture demonstrated on cross-modal text-to-image diffusion
- Matches transformer-based diffusion architectures in quality
- More computationally efficient than attention-based alternatives

## What PhotonFlow uses from this paper

Validates that **permutation + parallel processing** (exactly the Monarch computation pattern) is sufficient for generative diffusion/flow tasks. Supports PhotonFlow's approach of using structured permutation-based mixing instead of attention. Cited as evidence that attention-free generation is feasible.
