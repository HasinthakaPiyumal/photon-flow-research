# Dao et al. 2023 -- Monarch Mixer (M2)

**Full title:** Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture
**Authors:** D. Fu, S. Arora, J. Grber, I. Johnson, D. Hong, A. Rudra, C. Re
**Venue:** NeurIPS 2023 (Oral)
**Link:** https://arxiv.org/abs/2310.12109

## What this paper does

Replaces BOTH attention AND MLPs in Transformers with a single sub-quadratic primitive: Monarch matrices. The resulting architecture (M2) uses Monarch along two axes:
- **Sequence mixer:** Monarch matrices along the sequence dimension (replaces attention)
- **Dimension mixer:** Block-diagonal matrices along the model dimension (replaces MLP)

## Key results

- M2-BERT: 25% fewer params/FLOPs than BERT, matches GLUE quality
- M2-GPT: matches GPT-style Transformers at 360M params on The Pile
- First demonstration that Transformer quality is achievable without attention or MLPs

## What PhotonFlow uses from this paper

The **two-axis mixing principle**: applying Monarch along both spatial and feature dimensions. PhotonFlow reshapes the flat 784-dim MNIST vector to (49, 16) and applies:
- Monarch(49) for spatial mixing across 49 positions (sub-layer 1, replaces attention)
- Monarch(784) for feature mixing on the flat vector (sub-layer 2, replaces MLP)

This addresses the key expressivity gap between flat Monarch(784) and attention-based baselines.

## Implementation note

M2's actual sequence mixer uses Hyena-style gated convolutions computed via Monarch FFTs, not plain Monarch multiplication. PhotonFlow uses plain Monarch multiplication for MZI-compatibility, but the two-axis principle still applies.
