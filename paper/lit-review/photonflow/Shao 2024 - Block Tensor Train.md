# Shao et al. 2024 -- Block Tensor Train (BTT)

**Full title:** Compute Better Spent: Replacing Dense Layers with Structured Matrices
**Authors:** A. Shao, K. Bhatia, C. Re, T. Dao
**Venue:** arXiv 2024
**Link:** https://arxiv.org/abs/2406.06248

## What this paper does

Introduces Block Tensor-Train (BTT), a generalization of Monarch matrices with tunable rank. BTT contains Monarch as a special case and can interpolate between Monarch (low-rank) and dense (full-rank) by varying the BTT rank parameter.

## Key results

- BTT matches dense ViT-S/32 on ImageNet with 3.8x less compute
- Better scaling laws than dense matrices on multiple tasks
- Systematic comparison of structured matrix families (low-rank, TT, Monarch, BTT)

## What PhotonFlow uses from this paper

Not directly used in current implementation. Noted as a **potential future upgrade**: if Monarch proves too restrictive for expressivity, BTT allows increasing rank while maintaining structured (MZI-compatible) computation. The key question is whether BTT's higher rank maps cleanly to MZI mesh arrays.
