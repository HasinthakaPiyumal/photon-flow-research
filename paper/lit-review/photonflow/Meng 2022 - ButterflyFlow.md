# Meng 2022 - ButterflyFlow: Building Invertible Layers with Butterfly Matrices

**Citation:** Meng, Zhou, Choi, Dao, Ermon. "ButterflyFlow: Building invertible layers with butterfly matrices." ICML 2022, pp. 15360-15375. arXiv:2209.13774

**Why it matters for us:** This paper proves that butterfly-structured matrices (the same family as [[Dao 2022 - Monarch]]) are expressive enough to work inside a generative model. It is the strongest prior evidence that replacing dense layers with structured matrices does not wreck generation quality, which is the bet PhotonFlow is making.

## The one big idea

Normalizing flows need invertible layers. The standard approach is to use masked linear layers or 1x1 convolutions (as in Glow), which limits the class of transforms you can learn. Meng et al. showed that butterfly matrices (products of sparse butterfly factors with O(n log n) parameters) are both:

- **Invertible by construction**, because each butterfly factor is invertible and the product of invertible matrices is invertible. The Jacobian determinant can also be computed in O(n log n) time, which is essential for the normalizing flow likelihood.
- **Expressive enough** to represent permutations, Fourier transforms, convolutions, and many other linear maps that normalizing flows need. In fact, a product of two butterfly layers can represent any convolution matrix.

They build a normalizing flow called ButterflyFlow where every linear layer is a butterfly matrix, and add a block-wise variant for multi-channel image data.

## How butterfly layers work

A butterfly layer is a product of log(n) sparse "butterfly factors." Each level-k butterfly factor B(k, D) is a D x D block-diagonal matrix where each block is a 2-by-2 rotation (at level 1) or a larger block at higher levels. Composing factors at levels 1, 2, ..., log(n) gives a full butterfly matrix.

Key properties:

- **O(n log n) parameters and FLOPs** per layer, versus O(n^2) for dense.
- **Invertible**: each factor is invertible (block-diagonal with invertible blocks). The inverse of a level-k factor can be computed in O(n) time.
- **Efficient log-determinant**: the Jacobian determinant of the full butterfly layer can be computed in O(n log n) time by multiplying the determinants of each factor.
- **Expressiveness**: can represent any permutation matrix, any convolution matrix (via a product of two butterfly layers), and any Fourier-like transform.

## Why butterfly matrices are relevant to us

Butterfly matrices and [[Dao 2022 - Monarch]] matrices are close relatives. A Monarch matrix `M = P L P^T R` is essentially a two-factor butterfly. The key difference:

- **Butterfly matrices** use log(n) factors of sparse two-by-two rotations. Asymptotically efficient (O(n log n)) but not GPU friendly because of the many sequential sparse multiply steps.
- **Monarch matrices** collapse this into just two block-diagonal factors separated by a permutation. Fewer factors, larger blocks, better GPU utilization, and (critically for us) a direct mapping to two columns of MZI beamsplitters.

ButterflyFlow validates the core hypothesis: structured matrices from the butterfly family can drive a generative model without sacrificing quality. Monarch is the GPU-friendly (and MZI-friendly) member of that family, and PhotonFlow uses it instead.

Note that Tri Dao is a co-author on both ButterflyFlow and the Monarch paper. The same research group explored both representations, so the relationship is not a coincidence.

## What they actually showed

### Standard image benchmarks (bits per dimension, lower is better)

| Dataset | Glow (1x1 conv) | Emerging | ButterflyFlow | Residual Flows |
|---|---|---|---|---|
| MNIST | 1.05 | 1.05 | **1.05** | 0.97 |
| CIFAR-10 | 3.35 | 3.34 | **3.33** | 3.28 |
| ImageNet 32x32 | 4.09 | 4.09 | **4.09** | 4.01 |

ButterflyFlow matches or slightly beats Glow and Emerging convolutions on all three image datasets. Residual Flows (i-ResNet based) is better but uses a fundamentally different architecture that is not structured-matrix based.

### Structured data (where ButterflyFlow dominates)

- **Galaxy images** (periodic structure): ButterflyFlow achieves 1.95 bpd vs Glow's 2.02. The butterfly structure naturally captures the rotational periodicity in galaxy morphology.
- **MIMIC-III patient waveforms** (structured time series): ButterflyFlow achieves -27.92 avg NLL, massively outperforming the next best method (Woodbury at -11.47). This is a 2.4x improvement in log-likelihood, using **less than half the parameters** (15,280 vs 36,032 for Glow, 48,576 for Woodbury).

The MIMIC-III result is the standout. ButterflyFlow's ability to learn structured transforms from data gives it an enormous advantage on data with inherent structure that dense layers miss.

## What separates ButterflyFlow from PhotonFlow

| ButterflyFlow | PhotonFlow |
|---|---|
| Normalizing flow (likelihood-based) | Conditional flow matching (regression-based) |
| Butterfly matrices (log n factors) | Monarch matrices (2 factors) |
| Targets GPUs | Targets MZI photonic hardware |
| No hardware noise modeling | Shot noise + thermal crosstalk regularization |
| Full precision | 4-bit QAT for analog photonic precision |
| Generates via invertible transform + change of variables | Generates via ODE integration of learned vector field |

We are not extending ButterflyFlow. We are taking a different generative objective ([[Lipman 2023 - Flow Matching]]) and a different member of the structured matrix family ([[Dao 2022 - Monarch]]). But ButterflyFlow is the paper that tells the reviewer: "yes, structured matrices work in generative models, this has been tried and it works."

## How we use it

- As **evidence** that the butterfly/Monarch family of structured matrices is expressive enough for generation. Without this paper, a reviewer might reasonably ask "have structured matrices ever been used for image generation?" The answer is yes, and they match Glow-level quality.
- As a **point of comparison** for our architecture choice. We chose Monarch over butterfly because Monarch maps to MZI hardware more directly (two columns of beamsplitters, not log(n) columns).
- As **validation from the same research group** (Dao, Ermon at Stanford) that structured matrices are a serious approach to generative modeling, not a toy experiment.
- We do NOT use their normalizing flow framework. PhotonFlow uses conditional flow matching, which is a different training paradigm that does not require invertibility or Jacobian determinant computation.

## See also

- [[Index]]
- [[Dao 2022 - Monarch]] for the structured matrix we actually use (the hardware-friendly cousin of butterfly)
- [[Lipman 2023 - Flow Matching]] for the training objective we pair with Monarch layers
- [[Shen 2017 - Coherent Nanophotonic Circuits]] for the MZI hardware that motivates choosing Monarch over butterfly
