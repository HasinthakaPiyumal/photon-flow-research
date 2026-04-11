# Dao 2022 - Monarch: Expressive Structured Matrices

**Citation:** Dao, Chen, Sohoni, Desai, Poli, Grogan, Liu, Rao, Rudra, Re. "Monarch: Expressive structured matrices for efficient and accurate training." ICML 2022.

**Affiliations:** Stanford University, University at Buffalo. Note: Tri Dao is also a co-author on [[Meng 2022 - ButterflyFlow]], connecting both structured matrix approaches to the same research group.

**Why it matters for us:** This paper is the bridge between flow matching and photonic hardware. Monarch matrices are the structured linear layer that lets us replace attention without losing expressiveness, **and** they have the same computational shape as a cascade of MZI beamsplitters. We did not pick Monarch by accident. We picked it because the math of a Monarch product is essentially the math of an MZI mesh.

## The one big idea

A dense weight matrix has too many parameters and a sparse matrix is hard to run fast on a GPU. People had been looking for a class of structured matrices that is:

1. **Expressive enough** to represent the things real neural networks need (Fourier transforms, convolutions, attention-like mixing).
2. **Hardware friendly** so it actually runs faster than dense, not just on paper.
3. **Easy to project a dense matrix onto** so you can convert pretrained models.

Most prior structured matrices fail at least one of these. Sparse matrices fail #2 (sparse multiply is slow on GPUs). Fourier transforms fail #3. Butterfly matrices fail #2.

Monarch matrices nail all three.

## What a Monarch matrix actually is

A Monarch matrix M of size n x n, where n = m^2, is defined as:

```
M = P L P^T R
```

where:

- **R** is block diagonal with m blocks of size m x m. In formal notation: `R = diag(R_1, R_2, ..., R_m)` where each `R_i` is m x m.
- **L** is also block diagonal with m blocks of size m x m: `L = diag(L_1, L_2, ..., L_m)`.
- **P** is a fixed permutation. It reshapes a length-n vector into an m x m matrix, transposes it, and flattens it back. Formally, P is the stride permutation (also called the perfect shuffle) that maps index `i*m + j` to `j*m + i`.

That is the entire definition. Two block diagonal matrices and a permutation between them.

Operationally, computing `y = Mx` for a vector x of length n = m^2:

```
1. Reshape x into an m x m matrix X
2. Multiply each row of X by R_i:   X' = R * X         (m independent m x m multiplies)
3. Transpose:                        X'' = (X')^T       (the permutation P, free)
4. Multiply each row of X'' by L_i:  Y = L * X''        (m independent m x m multiplies)
5. Flatten Y back to vector y
```

**Parameters:** `2 * m * m^2 = 2 * n * sqrt(n)`, much less than the `n^2` of a dense matrix.

**FLOPs:** `O(n * sqrt(n))` = `O(n^{3/2})`, slower asymptotically than a butterfly matrix's `O(n log n)` but faster in wall-clock time on real GPUs because both passes are batched matrix multiplies (BGEMM), which GPUs love.

## Why this is the same shape as an MZI mesh

This is the part nobody outside the photonics world will see, but it is the entire reason we picked Monarch.

A silicon photonic neural network ([[Shen 2017 - Coherent Nanophotonic Circuits]]) implements an arbitrary unitary by cascading MZI beamsplitters in a triangular layout. Each MZI is a 2-by-2 unitary rotation. Light flows through one column of MZIs, then routes (which is a fixed permutation in waveguides), then flows through the next column.

Now look at a Monarch layer:

- **Block diagonal R** = a column of small matrix multiplies, each acting on a slice of the input. On a chip, this is one column of MZIs.
- **Permutation P** = a fixed shuffle of optical modes. On a chip, this is just how the waveguides are routed. It costs nothing.
- **Block diagonal L** = another column of small matrix multiplies. On a chip, another column of MZIs.

So `M = P L P^T R` is, quite literally, "do a column of MZIs, route the modes, do another column of MZIs, route them back." That is exactly what a photonic chip does anyway.

This is the one-to-one mapping that makes PhotonFlow work. Every Monarch layer in our network corresponds to two columns of MZIs on the proposed chip, with a permutation in between that costs nothing because it is just waveguide routing.

The Monarch paper does not mention photonics at all. The authors built it for GPUs. The fact that it also matches photonic hardware is a happy accident, but it is the accident our whole project depends on.

## Expressiveness

Monarch matrices are at least as expressive as butterfly matrices. The class **MM\*** (products of two Monarch matrices) can represent:

- Convolutions
- Hadamard transform
- Toeplitz matrices

The class **(MM\*)^2** (products of four Monarch matrices) can represent:

- Fourier transform
- Discrete sine and cosine transforms
- Fastfood and ACDC

In other words, two Monarch layers in a row are basically as flexible as the structured-matrix tools any signal-processing person would reach for. We use **a pair of Monarch layers** (L and R) inside each PhotonFlow block, which puts us in the MM\* class.

## Empirical results from the paper

This is not what we use them for, but it is good evidence that Monarch is not a toy:

- **ViT and MLP-Mixer on ImageNet:** swap dense layers with Monarch, train normally. 1.7x to 2x faster than dense, same accuracy.
- **GPT-2 on Wikitext-103:** same story. 1.8x to 2x faster, same perplexity.
- **GPT-2 pretraining on OpenWebText with reverse sparsification:** train with Monarch for the first 90 percent of steps, then unfold to dense for the last 10. 2x total speedup vs all-dense pretraining, no quality drop.
- **PDE solving and MRI reconstruction:** Monarch beats fixed Fourier transforms because it can learn a transform tailored to the data instead of using a hand-picked one.

The takeaway is that Monarch is not a compression trick that costs accuracy. On many tasks it is just a strictly better linear layer than dense.

## Projection: a free benefit

Theorem 1 of the paper says: given any dense matrix A, you can find the closest Monarch matrix to A in closed form, by reshaping A as a 4D tensor and taking the rank-1 SVD of each batch slice. Time complexity is `O(n^(5/2))`.

This means in principle we could take a pretrained DiT and project its attention weights onto Monarch matrices, then fine-tune. We have not tried this yet but it is a fallback if training PhotonFlow from scratch turns out to be hard.

## What we use, what we ignore

| Monarch paper concept | What we do with it |
|---|---|
| Monarch matrix `M = P L P^T R` | Use as the linear layer inside every PhotonFlow block |
| Pair of Monarch layers (MM\* class) | Use as the attention replacement |
| Block size hyperparameter | We follow the paper's default of 2 to 4 blocks |
| Reverse sparsification (sparse-to-dense training) | Not using. We are sparse all the way through because the chip is always sparse. |
| BERT fine-tuning via Monarch projection | Not using yet. Fallback if from-scratch training fails. |

## Things to watch out for

- The block size matters. Too few blocks and you do not get enough hardware speedup. Too many blocks and you lose expressiveness. The paper recommends 2 to 4 for most tasks. We use 4 in the paper draft.
- The permutation P is data-dependent on the matrix size. If we change feature dimensions between layers, we have to recompute it.
- Initialization matters. The paper notes that products of many factors are sensitive to init. Monarch has only two factors, which is why it is more stable than alternatives like Kaleidoscope. We still need to be careful about scaling.

## How we use it

- Every linear layer in `v_theta` that would have been attention or a dense projection is a Monarch layer instead.
- We initialize Monarch layers carefully so the residual stream is well-behaved at the start of training. Same trick as DiT's adaLN-Zero, applied to Monarch.
- The Monarch layers are what we profile in `torchonn` for the photonic simulation. Each Monarch layer maps to a known number of MZI passes, which is what gives us our latency and energy estimates.

## See also

- [[Index]]
- [[Shen 2017 - Coherent Nanophotonic Circuits]] for the photonic hardware that turns out to share Monarch's computation graph
- [[Meng 2022 - ButterflyFlow]] for the same research group's earlier work using butterfly matrices (the parent family of Monarch) in a generative model
- [[Ning 2025 - StrC-ONN]] for an alternative structured compression (block-circulant) for optical neural networks
- [[Peebles 2023 - DiT]] for what we are replacing
- [[Lipman 2023 - Flow Matching]] for the training loss that does not care which kind of linear layer we use
