# Peebles 2023 - Scalable Diffusion Models with Transformers (DiT)

**Citation:** Peebles, Xie. "Scalable diffusion models with transformers." ICCV 2023.

**Affiliations:** UC Berkeley (Peebles), New York University (Xie). Work done during an internship at Meta AI, FAIR Team.

**Why it matters for us:** This is the model we are NOT allowed to use, and it is also the baseline we compare against. DiT is the standard answer for "what backbone should a modern diffusion or flow model have." We need to know what it gets right so we know what we are giving up by using a photonic-friendly architecture.

## The one big idea

Diffusion papers had been stuck on convolutional U-Nets as the backbone. Peebles and Xie showed that the U-Net inductive bias is not actually doing the work. You can replace the U-Net with a plain transformer (a Vision Transformer style block stack) and the model gets better, not worse. The bigger you make the transformer, the lower the FID gets.

This is the paper that made transformers the default backbone for diffusion and flow matching.

## The DiT architecture

### Input pipeline

1. Take an image (e.g., 256 x 256 x 3) and encode it through a **pre-trained VAE** (from Stable Diffusion by Rombach et al.) to get a spatial latent of shape 32 x 32 x 4.
2. **Patchify** the latent into a sequence of T tokens, each of dimension d. Patch size p is a hyperparameter (p = 2, 4, or 8). Smaller p means more tokens (T = (I/p)^2) and more compute.
3. Add standard ViT **frequency-based positional embeddings**.

### DiT block

Each of the N transformer blocks contains:

1. Multi-head **self-attention** with softmax
2. **Layer normalization** (with conditioning, see below)
3. **Pointwise feed-forward MLP** (two linear layers with GELU activation)
4. Residual connections

### Output

A final layer norm, linear decoder, and unpatchify operation to map tokens back to the spatial latent shape. The VAE decoder then produces the final image.

## Conditioning mechanisms

They tested four ways to inject the timestep `t` and class label `c` into the network:

1. **In-context conditioning:** append `t` and `c` as two extra tokens to the input sequence.
2. **Cross-attention:** add a cross-attention layer to each block, attending to embeddings of `t` and `c`.
3. **Adaptive LayerNorm (adaLN):** replace standard LayerNorm with one whose scale and shift parameters are regressed from the sum of `t` and `c` embeddings.
4. **adaLN-Zero:** same as adaLN, but add a per-channel gating parameter and initialize all residual blocks as the identity.

**adaLN-Zero won.** For each block, it regresses six vectors from the time+class embedding:

```
(gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2) = MLP(t_emb + c_emb)
```

These are applied as:

```
x = x + alpha_1 * Attention(gamma_1 * LayerNorm(x) + beta_1)
x = x + alpha_2 * FFN(gamma_2 * LayerNorm(x) + beta_2)
```

The alpha parameters (gates) are **initialized to zero**, so each block starts as the identity function. This makes deep models train stably from the start.

## Model configurations

| Model | Layers N | Hidden dim d | Heads | Gflops (p=2) |
|---|---|---|---|---|
| DiT-S | 12 | 384 | 6 | 6.1 |
| DiT-B | 12 | 768 | 12 | 23.0 |
| DiT-L | 24 | 1024 | 16 | 80.7 |
| DiT-XL | 28 | 1152 | 16 | 118.6 |

Gflops are measured per forward pass and vary with patch size. Smaller patches (p=2) give the most tokens and the most compute.

## The scaling result

They sweep model size (S, B, L, XL) and patch size (8, 4, 2). The headline finding: **transformer Gflops correlate strongly with FID.** Either grow the model or shrink the patch size. Either way, more compute means lower FID. This is a clean scaling law.

The biggest model, DiT-XL/2 at 118.6 Gflops, hits **FID 2.27** on class-conditional ImageNet 256x256 with classifier-free guidance. That was state of the art when the paper came out, beating all previous U-Net diffusion models.

## What this means for PhotonFlow

DiT works because attention is incredibly expressive: every token sees every other token in one shot. Layer norm is also load-bearing. Both of these things are exactly what we cannot do on a photonic chip.

- **Softmax attention** requires computing `QK^T`, normalizing with a softmax (a transcendental nonlinearity over a vector), and multiplying by `V`. The softmax cannot be done with a saturable absorber. It would have to be offloaded to electronics, which kills the speed advantage.
- **Layer norm** requires computing a mean and variance over a feature dimension and dividing by them. The variance computation is not natively photonic. It also has to go to electronics.
- **GELU activation** is also non-photonic. It is a smooth approximation of ReLU involving erf(), which has no optical analog.

So DiT is a great architecture if you have a GPU. It is the wrong architecture if you have an MZI mesh.

## The two things we steal anyway

1. **Time conditioning by modulating normalization.** We do not use adaLN, but we do the analogous thing: we add the time embedding into the divisive power norm. The principle that "modulate the norm" is a strong way to inject timestep information comes from this paper.
2. **Initialize the network as the identity.** adaLN-Zero initializes the gating parameters alpha to zero so each block starts as the identity at the start of training. We do the same with our Monarch layers. It makes deep stacks train without instability.

## What we replace

| DiT component | PhotonFlow replacement | Why |
|---|---|---|
| Patchify + transformer tokens | Same patch tokenization, but tokens are processed by Monarch layers, not attention | Attention is not photonic |
| Multi-head softmax attention | Pair of [[Dao 2022 - Monarch]] layers (L and R) | Monarch is expressive and runs on an MZI mesh |
| Layer norm / adaLN-Zero | Divisive power normalization `x / (||x||_2 + eps)` with photodetector feedback | Layer norm requires off-chip electronics |
| GELU in the MLP | Saturable absorber `tanh(alpha*x)/alpha` | Only nonlinearity available on chip |
| Pre-trained VAE latent space | Direct pixel space (for CIFAR-10 and CelebA-64) | Our benchmarks are small enough to skip the VAE |

## The baseline number we care about

In our paper draft, **Experiment 1** is "standard CFM with attention on GPU, 200K steps." That is essentially a flow-matching DiT. It gives us the FID reference. Our success criterion is to land within 10 percent of this number while running on a photonic chip.

The paper draft reports we currently get within 8 to 10 percent on CIFAR-10 and CelebA-64. That is the gap we are paying for the privilege of running on photonic hardware.

## Things from DiT we are NOT using

- Classifier-free guidance. Adds complexity at sample time and the gain is task-specific.
- Latent diffusion (training in VAE space). For our small benchmarks (CIFAR-10, CelebA-64) we work in pixel space.
- The huge ImageNet 512 model. Way too big for our compute budget and not necessary to demonstrate the point.

## How we use it

- As the **target architecture** to beat in FID. Any photonic-friendly model has to be measured against a DiT-style attention baseline trained the same way.
- As the **source of design idioms**: timestep conditioning via norm modulation, and zero-initialized residuals.
- As the **clear statement of the problem**: "the modern diffusion / flow backbone is a transformer, and a transformer is not photonic." Without DiT in the literature, our paper would have a much weaker motivation.

## See also

- [[Index]]
- [[Lipman 2023 - Flow Matching]] for the training objective we plug into this kind of backbone
- [[Dao 2022 - Monarch]] for the layer that replaces attention in our version
- [[Zhu 2026 - Optical NN for Generative Models]] for the photonic baseline we also compare against
