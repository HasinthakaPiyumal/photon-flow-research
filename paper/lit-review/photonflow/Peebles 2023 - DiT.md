# Peebles 2023 - Scalable Diffusion Models with Transformers (DiT)

**Citation:** Peebles, Xie. "Scalable diffusion models with transformers." ICCV 2023.

**Why it matters for us:** This is the model we are NOT allowed to use, and it is also the baseline we compare against. DiT is the standard answer for "what backbone should a modern diffusion or flow model have." We need to know what it gets right so we know what we are giving up by using a photonic-friendly architecture.

## The one big idea

Diffusion papers had been stuck on convolutional U-Nets as the backbone. Peebles and Xie showed that the U-Net inductive bias is not actually doing the work. You can replace the U-Net with a plain transformer (a Vision Transformer style block stack) and the model gets better, not worse. The bigger you make the transformer, the lower the FID gets.

This is the paper that made transformers the default backbone for diffusion and flow matching.

## What a DiT block looks like

The forward pass is roughly:

1. Take a noisy latent of shape I x I x C (in their setup, 32 x 32 x 4 from a Stable Diffusion VAE)
2. **Patchify** it into a sequence of T tokens, each of dimension d. Patch size p is a hyperparameter. Smaller p means more tokens and more compute.
3. Add positional embeddings.
4. Run the sequence through N transformer blocks. Each block has:
   - Multi-head **self-attention** with softmax
   - **Layer norm**, but a special one (adaLN-Zero) that scales and shifts based on the timestep `t` and class label `c`
   - Feed-forward MLP
   - Residual connections
5. Final linear layer projects each token back into a noise prediction.

The conditioning trick they introduce is **adaLN-Zero**: instead of adding the timestep and class as extra tokens or via cross-attention, they regress per-channel scale, shift, and gating parameters from the time and class embeddings, and use those to modulate the layer norms inside the block. They initialize the gate to zero so each block starts as the identity, which makes large models train stably.

They tested four conditioning strategies. adaLN-Zero won.

## The scaling result

They sweep model size (S, B, L, XL) and patch size (8, 4, 2). The headline plot shows that **transformer Gflops** correlate strongly with FID. Either grow the model or shrink the patch size. Either way, more compute means lower FID.

The biggest model, DiT-XL/2 at 118.6 Gflops, hits FID 2.27 on class-conditional ImageNet 256x256 with classifier-free guidance. That was state of the art when the paper came out, beating all previous U-Net diffusion models.

## What this means for PhotonFlow

DiT works because attention is incredibly expressive: every token sees every other token in one shot. Layer norm is also load-bearing. Both of these things are exactly what we cannot do on a photonic chip.

- **Softmax attention** requires computing `QK^T`, normalizing it with a softmax, and multiplying by `V`. The softmax is a transcendental nonlinearity over a vector. It cannot be done with a saturable absorber. It would have to be offloaded to electronics, which kills the speed advantage.
- **Layer norm** requires computing a mean and variance over a feature dimension and dividing by them. The variance computation is not natively photonic. It also has to go to electronics.

So DiT is a great architecture if you have a GPU. It is the wrong architecture if you have an MZI mesh.

## The two things we steal anyway

1. **Time conditioning by modulating normalization.** We do not use adaLN, but we do the analogous thing: we add the time embedding into the divisive power norm. The principle that "modulate the norm" is a strong way to inject timestep information comes from this paper.
2. **Initialize the network as the identity.** adaLN-Zero is initialized so each block is the identity at the start of training. We do the same with our Monarch layers. It makes deep stacks train without instability.

## What we replace

| DiT component | PhotonFlow replacement | Why |
|---|---|---|
| Patchify + transformer tokens | Same patch tokenization, but tokens are processed by Monarch layers, not attention | Attention is not photonic |
| Multi-head softmax attention | Pair of [[Dao 2022 - Monarch]] layers (L and R) | Monarch is expressive and runs on an MZI mesh |
| Layer norm / adaLN-Zero | Divisive power normalization with photodetector feedback | Layer norm requires off-chip electronics |
| GELU in the MLP | Saturable absorber `tanh(alpha*x)/alpha` | Only nonlinearity available on chip |

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
