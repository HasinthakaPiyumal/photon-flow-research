# PhotonFlow Lit Review

This vault holds the focused literature notes for the PhotonFlow project. The aim is not to summarize each paper completely. The aim is to capture, in plain English, the one or two ideas from each paper that we actually use, and why.

## Recommended study order

Read in this order. Each paper builds on the ones before it. By the end you will understand every design decision in PhotonFlow.

### Step 1: Understand the hardware constraint

> **1. [[Shen 2017 - Coherent Nanophotonic Circuits]]**
> Start here. This paper shows that MZI meshes can do matrix-vector multiplication at the speed of light using SVD decomposition (`M = UΣV†`). It defines the three photonic primitives we are allowed to use: MZI beamsplitters for linear transforms, saturable absorbers for nonlinearity, and photodetectors for readout. It also exposes the noise sources (phase encoding noise, thermal crosstalk, shot noise) that constrain everything downstream.
>
> After reading: you should understand why "optical matrix multiply is fast and cheap" and what breaks when you try to run a full neural network on a chip.

> **2. [[Ning 2024 - Photonic-Electronic Integration]]**
> Read this second. It is the comprehensive survey that quantifies everything Shen demonstrated. You will learn the actual numbers: 4-6 bit effective MZI precision, 0.1 dB loss per beamsplitter, sub-fJ per MAC energy, and the critical bottleneck (opto-electronic-opto conversion kills latency whenever you leave the optical domain). These numbers set our simulation parameters (sigma_s = 0.02, sigma_t = 0.01) and success criteria (< 1 ns/step, < 1 pJ/sample).
>
> After reading: you should understand why every operation in PhotonFlow must be photonic-native, and where our noise/precision numbers come from.

### Step 2: Understand the generative modeling objective

> **3. [[Lipman 2023 - Flow Matching]]**
> Now shift to the ML side. Flow matching gives us the training loss: learn a vector field `v_theta(x_t, t)` that transports noise to data along straight OT paths. The CFM objective is a simple regression loss (`||v_theta - (x_1 - x_0)||^2`). The key property for us: the loss does not constrain the network architecture at all. You can use any architecture for `v_theta`, which is the freedom PhotonFlow exploits.
>
> After reading: you should understand the CFM loss, the OT interpolation `x_t = (1-t)x_0 + tx_1`, and why flow matching is simpler and more stable than GANs or diffusion.

> **4. [[Peebles 2023 - DiT]]**
> Read this right after Lipman. DiT is the standard backbone for flow/diffusion models: a transformer with softmax attention and adaLN-Zero conditioning. It achieves FID 2.27 on ImageNet. It is also the architecture we cannot use, because softmax attention and LayerNorm are not photonic. DiT defines the problem: "the best backbone is a transformer, and a transformer cannot run on a photonic chip."
>
> After reading: you should understand what we are giving up (attention expressiveness) and what we steal anyway (identity-initialized residuals, timestep conditioning via norm modulation).

### Step 3: Understand the bridge -- structured matrices

> **5. [[Dao 2022 - Monarch]]**
> This is the bridge paper. Monarch matrices (`M = PLP^TR`) are structured linear layers with `O(n√n)` parameters that are both GPU-efficient and, crucially, have the same computational graph as a cascade of MZI beamsplitters. Two block-diagonal multiplies separated by a fixed permutation -- that is literally what a photonic chip does. Monarch replaces attention in our architecture.
>
> After reading: you should see why Monarch is not just a compression trick but a natural fit for MZI hardware, and why a pair of Monarch layers (the MM* class) is expressive enough to replace attention.

> **6. [[Meng 2022 - ButterflyFlow]]**
> Read this as validation of the Monarch choice. ButterflyFlow proves that butterfly-structured matrices (the parent family of Monarch) work inside a generative model, achieving Glow-level density estimation on CIFAR-10 and dominating on structured data (MIMIC-III). This answers the reviewer question: "have structured matrices ever been used for generation?" Yes, and they work.
>
> After reading: you should understand the relationship between butterfly and Monarch matrices, and why we chose Monarch (2 factors, MZI-friendly) over butterfly (log n factors, GPU-unfriendly).

### Step 4: Understand hardware-aware training

> **7. [[Jacob 2018 - Quantization and Training]]**
> MZI phase shifters have only 4-6 bits of effective precision. Naive rounding of float32 weights to 4 bits destroys accuracy. QAT solves this by simulating quantization during training (`r = S(q - Z)` with straight-through estimator for gradients). We apply 4-bit QAT as a 10K-step fine-tuning stage after CFM training converges.
>
> After reading: you should understand the quantization scheme, the straight-through estimator trick, and why QAT is applied as fine-tuning (not from the start) in PhotonFlow.

> **8. [[Ning 2025 - StrC-ONN]]**
> Read this as independent validation of the whole approach. StrC-ONN shows that structured compression (block-circulant, a cousin of Monarch) plus hardware-aware training works on photonic hardware, achieving 74.91% parameter reduction with minimal accuracy loss. It confirms the design pattern: train with hardware imperfections in the loop, not as an afterthought.
>
> After reading: you should see that PhotonFlow and StrC-ONN independently converge on the same principle (structured matrices + noise-aware training for photonic chips), reinforcing both approaches.

### Step 5: Understand the competition

> **9. [[Zhu 2026 - Optical NN for Generative Models]]**
> Read this last. It is our primary photonic baseline: an optical GAN on a real fabricated chip (4x4 MZI mesh, 42 phase shifters). They generate 8x8 MNIST digits at 1.76 ns latency. The limitations are clear: GAN instability on noisy hardware, tiny scale (64 output neurons), and no noise-aware training or QAT. PhotonFlow addresses all of these with flow matching, Monarch layers, and the full noise+QAT training pipeline.
>
> After reading: you should understand exactly where PhotonFlow improves over the state of the art and what claims we need to defend in the paper.

## How they fit together

```
Lipman 2023 (flow matching loss)
        |
        v
   PhotonFlow vector field network
   /        |          \
  /         |           \
Dao 2022    saturable    divisive
Monarch     absorber     power norm
  |         (Shen 2017)  (microring)
  |
  +-- Meng 2022 validates structured matrices for generation
  |
  v
MZI mesh array (Shen 2017)
  |
  +-- Ning 2024 provides hardware specs and noise models
  +-- Ning 2025 confirms structured compression works on MZI
  |
  v
4-bit QAT fine-tune (Jacob 2018)
  |
  v
Compare against:
  - DiT baseline on GPU (Peebles 2023)
  - Optical GAN baseline (Zhu 2026)
```

## The five-stage pipeline and which papers feed each stage

| Stage | What happens | Key papers |
|---|---|---|
| 1. MZI hardware enumeration | List photonic primitives available on chip | [[Shen 2017 - Coherent Nanophotonic Circuits]], [[Ning 2024 - Photonic-Electronic Integration]] |
| 2. Architecture co-design | Build PhotonFlowBlock with Monarch layers | [[Dao 2022 - Monarch]], [[Meng 2022 - ButterflyFlow]], [[Peebles 2023 - DiT]] |
| 3. Training | CFM loss + noise regularization + QAT | [[Lipman 2023 - Flow Matching]], [[Jacob 2018 - Quantization and Training]] |
| 4. Photonic simulation | Profile in torchonn with hardware noise | [[Ning 2024 - Photonic-Electronic Integration]], [[Ning 2025 - StrC-ONN]] |
| 5. Evaluation | FID, latency, energy vs baselines | [[Peebles 2023 - DiT]], [[Zhu 2026 - Optical NN for Generative Models]] |

## What is intentionally NOT here

- Detailed math derivations. Read the original papers for those.
- Anything about diffusion sampling tricks (DDIM, classifier-free guidance, etc.) that we are not using.
- Long related-work sections from each paper. Only the parts that touch our design.

## Open questions to come back to

- Does the 8 to 10 percent FID gap close if we train longer? The paper draft does not commit on this yet.
- Is divisive power normalization stable enough at 4-bit precision? The Dao paper does not test this case.
- What is the actual fJ-per-MAC number on a real chip vs the estimate we get from `torchonn`? Need exp6 to answer.
- How does StrC-ONN's compression compare to native Monarch training in terms of final FID? Could be worth an ablation.
- Can the Zhu 2026 optical GAN be improved with noise-aware training, or is the GAN instability fundamental on noisy hardware?
