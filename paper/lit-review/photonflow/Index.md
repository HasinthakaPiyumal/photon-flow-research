# PhotonFlow Lit Review

This vault holds the focused literature notes for the PhotonFlow project. The aim is not to summarize each paper completely. The aim is to capture, in plain English, the one or two ideas from each paper that we actually use, and why.

## The nine papers we build on

PhotonFlow sits at the intersection of three worlds: **photonic hardware**, **generative modeling**, and **efficient neural network design**. Each note below covers one paper and ends with a short "How we use it" section.

### Photonic hardware foundation

- [[Shen 2017 - Coherent Nanophotonic Circuits]]  
  The first paper to actually run a neural network on a silicon photonic chip. Shows that an MZI mesh can implement a matrix-vector multiply, and uses a saturable absorber as the nonlinearity. This is the hardware foundation for everything we do.

- [[Ning 2024 - Photonic-Electronic Integration]]  
  Comprehensive survey of photonic-electronic integration for AI accelerators. Provides the hardware specs we simulate: MZI precision (4-6 bits), optical loss (0.1 dB/beamsplitter), detector noise, and thermal crosstalk coefficients.

- [[Ning 2025 - StrC-ONN]]  
  Structured compression for optical neural networks. Independent validation that block-diagonal-plus-permutation structures (the Monarch/butterfly family) are the natural fit for MZI hardware. Confirms that structured compression plus quantization works on photonic chips.

### Generative modeling

- [[Lipman 2023 - Flow Matching]]  
  Introduces conditional flow matching. We keep the loss as is and only redesign the network that learns the vector field.

- [[Peebles 2023 - DiT]]  
  Shows that a transformer can replace the U-Net in diffusion. This is the model we are NOT allowed to use, because it relies on softmax attention. It tells us what we are competing against.

- [[Zhu 2026 - Optical NN for Generative Models]]  
  Our primary baseline competitor. An optical GAN for image generation. Uses a different generative objective (GAN) and different hardware primitives (MRM-based). PhotonFlow aims to beat it with more stable training (flow matching) and better hardware mapping (Monarch on MZI).

### Structured linear algebra

- [[Dao 2022 - Monarch]]  
  Defines Monarch matrices, a class of structured linear layers that are both expressive and hardware friendly. We use these to replace attention. The key insight is that a Monarch layer is the same kind of computation graph that an MZI mesh runs.

- [[Meng 2022 - ButterflyFlow]]  
  Proves that butterfly-structured matrices (the same family as Monarch) work inside generative models. Validates that replacing dense layers with structured matrices does not wreck generation quality.

### Hardware-aware training

- [[Jacob 2018 - Quantization and Training]]  
  Defines quantization-aware training (QAT). We adapt this for 4-bit MZI phase precision. Applied as a fine-tuning stage after flow matching training to bridge the simulation-to-hardware gap.

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
