# Zhu 2026 - A Fully Real-Valued End-to-End Optical Neural Network for Generative Model

**Citation:** Jiang, Wu, Cheng, Dong. "A fully real-valued end-to-end optical neural network for generative model." Frontiers of Optoelectronics, vol. 19, issue 1, article 4, 2026. DOI: 10.2738/foe.2026.0004

**Note on citation:** Our paper draft cites this as "H. Zhu et al." following an earlier preprint version. The published journal version lists Shan Jiang as first author, with Bo Wu, Qixiang Cheng (University of Cambridge), and Jianji Dong (Huazhong University of Science and Technology) as co-authors.

**Why it matters for us:** This is our primary baseline competitor. It is the closest existing work to PhotonFlow: an optical neural network designed for image generation. Understanding what it does right and where it falls short is essential for positioning our paper.

## The one big idea

Previous optical neural networks operated in the complex domain (because light is a wave with amplitude and phase) and needed electrical post-processing to map back to real-valued outputs between layers. The problem: electrical post-processing breaks optical cascadability, destroying the speed advantage of doing computation in light.

Jiang et al. solve this by using **dual micro-ring modulators (MRMs)** biased at different resonance wavelengths (lambda_n+ and lambda_n-) for real-valued encoding. The positive and negative components of a real-valued signal are carried on separate wavelengths and recombined via a differential photocurrent. This allows the next layer to receive a true real-valued input without any electronic conversion.

They experimentally demonstrate:

1. A tanh-like nonlinear activation function using differential-photocurrent-driven MRMs (10 dB extinction ratio).
2. An iris classification task achieving 98% accuracy.
3. A GAN generator using natural optical noise as input for on-chip image generation.

## The actual hardware

The photonic chip integrates:

- **4 input ports** with dual MRM encoders for real-valued input
- **4x4 MZI mesh** for the first linear layer (matrix computation)
- **Nonlinear activation layer** using differential photocurrent driving MRMs of the next layer
- **2x4 linear layer** (second MZI mesh)
- **2 output ports**
- **42 phase shifters** (each requiring 20 mW for a pi phase shift)
- **Unbalanced MZIs (UMZIs)** with ~4.5 nm free spectral range for wavelength separation

This is a real fabricated chip, not a simulation. The architecture is small (4 neurons input, 4 hidden, 2 output for iris; 4 input, 4 hidden, 64 output for GAN) but it is the first demonstration of end-to-end real-valued optical generation.

## Experimental results

### Iris classification

- 150 samples (3 classes, 50 each), 105 training, 45 testing
- **98% accuracy** on the optical chip
- Loss converges within 20 iterations

### GAN image generation

- Dataset: MNIST digit "7", **downsampled to 8x8 resolution**
- Architecture: 4 input neurons, 4 hidden neurons, 64 output neurons (8x8 image)
- Input: natural optical noise from a partially coherent ASE light source (0.1 nm bandwidth)
- Training: adversarial setup with cross-entropy loss and Adam optimizer
- Key finding: real-valued inputs with real-valued activation outperform nonnegative input + ReLU configurations

### Hardware performance

- **Latency**: ~1.76 ns per inference (dominated by opto-electronic conversion time)
- **Energy**: 37 pJ per operation
- **Power**: 1.53 W total (0.62 W laser + 0.84 W phase shifters + 2.4 mW nonlinear activation + 70.57 mW external circuits)
- **Throughput**: 72 operations per inference pass

## How PhotonFlow differs

| Jiang/Zhu 2026 (Optical GAN) | PhotonFlow |
|---|---|
| GAN objective (minimax, adversarial) | Conditional flow matching (simple regression) |
| Single forward pass at sample time | ODE integration (multiple steps, but each is sub-ns) |
| Real-valued dual-MRM encoding | Monarch layers on MZI meshes |
| tanh-like activation via differential photocurrent | Saturable absorber `tanh(alpha*x)/alpha` |
| Small scale: 4 input, 64 output (8x8 images) | Targets CIFAR-10 (32x32) and CelebA-64 |
| No noise-aware training | Shot noise + thermal crosstalk regularization during training |
| No quantization-aware training | 4-bit QAT fine-tuning stage |
| 37 pJ/OP energy | Target: < 1 pJ per generated sample |

### The stability argument

The fundamental difference is training stability. GAN training is a minimax game: the generator and discriminator fight each other. This is already fragile on a GPU, and gets worse when hardware noise is injected. The paper does not report FID scores or compare against electronic GAN baselines, which makes it hard to assess generation quality quantitatively.

Flow matching ([[Lipman 2023 - Flow Matching]]) is a regression loss. There is no adversarial game. Training loss decreases monotonically. This stability is critical when training with photonic noise injection, because the noise makes the optimization landscape harder, and the last thing you want is an already-unstable adversarial objective on top of that.

### The scale argument

The GAN demonstration is at 8x8 resolution with 64 output neurons. PhotonFlow targets 32x32 (CIFAR-10) and 64x64 (CelebA-64) with 6 to 8 blocks of Monarch layers. The scale difference is roughly 16x to 64x in output dimension. This is where structured matrices ([[Dao 2022 - Monarch]]) become essential: they scale as O(n * sqrt(n)) instead of O(n^2), making larger networks feasible on photonic hardware.

## What we can learn from their design

1. **Real-valued encoding works.** The dual-MRM approach proves that optical NNs can operate in the real domain. We use a different mechanism (Monarch layers handle real-valued computation naturally through block-diagonal structure), but the principle is validated.
2. **Optical noise as generative input.** They use natural ASE noise from a partially coherent light source as the latent input to the GAN. This is clever, using the hardware's own noise as the randomness source. We could potentially do the same for the source distribution `x_0 ~ N(0, I)` in flow matching.
3. **The 1.76 ns latency benchmark.** This gives us a hardware reference point. Our target of < 1 ns per ODE step is in the same ballpark and physically reasonable.

## How we use it

- As the **primary photonic baseline** in our experiments. Our success criterion is to beat this on generation quality (FID) while maintaining comparable or better latency and energy.
- As **motivation** for why flow matching is better than GANs for photonic hardware. Their lack of FID reporting and the known instability of GANs on noisy hardware is the cautionary tale.
- As **related work** that establishes photonic generative models as a real research direction with fabricated hardware, not a thought experiment.
- As a **hardware reference** for latency and energy numbers on a real photonic chip.
- Their observation that real-valued optical networks outperform nonnegative-only architectures validates our use of the saturable absorber (which outputs both positive and negative values) over ReLU.

## See also

- [[Index]]
- [[Lipman 2023 - Flow Matching]] for why we chose flow matching over the GAN objective used here
- [[Shen 2017 - Coherent Nanophotonic Circuits]] for the earlier MZI neural network demonstration
- [[Ning 2024 - Photonic-Electronic Integration]] for the broader hardware landscape
- [[Jacob 2018 - Quantization and Training]] for the QAT stage they lack and we include
