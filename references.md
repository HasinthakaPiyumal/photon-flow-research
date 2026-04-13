# References

Core papers for the PhotonFlow project. PDFs are stored in `paper/lit-review/pdfs/`.

---

## [1] Y. Shen et al., "Deep learning with coherent nanophotonic circuits," *Nature Photonics*, vol. 11, no. 7, pp. 441-446, 2017.

Foundational work proving that MZI meshes can perform matrix-vector multiplications for neural network inference at the speed of light. PhotonFlow builds on this by targeting generative models instead of classifiers on the same MZI hardware.

## [2] Y. Lipman et al., "Flow matching for generative modeling," *Proc. ICLR*, 2023.

Introduces conditional flow matching (CFM), the training objective PhotonFlow uses. CFM learns a vector field that transports noise to data along straight ODE paths -- simpler and more stable than diffusion score matching.

## [3] W. Peebles and S. Xie, "Scalable diffusion models with transformers," *Proc. ICCV*, 2023.

Defines the DiT architecture that pairs transformers with diffusion/flow objectives. PhotonFlow's block structure is inspired by DiT but replaces softmax attention and LayerNorm with photonic-compatible alternatives.

## [4] T. Dao et al., "Monarch: Expressive structured matrices for efficient and accurate training," *Proc. ICML*, 2022.

Introduces Monarch matrices (M = P L P^T R), which decompose dense layers into block-diagonal factors with a fixed permutation. PhotonFlow uses Monarch layers as the direct replacement for self-attention because their butterfly structure maps exactly to cascaded MZI mesh arrays.

## [5] C. Meng et al., "ButterflyFlow: Building invertible layers with butterfly matrices," *Proc. ICML*, pp. 15360-15375, 2022.

Demonstrates that butterfly-structured matrices (the same family as Monarch) work well inside normalizing flows. Validates that structured linear maps preserve expressiveness in generative models -- supporting PhotonFlow's choice to replace dense attention with Monarch layers.

## [6] S. Ning, H. Zhu, C. Feng, J. Gu, Z. Jiang et al., "Photonic-electronic integrated circuits for high-performance computing and AI accelerators," *J. Lightwave Technol.*, vol. 42, pp. 7834-7859, 2024. arXiv:2403.14806.

Comprehensive survey of photonic-electronic integration for AI. Provides the hardware context for PhotonFlow: MZI mesh specifications, optical loss budgets (0.1 dB/beamsplitter), detector noise models, and the 4-to-6-bit effective precision that motivates our QAT stage.

## [7] X. Ning et al., "StrC-ONN: Hardware-efficient structured compression for optical neural networks," *arXiv preprint*, 2025.

Proposes structured compression techniques specifically for optical neural networks. Directly relevant to PhotonFlow's weight quantization strategy and shows how to reduce hardware footprint while maintaining accuracy on MZI meshes.

## [8] H. Zhu et al., "A fully real-valued end-to-end optical neural network for generative model," *Frontiers of Optoelectronics*, 2026.

The primary baseline competitor. Demonstrates an optical GAN for image generation but relies on electrical post-processing between layers. PhotonFlow aims to surpass this by eliminating all electronic fallbacks through photonic-native operations.

## [9] B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," *Proc. CVPR*, 2018.

Establishes the quantization-aware training (QAT) framework that PhotonFlow adapts for photonic hardware. We apply their approach with 4-bit precision to match the limited resolution of analog MZI phase shifters.

## [10] P. Esser et al., "Scaling rectified flow transformers for high-resolution image synthesis," *Proc. ICML*, 2024. arXiv:2403.03206.

The Stable Diffusion 3 paper. Introduces logit-normal timestep sampling for rectified flow training: instead of t ~ U[0,1], sample t = sigmoid(N(m, s)) with m=0, s=1. This biases training toward intermediate timesteps where velocity prediction is hardest, consistently outperforming uniform sampling. PhotonFlow adopts this as the default timestep sampling strategy.

## [11] T. Karras et al., "Analyzing and improving the training dynamics of diffusion models," *Proc. CVPR*, 2024. arXiv:2312.02696.

Systematic analysis of training dynamics for diffusion/flow models. Demonstrates that EMA (Exponential Moving Average) of model weights with decay=0.9999 produces substantially better generation quality than raw training weights. PhotonFlow uses EMA for all sampling and evaluation.

## [12] H. Yao et al., "FasterDiT: Towards faster diffusion transformers training without architecture modification," *Proc. NeurIPS*, 2024. arXiv:2410.10356.

Introduces velocity direction supervision: adding a cosine similarity loss term alongside MSE to enforce correct flow direction, not just magnitude. Combined loss: MSE + λ(1 - cos_sim). Achieves 7x training speedup on ImageNet DiT while matching FID. PhotonFlow uses λ=0.5.

## [13] X. Wang et al., "Residual connections harm generative representation learning," *Proc. ICLR*, 2025. arXiv:2404.10947.

Shows that standard residual connections (y = x + f(x)) cause shallow-layer echoes that harm generative models. Proposes depth-decayed residuals: y = α_d·x + f(x) where α_d decreases with depth. PhotonFlow uses linear decay from 1.0 (first block) to 1/N (last block).

## [14] "Curriculum sampling: A two-phase curriculum for efficient training of flow matching," *arXiv preprint*, 2026. arXiv:2603.12517.

Proposes two-phase timestep curriculum: logit-normal sampling for structure learning (phase 1), switching to uniform for boundary refinement (phase 2). Achieves 16% relative FID improvement and 33% faster convergence over static sampling. PhotonFlow transitions at 60% of training.

## [15] "Time dependent loss reweighting for flow matching and diffusion models is theoretically justified," *arXiv preprint*, 2025. arXiv:2511.16599.

Provides theoretical justification for time-dependent loss weighting. Weights per-sample loss by w(t) = max(1, γ/(1-t+ε)) to emphasize hard timesteps near t=1. PhotonFlow uses γ=5.0.

## [16] "Unveiling the secret of AdaLN-Zero in diffusion transformer," *OpenReview*, 2024.

Analyzes adaLN-Zero initialization and finds that small Gaussian init (std=0.02) slightly outperforms pure zero-init (~2% FID improvement). Zero-init remains the most important component but the small perturbation improves early training dynamics. PhotonFlow uses std=0.02.
