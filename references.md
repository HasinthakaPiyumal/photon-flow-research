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
