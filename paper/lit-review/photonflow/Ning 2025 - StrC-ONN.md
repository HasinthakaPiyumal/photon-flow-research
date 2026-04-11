# Ning 2025 - StrC-ONN: Hardware-Efficient Structured Compression for Optical Neural Networks

**Citation:** Ning, Zhu, Feng, Gu, Pan, Chen. "Hardware-efficient photonic tensor core: Accelerating deep neural networks with structured compression." Optica, vol. 12, issue 7, pp. 1079-1089, 2025. arXiv:2502.01670

**Why it matters for us:** This paper tackles the same problem we face from the compression side: how do you fit a useful neural network onto a photonic chip with limited precision and limited size? Their block-circulant compression approach validates that structured (non-dense) weight matrices are the right tool for optical neural networks. It also confirms that hardware-aware training can compensate for on-chip imperfections, which is exactly what we do with shot noise and thermal crosstalk injection.

## The one big idea

Optical neural networks built on MZI meshes have a hard constraint: the number of MZIs scales quadratically with the matrix dimension. A large dense matrix needs a huge chip. Additionally, the electro-optical interfaces (DACs to set phase shifters, ADCs to read detectors) are expensive and power-hungry.

StrC-ONN proposes **block-circulant structured compression**: instead of implementing a full dense matrix on the photonic tensor core, decompose it into block-circulant factors. A block-circulant matrix is fully defined by its first row of blocks, so a matrix that would need n^2 parameters only needs n parameters. This dramatically reduces both the number of MZIs needed and the number of DAC/ADC channels.

The key insight: not all compression schemes work well on photonic hardware. Random sparsity is hard to route on a chip. Block-circulant structure is regular and maps cleanly to the physical layout of a photonic tensor core.

## What block-circulant compression actually is

A block-circulant matrix of size n x n with block size b has the form:

```
C = [c0  c_{k-1}  c_{k-2}  ...  c1  ]
    [c1  c0       c_{k-1}  ...  c2  ]
    [c2  c1       c0       ...  c3  ]
    [...                         ... ]
    [c_{k-1} c_{k-2} ...        c0  ]
```

where each c_i is a b x b block and k = n/b. The entire matrix is defined by the k blocks in the first row.

This is different from the block-diagonal structure in [[Dao 2022 - Monarch]]:

- **Block-diagonal** (Monarch): independent blocks, no interaction between blocks except through the permutation P. Parameters: O(n * sqrt(n)).
- **Block-circulant** (StrC-ONN): blocks are shared and shifted cyclically. Parameters: O(n). Even more compressed, but less expressive.

Both are structured and hardware-friendly. Both avoid the O(n^2) parameter count of dense layers. StrC-ONN is more aggressively compressed; Monarch is more expressive. For PhotonFlow we choose Monarch because we need the expressiveness to match attention.

## Hardware-aware training framework

StrC-ONN includes a hardware-aware training procedure that models on-chip nonidealities:

- **Phase quantization**: DACs have limited precision, so phase values are rounded.
- **Optical crosstalk**: adjacent waveguides and phase shifters interfere.
- **Insertion loss**: signal attenuates through each MZI stage.
- **Thermal drift**: phase shifters change over time due to heating.

These imperfections are simulated during training so the model learns to be robust to them. This is the same principle as PhotonFlow's shot noise and thermal crosstalk injection, and as [[Jacob 2018 - Quantization and Training]]'s simulated quantization.

The combination validates a design pattern: **train with the hardware imperfections in the loop, not as an afterthought.**

## Results

- **Parameter reduction**: up to **74.91%** compared to uncompressed optical neural networks, while maintaining competitive accuracy on image classification tasks.
- **Power efficiency**: expected **3.56x improvement** compared to conventional uncompressed photonic tensor cores, because fewer MZIs means fewer phase shifters to drive.
- **Accuracy**: comparable to uncompressed models on image processing and classification benchmarks. The structured compression does not significantly degrade accuracy.

The paper is from the same group (Ning, Zhu, Feng, Gu, Pan, Chen at UT Austin) as the [[Ning 2024 - Photonic-Electronic Integration]] survey. They have deep knowledge of the hardware constraints and their compression scheme is designed with real fabrication in mind.

## Connection to PhotonFlow

StrC-ONN and PhotonFlow arrive at a similar conclusion from different directions:

- **StrC-ONN** starts with a dense pretrained network and compresses it into block-circulant factors that fit on a photonic chip. It is a post-hoc compression approach.
- **PhotonFlow** trains with structured (Monarch) layers from the start, so no post-training compression is needed. The structured constraint is baked into the architecture.

Both approaches agree that:

1. Dense layers are impractical on photonic hardware.
2. Structured matrices are the solution.
3. Hardware-aware training (modeling noise, quantization, crosstalk during training) is essential.
4. The accuracy cost of structured compression is acceptable.

The key difference is that block-circulant is more compressed but less expressive than Monarch. For classification (StrC-ONN's target), this is fine. For generation (PhotonFlow's target), we need the extra expressiveness of Monarch layers to match the quality of attention-based models.

## How we use it

- As **supporting evidence** that structured matrices are the right choice for optical neural networks. StrC-ONN and PhotonFlow independently converge on the same general approach (structured + hardware-aware training), even though the specific structures differ.
- As **validation of hardware-aware training**. Their results confirm that modeling on-chip imperfections during training recovers most of the accuracy lost to hardware constraints. This directly supports our noise injection approach.
- As a **related work citation** from the leading group in photonic neural network hardware (same authors as [[Ning 2024 - Photonic-Electronic Integration]]).
- As **context for the compression-vs-expressiveness tradeoff**. StrC-ONN shows how far you can push compression (74.91% parameter reduction) before accuracy degrades. We operate at a less aggressive compression point (Monarch: O(n * sqrt(n)) vs block-circulant: O(n)) to preserve generation quality.
- We do NOT use their block-circulant compression directly. PhotonFlow trains with Monarch layers natively.

## See also

- [[Index]]
- [[Ning 2024 - Photonic-Electronic Integration]] for the hardware survey from the same group
- [[Dao 2022 - Monarch]] for the structured matrix we use (more expressive than block-circulant, still hardware-friendly)
- [[Jacob 2018 - Quantization and Training]] for the QAT framework that complements hardware-aware training
- [[Shen 2017 - Coherent Nanophotonic Circuits]] for the MZI hardware foundation
