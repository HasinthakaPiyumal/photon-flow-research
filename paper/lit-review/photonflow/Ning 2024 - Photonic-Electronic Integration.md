# Ning 2024 - Photonic-Electronic Integrated Circuits for AI Accelerators

**Citation:** Ning, Zhu, Feng, Gu, Jiang, Ying, Midkiff, Jain, Hlaing, Pan, Chen. "Photonic-electronic integrated circuits for high-performance computing and AI accelerators." J. Lightwave Technol., vol. 42, pp. 7834-7859, 2024. arXiv:2403.14806

**Why it matters for us:** This is the comprehensive hardware survey that gives PhotonFlow its physical grounding. Every time we claim a specific optical loss budget, a specific precision limit, or a specific energy number, we are pulling from the landscape this paper maps out. At 137 pages (arXiv version) with 11 authors from UT Austin (the same group behind [[Ning 2025 - StrC-ONN]]), it is the most thorough review of photonic computing for AI available.

## The one big idea

Photonic computing is real but not magic. As Moore's Law slows and AI compute demand explodes, integrated photonics offers a genuine alternative for specific workloads. But the path forward is not pure optics; it is **photonic-electronic integration**, where optical components handle the heavy linear algebra and electronic components handle control, calibration, nonlinearities, and I/O.

The paper surveys the full stack:

- **Device level**: MZI meshes, microring resonators (MRRs), wavelength-division multiplexing (WDM), coherent detection, phase-change materials, electro-optic modulators.
- **Circuit level**: photonic tensor cores, optical matrix-vector multipliers, photonic crossbar arrays.
- **Architecture level**: systolic arrays with photonic accelerators, digital-analog hybrid systems, dataflow architectures.
- **Software level**: hardware-aware training, compilation, mapping neural networks to photonic circuits.

The key message: photonic AI accelerators can deliver orders-of-magnitude improvements in energy efficiency and latency for matrix-vector multiplications, but only if the architecture is designed around the constraints of optics (limited precision, optical loss, noise, thermal sensitivity).

## The hardware specs we inherit

This is where our simulation parameters come from:

### MZI mesh precision

- **Effective precision**: 4 to 6 bits, depending on the DAC resolution, thermal stability, and calibration quality.
- Phase shifters are analog devices. The voltage-to-phase mapping is nonlinear and temperature-dependent. After calibration, drift limits the effective precision to a few bits.
- This is why PhotonFlow targets 4-bit QAT in [[Jacob 2018 - Quantization and Training]]. 4 bits is the conservative floor.

### Optical loss

- **Per beamsplitter**: approximately 0.1 dB per MZI stage. This is the insertion loss through one 2x2 coupler + phase shifter.
- In a deep network, loss accumulates. A 10-stage MZI mesh loses about 1 dB (21% of signal power). A 20-stage mesh loses 2 dB (37%).
- This sets a practical limit on network depth unless optical amplification is used between stages.

### Detector noise

- Photodetectors have **shot noise** proportional to the square root of photon count. At typical operating power levels for neural network inference, this gives additive noise on the order of sigma = 0.01 to 0.03.
- Our sigma_s = 0.02 for shot noise sits in the middle of this range.

### Thermal crosstalk

- Adjacent thermo-optic phase shifters heat each other. When you set one phase to a specific value, the heat leaks to neighbors and shifts their phases.
- The survey documents crosstalk coefficients that decay with distance but are significant for nearest neighbors.
- Our sigma_t = 0.01 for thermal crosstalk models this as additive Gaussian noise, which is a simplification but captures the right order of magnitude.

### Energy per operation

- The **optical multiply itself** is sub-femtojoule per MAC: light passes through the beamsplitter, and the multiplication happens passively.
- **Total system energy** (including DACs, phase shifter heaters, laser, photodetectors, ADCs) is in the range of 1 to 10 fJ/MAC for current designs.
- Our target of < 1 pJ per generated sample is derived from these per-MAC numbers multiplied by the MAC count of a full PhotonFlow forward pass (multiple ODE steps, each with 6-8 Monarch layers).

## What the survey says about neural network architectures on photonic chips

The paper is blunt about the state of the field:

1. **Most demonstrations are toy-scale.** The architectures deployed so far are simple fully-connected or convolutional networks with a handful of neurons. Nobody has run a generative model on photonic hardware (the [[Zhu 2026 - Optical NN for Generative Models]] paper came after this survey).
2. **The electronic-photonic interface is the bottleneck.** Every time you need a nonlinear operation that cannot be done optically (like softmax, standard ReLU, or LayerNorm), you have to convert from optical to electronic, do the operation, and convert back. Each opto-electronic-opto (O-E-O) conversion costs latency (~ns) and energy, undoing the speed advantage.
3. **Hardware-software co-design is essential.** You cannot take a standard neural network and hope it runs well on photonic hardware. The architecture must be designed around what the optics can do natively.

This is exactly the problem PhotonFlow solves. By using only photonic-native operations (Monarch layers for linear, saturable absorber for activation, divisive power normalization instead of LayerNorm), we avoid the electronic bottleneck entirely. Our architecture is co-designed for the hardware from the ground up.

## The simulation framework

The paper describes how to model a photonic chip in software:

1. **Decompose** each weight matrix into MZI phases using SVD or Clements decomposition.
2. **Apply phase quantization** to match the DAC precision (4 to 6 bits).
3. **Add optical loss** (0.1 dB per stage) as signal attenuation through the mesh.
4. **Add detector noise** as additive Gaussian at the output.
5. **Add thermal crosstalk** as correlated noise between adjacent phases.

This is the recipe for our Experiment 6 (photonic hardware simulation via `torchonn`). We use `torchonn` to run these steps on trained PhotonFlow weights and measure the degradation.

## Perspectives on the field

The paper identifies several drivers for the future of photonic computing:

- **Heterogeneous integration**: combining photonic and electronic components on the same chip or in the same package, reducing the latency and energy cost of O-E-O conversion.
- **Photonic interconnects**: even if computation stays electronic, photonic data movement between chips can save energy at data-center scale.
- **Analog photonic computing**: accepting limited precision (4-6 bits) and designing networks that work within this constraint, which is exactly what QAT and hardware-aware training enable.
- **Application-specific photonic processors**: designing the photonic chip for a specific workload (like inference of a particular model) rather than trying to build a general-purpose photonic computer.

PhotonFlow fits squarely in the "application-specific analog photonic computing" category: we design a specific generative model architecture around the constraints of a specific class of photonic hardware.

## How we use it

- As the **source of truth for hardware parameters**. When a reviewer asks "where does sigma_s = 0.02 come from?" the answer is: it is consistent with the detector noise levels documented in this survey.
- As the **motivation for our design constraints**. The survey makes clear that O-E-O conversion is the bottleneck, justifying our all-photonic architecture.
- As the **reference for our energy and latency claims**. Our sub-pJ, sub-ns targets come from scaling the per-MAC numbers in this survey by the MAC count of a PhotonFlow forward pass.
- As **context for the field**. This paper tells the reviewer that photonic AI accelerators are a real and active area with dozens of groups publishing hardware demonstrations.
- As the **source of the simulation methodology** for Experiment 6.

## See also

- [[Index]]
- [[Shen 2017 - Coherent Nanophotonic Circuits]] for the original MZI neural network demonstration that this survey builds upon
- [[Ning 2025 - StrC-ONN]] by the same group, applying structured compression to the hardware constraints described here
- [[Zhu 2026 - Optical NN for Generative Models]] for a recent effort to run generative models on photonic hardware
- [[Jacob 2018 - Quantization and Training]] for the QAT technique that addresses the limited precision documented here
