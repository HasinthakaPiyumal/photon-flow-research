# Shen 2017 - Deep Learning with Coherent Nanophotonic Circuits

**Citation:** Shen, Harris, Skirlo, Prabhu, et al. "Deep learning with coherent nanophotonic circuits." Nature Photonics 11(7), 441 to 446, 2017. DOI: 10.1038/NPHOTON.2017.93

**Why it matters for us:** This is the paper that showed a real silicon photonic chip can run a neural network end to end. It is the hardware foundation for PhotonFlow. Every claim we make about MZI meshes doing matrix multiplication at the speed of light traces back to here.

## The one big idea

A neural network layer is just a matrix multiply followed by a nonlinearity. Both of those things can be done in optics:

- Matrix multiply happens for free if you wire light through a mesh of beamsplitters and phase shifters. No power is consumed during the multiplication itself.
- Nonlinearity happens for free if the light passes through a saturable absorber, which is just a material that becomes more transparent when you push more light through it.

Put many such layers in a row and you get a fully optical neural network.

## How they decompose a matrix into optics

Any real matrix M can be written as `M = U Sigma V*` using SVD, where U and V are unitary and Sigma is a diagonal scaling matrix.

- U and V are implemented with a cascaded array of Mach-Zehnder interferometers (MZIs). Each MZI is a small two-by-two unitary that you control with two phase shifters.
- Sigma is implemented with optical attenuators.

So a single matrix multiply turns into a programmed pattern of phase shifts on a chip. Once you set the phases, the chip just runs.

The chip in the paper has 56 programmable MZIs. They use it to implement a four-neuron, two-layer network for vowel recognition.

## What they actually measured

- Chip got 76.7 percent accuracy on the vowel task. A 64-bit digital computer running the same trained weights got 91.7 percent. The gap is mostly from limited precision and thermal crosstalk between phase shifters, not from anything fundamental.
- They estimate the chip would be at least two orders of magnitude faster than electronic neural networks at forward propagation, and roughly 5 orders of magnitude more energy efficient than a GPU at the time (about 5 / mN fJ per FLOP for an m-layer network with N neurons).

## The error model we inherit

This is the part that matters most for PhotonFlow training. The paper isolates two main noise sources on a real chip:

- **Phase encoding noise** (sigma_phi): you cannot set a phase shifter to exactly the value you want. They measured around 5e-3 radians on individual MZIs in the lab, but the full chip gets worse because of...
- **Thermal crosstalk:** heating one phase shifter slightly heats its neighbors. This is the dominant excess noise in their setup.
- **Photodetection noise** (sigma_D): roughly 0.1 percent in their experiment.

For PhotonFlow, we model these during training as additive Gaussian noise injected after each Monarch layer. The sigma values in our paper draft (sigma_s = 0.02 for shot noise, sigma_t = 0.01 for thermal crosstalk) are calibrated to land in the same ballpark as Shen's measurements.

## The activation function we inherit

They model the saturable absorber as a fixed nonlinear transmittance curve. Graphene on a silicon waveguide is given as a working example.

In PhotonFlow we use `sigma(x) = tanh(alpha * x) / alpha` with alpha = 0.8 as a simple smooth approximation. The exact shape does not matter much, as long as it is smooth, monotonic, and saturates. Tanh is fine.

## Limitations to be aware of

- The chip has only 56 MZIs. Real generative models need orders of magnitude more. The paper argues 1000-neuron chips are within current fab capability and 4096-device chips already exist for other applications.
- The nonlinearity in their experiment was actually simulated on a CPU between layers, not run optically. The paper is honest about this. The optical activation is treated as a forward-looking design.
- Thermal crosstalk would be eliminated in a static inference-only chip, but is a real problem during training and calibration.

## How we use it

- We take the MZI mesh as the unit of compute. Every linear layer in our model has to be expressible as a cascade of two-by-two unitaries. This is what forces us to use [[Dao 2022 - Monarch]] matrices instead of dense layers.
- We take the saturable-absorber nonlinearity as the only allowed activation function. No ReLU. No GELU.
- We borrow the noise model. Shot noise and thermal crosstalk are injected during training, not just at evaluation, so the model learns to be robust to the conditions a real chip will impose.
- The energy and latency numbers from this paper (sub-pJ per sample, sub-ns per ODE step) are the reason PhotonFlow is worth doing at all. They are our success criteria.

## See also

- [[Index]]
- [[Dao 2022 - Monarch]] for why a Monarch layer is the right abstraction for an MZI mesh
- [[Lipman 2023 - Flow Matching]] for the training objective we run on top of this hardware
