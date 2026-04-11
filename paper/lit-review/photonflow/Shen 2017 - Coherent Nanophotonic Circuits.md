# Shen 2017 - Deep Learning with Coherent Nanophotonic Circuits

**Citation:** Shen, Harris, Skirlo, Prabhu, Baehr-Jones, Hochberg, Sun, Zhao, Larochelle, Englund, Soljacic. "Deep learning with coherent nanophotonic circuits." Nature Photonics 11(7), 441-446, 2017. DOI: 10.1038/NPHOTON.2017.93

**Affiliations:** MIT (Shen, Harris, Skirlo, Prabhu, Englund, Soljacic), Elenion Technologies (Baehr-Jones, Hochberg), Huawei (Sun, Zhao), Twitter/MILA (Larochelle).

**Why it matters for us:** This is the paper that showed a real silicon photonic chip can run a neural network end to end. It is the hardware foundation for PhotonFlow. Every claim we make about MZI meshes doing matrix multiplication at the speed of light traces back to here.

## The one big idea

A neural network layer is just a matrix multiply followed by a nonlinearity. Both of those things can be done in optics:

- Matrix multiply happens for free if you wire light through a mesh of beamsplitters and phase shifters. No power is consumed during the multiplication itself.
- Nonlinearity happens for free if the light passes through a saturable absorber, which is just a material that becomes more transparent when you push more light through it.

Put many such layers in a row and you get a fully optical neural network.

## How they decompose a matrix into optics

Any real-valued matrix M can be decomposed via SVD:

```
M = U Sigma V^dagger
```

where U is an m x m unitary matrix, Sigma is an m x n rectangular diagonal matrix with non-negative real numbers, and V^dagger is the conjugate transpose of the n x n unitary matrix V.

On a photonic chip:

- **U and V^dagger** are implemented with cascaded arrays of MZIs. Each MZI is a 2x2 unitary rotation controlled by an internal phase shifter (theta, sets splitting ratio) and an external phase shifter (phi, sets differential output phase). Any unitary of rank N can be decomposed into sets of SU(2) rotations implemented by cascaded MZIs.
- **Sigma** is implemented with optical attenuators (programmable MZIs that rotate light to an untracked mode). Each diagonal entry is: `Sigma_ii = sin(theta_i / 2)`.

So a single matrix multiply turns into a programmed pattern of phase shifts on a chip. Once you set the phases, the chip just runs.

The chip in the paper has 56 programmable MZIs. They use it to implement a four-neuron, two-layer network for vowel recognition, requiring a total of 48 phase shifter settings (4 layers x 6 MZIs x 2 phases each).

## The saturable absorber equation

The nonlinear activation in the optical domain is modeled by the saturable absorber. The paper gives the governing equation:

```
sigma * tau_s * I_0 = (1/2) * ln(T_m / T_0) / (1 - T_m)
```

where:
- `sigma` is the absorption cross-section
- `tau_s` is the radiative lifetime of the absorber material
- `T_0` is the initial transmittance (a material constant)
- `I_0` is the incident intensity (input)
- `T_m` is the transmittance of the absorber (solve for this given I_0)

The output intensity is then `I_out = I_0 * T_m(I_0)`. Graphene layers integrated on nanophotonic waveguides have been demonstrated as saturable absorbers.

In PhotonFlow we approximate this with `sigma(x) = tanh(alpha * x) / alpha` with alpha = 0.8. The exact shape does not matter much, as long as it is smooth, monotonic, and saturates.

## What they actually measured

- **Vowel recognition:** 360 data points (90 people, 4 vowel phonemes), using 4 log area ratio coefficients as features. Half for training, half for testing.
- **ONN accuracy:** 76.7% (138/180 correct). A 64-bit digital computer running the same trained weights got 91.7% (165/180). The gap is from limited precision and thermal crosstalk, not from anything fundamental.
- **MZI fidelity:** 99.8 +/- 0.003 for the 720 OIU instances used in the experiment, corresponding to approximately 2.24% measurement uncertainty per output port.
- The nonlinearity was simulated on a CPU between layers (the paper is honest about this). The optical activation is a forward-looking design.

## Energy and speed formulas from the paper

The power consumption of the ONN during computation is dominated by the optical power to trigger the saturable absorber and achieve sufficient SNR at the photodetectors.

**Shot-noise-limited SNR:**

```
SNR ~ sqrt(1/n)
```

where n is the number of photons per pulse (assuming shot-noise-limited detection).

**Energy efficiency:**

```
P/R = 5/(m*N) fJ per FLOP
```

for an m-layer network with N neurons. This assumes a saturable absorber threshold of p ~ 1 MW/cm^2 (valid for many dyes, semiconductors, and graphene) and a waveguide cross-section of A = 0.2 x 0.5 um^2.

This is at least **five orders of magnitude** better than conventional GPUs (~100 pJ/FLOP) and at least **three orders of magnitude** better than an ideal electronic computer (~1 pJ/FLOP assuming 16-bit FLOPs and no data movement energy).

The ONN can operate at photodetection rates exceeding 100 GHz, making it at least two orders of magnitude faster than electronic neural networks restricted to GHz clock rates.

## The error model we inherit

This is the part that matters most for PhotonFlow training. The paper isolates the noise sources on a real chip:

- **Phase encoding noise** (`sigma_phi`): you cannot set a phase shifter to exactly the value you want. They measured `sigma_phi ~ 5 x 10^-3` radians on individual MZIs in the lab, but the full chip gets worse because of thermal crosstalk.
- **Thermal crosstalk:** heating one phase shifter slightly heats its neighbors. This is the dominant excess noise in their setup. Can be compensated through additional calibration or reduced by adding thermal isolation trenches. In a static inference-only chip, thermal crosstalk would be eliminated.
- **Photodetection noise** (`sigma_D`): roughly 0.1% in their experiment. Limited by the dynamic range of the photodetectors (30 dB in their setup).
- **Phase precision:** 16 bits in their DAC, but effective precision is lower due to thermal drift and crosstalk.

For PhotonFlow, we model these during training as additive Gaussian noise injected after each Monarch layer. The sigma values in our paper draft (sigma_s = 0.02 for shot noise, sigma_t = 0.01 for thermal crosstalk) are calibrated to land in the same ballpark as Shen's measurements. See [[Ning 2024 - Photonic-Electronic Integration]] for the broader hardware survey that documents typical noise ranges across multiple photonic platforms.

## Limitations to be aware of

- The chip has only 56 MZIs. Real generative models need orders of magnitude more. The paper argues 1000-neuron chips are within current fab capability and 4096-device chips already exist for other applications.
- The nonlinearity in their experiment was actually simulated on a CPU between layers. The optical activation is treated as a forward-looking design.
- The paper also proposes an on-chip training alternative using the finite difference method with forward propagation instead of backpropagation. This is possible because each forward-propagation step on an ONN takes constant time (limited by photodetection rate, > 100 GHz).

## How we use it

- We take the MZI mesh as the unit of compute. Every linear layer in our model has to be expressible as a cascade of two-by-two unitaries. This is what forces us to use [[Dao 2022 - Monarch]] matrices instead of dense layers.
- We take the saturable-absorber nonlinearity as the only allowed activation function. No ReLU. No GELU.
- We borrow the noise model. Shot noise and thermal crosstalk are injected during training, not just at evaluation, so the model learns to be robust to the conditions a real chip will impose.
- The energy and latency numbers from this paper (sub-pJ per sample, sub-ns per ODE step) are the reason PhotonFlow is worth doing at all. They are our success criteria.

## See also

- [[Index]]
- [[Dao 2022 - Monarch]] for why a Monarch layer is the right abstraction for an MZI mesh
- [[Lipman 2023 - Flow Matching]] for the training objective we run on top of this hardware
- [[Ning 2024 - Photonic-Electronic Integration]] for the comprehensive hardware survey that extends these measurements
- [[Zhu 2026 - Optical NN for Generative Models]] for a later demonstration of optical neural networks for generation
