# Jacob 2018 - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

**Citation:** Jacob, Kligys, Chen, Zhu, Tang, Howard, Adam, Kalenichenko. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." CVPR 2018. arXiv:1712.05877

**Why it matters for us:** This paper defines the quantization-aware training (QAT) framework that PhotonFlow adapts for photonic hardware. MZI phase shifters have 4 to 6 bits of effective precision. We cannot just train in float32 and hope it works at 4 bits. Jacob et al. showed how to train with quantization in the loop so the model learns to be robust to low precision. Their work is implemented in TensorFlow Lite and is the industry standard for deploying quantized models.

## The one big idea

Neural networks trained in float32 can be made to run entirely in integer arithmetic (no floats at all during inference) with minimal accuracy loss, provided you simulate the quantization during training. The paper proposes a complete system:

1. A **quantization scheme** that maps floats to integers via an affine transform.
2. A **training procedure** that simulates quantization in the forward pass while keeping float weights for gradient updates.
3. A **fused inference pipeline** where every operation from input to output is integer-only.

## The quantization scheme

The core mapping between real values `r` and quantized values `q` is:

```
r = S * (q - Z)
```

where:

- `S` (scale) is a positive real number, represented as a fixed-point multiplier at inference time.
- `Z` (zero-point) is an integer of the same type as `q`. It is the quantized value that corresponds to `r = 0`.
- `q` is an unsigned 8-bit integer (uint8) for their standard scheme.

The zero-point ensures that real zero is exactly representable in the quantized domain. This is important because zero-padding, ReLU outputs, and skip connections all produce exact zeros that must map cleanly.

For matrix multiplication, the multiplier `M = S1 * S2 / S3` (ratio of input scales to output scale) is decomposed as `M = 2^(-n) * M0` where `M0` is in [0.5, 1). The multiplication by `M0` is done with a fixed-point int32 multiply, and the `2^(-n)` is a bit shift. This avoids all floating-point operations at inference time.

## Quantization-aware training (simulated quantization)

During training:

1. **Weights and activations are stored in float32** and updated normally via backpropagation.
2. The **forward pass inserts fake quantization nodes**: float values are quantized to integers and immediately dequantized back to floats. This introduces the quantization error into the forward pass, so the loss function sees the effect of limited precision.
3. **Backpropagation uses the straight-through estimator**: gradients flow through the quantize/dequantize operation as if it were the identity function. This is biased but works well in practice.
4. **Batch normalization is folded** into the preceding convolution during training: the BN parameters (gamma, beta, running mean, running variance) are absorbed into the weight and bias of the conv layer. This is critical because BN involves division and square root, which are not cheap in integer arithmetic.

For **weight quantization**: ranges are determined by the actual min/max of the weight tensor (per output channel for convolutions).

For **activation quantization**: ranges are learned during training using exponential moving averages of the observed min/max activations. This is more robust than using the batch min/max directly.

## What they actually measured

### ResNet on ImageNet (8-bit weights, 8-bit activations)

| Model | Float accuracy | 8-bit accuracy | Gap |
|---|---|---|---|
| ResNet-50 | 76.4% | 74.9% | -1.5% |
| ResNet-100 | 78.0% | 76.6% | -1.4% |
| ResNet-150 | 78.8% | 76.7% | -2.1% |

### InceptionV3 on ImageNet

| Precision | Top-1 accuracy | Top-5 recall |
|---|---|---|
| Float | 78.4% | 94.1% |
| 8-bit QAT | 75.4% | 92.5% |
| 7-bit QAT | 75.0% | 92.4% |

7-bit is close to 8-bit, suggesting some headroom exists. The paper does NOT test below 7 bits for InceptionV3.

### MobileNet on ImageNet

Integer-quantized MobileNets achieve higher accuracy than floating-point MobileNets **at the same latency budget**. At 8-bit, the latency drops by roughly 1.5 to 2x on Qualcomm Snapdragon 835/821 while accuracy loss is minimal.

### Face attributes ablation (the 4-bit data point)

This is the most relevant result for PhotonFlow. They tested various weight/activation bit combinations:

- **8-bit weights, 8-bit activations**: -0.9% to -1.3% accuracy loss
- **4-bit weights, 8-bit activations**: **-11.4% to -14.0% accuracy loss**
- **8-bit weights, 4-bit activations**: -3.1% to -3.7% accuracy loss

The 4-bit weight result is sobering: naive 4-bit quantization causes major accuracy degradation even with QAT. This is why PhotonFlow applies QAT as a careful fine-tuning stage after the model has already converged with photonic noise, not as part of initial training.

## What we adapt for PhotonFlow

| Jacob 2018 | PhotonFlow QAT |
|---|---|
| uint8 (8-bit) weights and activations | 4-bit quantized MZI phases |
| Affine mapping `r = S(q - Z)` | Same principle, adapted for phase angles |
| Per-channel scale and zero point for weights | Per-layer scale (MZI meshes have uniform precision across the mesh) |
| Integer multiply-accumulate at inference | Analog optical matrix multiply at inference |
| Straight-through estimator for gradients | Same |
| BN folding into conv weights | Not applicable (no batch norm; we use divisive power normalization) |
| Applied during full training | Applied as fine-tuning after CFM training (10K steps, Experiment 4) |

The main difference is that we apply QAT as a **fine-tuning stage**, not during initial training. The reason: flow matching training with Monarch layers is already a departure from the standard architecture. Adding quantization noise on top of shot noise and thermal crosstalk from the start would make optimization too noisy. So we first train to convergence with float precision + photonic noise, then fine-tune with 4-bit QAT for 10K steps.

The face attributes ablation (Table 4.7 in the paper) confirms that 4-bit is the danger zone where naive QAT starts failing. This motivates our two-stage approach: pre-train with noise robustness first, then carefully quantize.

## Why 4 bits specifically

The [[Ning 2024 - Photonic-Electronic Integration]] survey documents that MZI phase shifters on current silicon photonic chips have 4 to 6 bits of effective precision, limited by:

- DAC resolution for setting phase voltages
- Thermal drift after calibration
- Manufacturing variation in beamsplitter ratios

4 bits is the conservative end. If we can make the model work at 4 bits, it works on any current or near-future chip.

## How we use it

- As the **framework for our quantization stage** (Stage 3 in the methodology). We implement fake quantization in the forward pass of Monarch layers, with straight-through estimators for backprop. The implementation lives in `hardware/qat.py`.
- As the **justification** for the QAT approach. Jacob et al. proved that simulated quantization during training is strictly better than post-training quantization for all tested architectures.
- As a **warning** about 4-bit precision. Their face attributes ablation shows that 4 bits is hard. We address this by first making the model noise-robust (via shot noise and thermal crosstalk training), then applying 4-bit QAT as a gentler fine-tuning step.
- We do NOT use their TensorFlow Lite inference pipeline. Our inference target is a photonic chip, not a mobile CPU. But the training-time simulation of quantization is the same principle.

## See also

- [[Index]]
- [[Ning 2024 - Photonic-Electronic Integration]] for the hardware specs that set the 4-bit precision target
- [[Ning 2025 - StrC-ONN]] for how structured compression interacts with quantization on photonic hardware
- [[Dao 2022 - Monarch]] for the layers we are quantizing
- [[Shen 2017 - Coherent Nanophotonic Circuits]] for the MZI hardware whose precision limitations drive the need for QAT
