# PhotonFlow: සම්පූර්ණ පර්යේෂණ විග්‍රහය

> මේ document එක PhotonFlow research paper එකේ සහ එහි references වල ගැඹුරු විග්‍රහයක්.
---

## 1. PhotonFlow කියන්නේ මොකක්ද?

මචං, PhotonFlow කියන්නේ **generative AI model** එකක්. ඒත් මෙතන special එක මොකක්ද කියනවනම්, මේ model එක design කරලා තියෙන්නේ **silicon photonic hardware** (ආලෝක පාදක chip) එකක run වෙන්න පුළුවන් විදිහට.

සාමාන්‍යයෙන් AI models run කරන්නේ GPU (Graphics Processing Unit) එකක. GPU power ගොඩක් කනවා, heat ගොඩක් generate කරනවා, speed එකටත් limits තියෙනවා. ඒත් **photonic chips** වල ගණනය කරන්නේ **ආලෝකය (light)** use කරලා. ආලෝකය speed of light එකෙන් ගමන් කරන නිසා, electronic chips වලට වඩා **100x - 1000x faster** වෙන්න පුළුවන්. Energy consumption එකත් **femtojoule** level එකේ - GPU එකට වඩා **100,000x** අඩුයි!

### ප්‍රශ්නය මොකක්ද?

හරි, photonic chips හොඳයි කියලා අපි දන්නවා. ඒත් ප්‍රශ්නය මොකක්ද කියනවනම්, දැනට තියෙන generative models (flow matching, diffusion models) වල components කිහිපයක් photonic chip එකක run කරන්න බෑ:

| Problem Component | ඇයි බැරි? |
|---|---|
| **Softmax attention** | Softmax කියන්නේ `e^x / sum(e^x)` - exponential function එකක්. මේක optical domain එකේ implement කරන්න බෑ. Electronic circuits වලට යන්නම ඕන. |
| **LayerNorm** | Mean, variance calculate කරලා divide කරන්න ඕන. Optical domain එකේ variance compute කරන්න photonic primitive එකක් නෑ. |
| **ReLU / GELU** | මේ activation functions photonic chip එකක directly implement කරන්න බෑ. |

මේ operations **electronic** domain එකට offload කරන්න ඕන. ඒ කියන්නේ:

```
Optical → Electronic → Optical → Electronic → ...
```

මේ **opto-electronic-opto (O-E-O) conversion** එක හැම step එකකම වෙනවා. ඒකෙන් වෙන්නේ:

1. **Latency** (ප්‍රමාදය) nanoseconds ගණනක් add වෙනවා
2. **Energy** waste වෙනවා
3. **Speed advantage** එක මුළුමනින්ම නැති වෙනවා

### PhotonFlow එකේ solution එක

PhotonFlow කියන්නේ මේ ප්‍රශ්නයට answer එක. අපි කළේ මොකක්ද කියනවනම්, **හැම non-photonic operation එකක්ම photonic-native operation එකකින් replace කිරීම**:

| Standard Component | PhotonFlow Replacement | Photonic Hardware |
|---|---|---|
| Softmax attention | **Monarch (butterfly) linear layers** | MZI mesh array |
| LayerNorm | **Divisive power normalization** `x / (‖x‖₂ + ε)` | Microring resonator + photodetector feedback |
| ReLU / GELU | **Saturable absorber** `σ(x) = tanh(αx) / α`, α=0.8 | Graphene waveguide insert |

මේ විදිහට, **කිසිම operation එකක් electronic domain එකට offload කරන්නේ නැතුව**, හුළුමනින්ම optical domain එකේ inference run කරන්න පුළුවන්.

---

## 2. Research Pipeline: පියවර 5ක ක්‍රමවේදය

PhotonFlow paper එකේ methodology එක stages 5 කට organize කරලා තියෙනවා. මේවා තේරුම් ගන්න ඕන research එක හරියට understand කරගන්න.

### Stage 1: MZI Hardware Enumeration

මුලින්ම අපි photonic chip එකේ **මොනවා කරන්න පුළුවන්ද** කියලා list එකක් හදනවා.

**MZI (Mach-Zehnder Interferometer)** කියන්නේ photonic computing වල **මූලික building block** එක. මේක ගැන ගැඹුරින් පස්සේ Shen 2017 paper එක explain කරනකොට කතා කරනවා. ඒත් මෙතනදී basic idea එක මේකයි:

- **Butterfly linear transforms** → MZI beamsplitter cascades වලින් implement කරන්න පුළුවන්
- **Optical power detection** → Photodetectors වලින්
- **Saturable-absorber nonlinearity** → Graphene waveguide inserts වලින්

**Softmax, LayerNorm, ReLU** - මේවාට photonic primitive එකක් නැති නිසා, exclude කරනවා.

### Stage 2: Architecture Co-design

මේ stage එකේදී **PhotonFlowBlock** එක build කරනවා. හැම block එකක්ම මේ components වලින් හැදිලා:

```
Input x_t → [Monarch L] → [Monarch R] → [Optical Activation σ] → [Power Norm N] → [⊕ Time Embed] → Output v_θ
```

1. **Monarch L** - පළමු block-diagonal matrix multiply
2. **Monarch R** - දෙවන block-diagonal matrix multiply (permutation එකක් අතරමැද)
3. **Optical Activation** σ(x) = tanh(αx)/α - saturable absorber
4. **Divisive Power Normalization** x/(‖x‖₂ + ε) - photodetector feedback loop
5. **Time Embedding** - timestep t add කරනවා

Blocks 6 - 8 ක් stack කරලා full vector field network `v_θ(x_t, t)` එක හදනවා.

### Stage 3: Training

Training loss එක **Conditional Flow Matching (CFM)**:

```
L(θ) = E_{t, x₀, x₁} [ ‖v_θ(x_t, t) - (x₁ - x₀)‖² ]
```

මේකට extra regularizers දෙකක් add කරනවා:
- **Shot noise** injection: σ_s = 0.02 (quantum photon noise)
- **Thermal crosstalk** injection: σ_t = 0.01 (adjacent phase shifter heating)

ඊට පස්සේ **4-bit Quantization-Aware Training (QAT)** fine-tuning stage එකක් (steps 10K).

### Stage 4: Photonic Simulation

`torchonn` library එකෙන් trained weights photonic chip එකක simulate කරනවා:
- MZI phase quantization (4-bit)
- Optical loss (0.1 dB per beamsplitter)
- Detector noise

### Stage 5: Evaluation

- **FID** (Frechet Inception Distance) - image quality measure කරනවා
- **Photonic latency** - ns/step
- **Energy** - fJ/MAC, pJ/sample

Success criteria:
- FID: GPU baseline එකට 10% ඇතුළත
- Latency: < 1 ns per ODE step
- Energy: < 1 pJ per generated sample

---

## 3. References ගැඹුරු විග්‍රහය

දැන් අපි හැම reference paper එකක්ම deeply explain කරනවා, PhotonFlow එකට එක එකක් connect වෙන්නේ කොහොමද කියලා.

---

### 📄 Reference 1: Shen 2017 - Deep Learning with Coherent Nanophotonic Circuits

**Paper:** Shen, Harris, et al. "Deep learning with coherent nanophotonic circuits." *Nature Photonics* 11(7), 441-446, 2017.

**මේ paper එක ඇයි important?**

මේක PhotonFlow එකේ **hardware foundation** එක. Photonic chip එකක neural network එකක් run කරන්න පුළුවන් කියලා ලෝකයට පළමුවෙන්ම prove කළ paper එක මේකයි. MIT එකේ researchers ලා ඇත්තටම **silicon photonic chip** එකක් හදලා, එකේ neural network එකක් run කරලා vowel recognition task එකක් කළා.

**Core Idea: MZI Mesh එකෙන් Matrix Multiplication**

Neural network layer එකක basic operation එක මොකක්ද? **Matrix multiplication** එකක් (**linear transform**) + **nonlinearity** එකක්. Shen et al. prove කළේ මේ දෙකම optics වලින් කරන්න පුළුවන් කියලා.

**Matrix decomposition with SVD:**

ඕනම real-valued matrix එකක් M, SVD (Singular Value Decomposition) එකෙන් decompose කරන්න පුළුවන්:

```
M = U Σ V†
```

මෙතන:
- **U** සහ **V†** → **unitary matrices** (rotation matrices වගේ). මේවා **cascaded MZI arrays** වලින් implement කරන්න පුළුවන්. හැම MZI එකක්ම 2x2 unitary rotation එකක්.
- **Σ** → **diagonal matrix** (scaling). Optical attenuators වලින් implement කරනවා.

**MZI කියන්නේ මොකක්ද? (Mach-Zehnder Interferometer)**

MZI එකක් තේරුම් ගන්න, light beam එකක් දෙකට split කරනවා කියලා හිතන්න (beamsplitter එකකින්). ඊට පස්සේ එක path එකක phase shift එකක් apply කරනවා (phase shifter). අන්තිමට ආයේ combine කරනවා (second beamsplitter). Phase shift එක change කරලා, output එකේ light intensity ratio එක control කරන්න පුළුවන්.

```
Input light → [Beamsplitter] → Path A (phase θ) → [Beamsplitter] → Output 1
                              → Path B (phase φ) →                → Output 2
```

මේ MZI එක mathematically 2x2 unitary matrix එකක්:

```
MZI(θ, φ) = [[cos(θ/2), -sin(θ/2)],
             [sin(θ/2),  cos(θ/2)]] × phase(φ)
```

MZI arrays cascade කරලා **ඕනම unitary matrix** එකක් implement කරන්න පුළුවන්. මේක 2017 දී Shen et al. ප්‍රායෝගිකව demonstrate කළා.

**Saturable Absorber - Optical Nonlinearity**

Linear transform එකට පස්සේ nonlinearity එකක් ඕන. Electronics වල ReLU, sigmoid use කරනවා. Photonics වල **saturable absorber** එකක් use කරනවා.

Saturable absorber කියන්නේ material එකක් (graphene වගේ) - light intensity වැඩි වෙනකොට **transparent** වෙනවා. Low intensity එකේදී absorb කරනවා, high intensity එකේදී pass through කරනවා. මේක mathematically tanh-like function එකක්:

```
σ(x) ≈ tanh(αx) / α     (α = 0.8 PhotonFlow වලදී)
```

**Experiment Results:**

- Chip එකේ **programmable MZIs 56** ක් තිබුණා
- **Vowel recognition** task: 76.7% accuracy (digital computer එකෙන් 91.7%)
- Accuracy gap එක එන්නේ **limited precision** (thermal crosstalk) නිසා, fundamental limitation එකක් නෙවෙයි
- MZI fidelity: **99.8%** ± 0.003

**Noise Model - PhotonFlow Training එකට Critical:**

මේ paper එකෙන් තමයි අපේ noise parameters එන්නේ:

- **Phase encoding noise** (σ_φ ≈ 5×10⁻³ radians) - phase shifter exactly set කරන්න බැරි නිසා
- **Thermal crosstalk** - එක phase shifter එකක් heat කරනකොට, ළඟ ඉන්න ones වලටත් affect වෙනවා
- **Photodetection noise** (σ_D ≈ 0.1%) - detector dynamic range limit
- **Shot noise** - quantum level photon counting noise

PhotonFlow training එකේදී මේ noise models simulate කරනවා (σ_s = 0.02, σ_t = 0.01), model එක real chip conditions වලට robust වෙන්න.

**Energy and Speed:**

- Energy per FLOP: **~5/(m×N) fJ** - GPU එකට වඩා **100,000x** efficient!
- Speed: photodetection rate **> 100 GHz** - electronic networks ට වඩා **100x** fast
- Formula: `P/R = 2m × N × 10¹⁴ FLOPs J⁻¹`

**PhotonFlow එකට connection:**

මේ paper එකෙන් අපි ගන්නේ:
1. MZI mesh = compute unit → Monarch layers force කරන්නේ
2. Saturable absorber = only allowed activation
3. Noise model = training regularization
4. Energy/latency targets = success criteria

---

### 📄 Reference 2: Ning 2024 - Photonic-Electronic Integrated Circuits for AI Accelerators

**Paper:** Ning, Zhu, Feng, et al. "Photonic-electronic integrated circuits for high-performance computing and AI accelerators." *J. Lightwave Technol.*, vol. 42, pp. 7834-7859, 2024.

**මේ paper එක ඇයි important?**

මේක **137 pages** තියෙන massive survey paper එකක්. UT Austin group එකෙන් (PhotonFlow එකේ hardware parameters එන්නේ මේ group එකේ research වලින්). Photonic computing field එක ගැන **most comprehensive review** එක මේකයි.

**Key Hardware Specs (අපේ simulation parameters එන්නේ මෙතනින්):**

| Parameter | Value | PhotonFlow Impact |
|---|---|---|
| MZI effective precision | **4-6 bits** | 4-bit QAT target |
| Loss per beamsplitter | **~0.1 dB** | Simulation optical loss |
| Detector shot noise | **σ = 0.01 - 0.03** | σ_s = 0.02 |
| Thermal crosstalk | Nearest-neighbor coupling | σ_t = 0.01 |
| Energy per MAC (optical) | **sub-fJ** | < 1 pJ/sample target |
| Total system energy (with DAC/ADC) | **1-10 fJ/MAC** | Energy budget |

**O-E-O Bottleneck (ප්‍රධාන ප්‍රශ්නය):**

Paper එක clearly identify කරනවා, photonic computing වල **biggest bottleneck** එක **opto-electronic-opto conversion** එක කියලා. ඕනම electronic operation එකකට light → electricity → light convert කරන්න ඕන. හැම conversion එකකම:
- **Latency cost** ≈ nanoseconds
- **Energy cost** ≈ significant fraction of total
- **Speed advantage** නැති වෙනවා

**PhotonFlow මේකට solve කරන්නේ කොහොමද?**

අපි **සියලුම** operations photonic-native කළා. O-E-O conversion **zero**. Monarch layers, saturable absorber, divisive power norm - හැමදේම optically implement වෙනවා.

**Survey එකේ key findings:**

1. Most photonic NN demos are **toy-scale** (neurons කිහිපයක් විතරයි)
2. **Hardware-software co-design** essential - standard NN එකක් chip එකට port කරන්න බෑ
3. **Analog photonic computing** = limited precision accept කරලා, ඒකට train කරනවා → QAT
4. **Application-specific** design ≫ general-purpose → PhotonFlow exactly මේකයි

**Simulation methodology (Experiment 6 එකට):**

1. Weight matrix → MZI phases (SVD/Clements decomposition)
2. Phase quantization (4-6 bits)
3. Add optical loss (0.1 dB/stage)
4. Add detector noise (Gaussian)
5. Add thermal crosstalk (correlated noise)

---

### 📄 Reference 3: Lipman 2023 - Flow Matching for Generative Modeling

**Paper:** Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow matching for generative modeling." *ICLR 2023*.

**මේ paper එක ඇයි important?**

මේකෙන් තමයි PhotonFlow එකේ **training objective** එක එන්නේ. Flow matching loss එක. මේ paper එක Meta AI (FAIR) එකෙන් ආවා.

**Flow Matching - Basic Idea:**

Diffusion models ගැන අහලා ඇති නේද? DALL-E, Stable Diffusion, Midjourney - මේවා හැමදේම diffusion-based. ඒවාට variance schedules, score functions, reverse process කියලා complicated concepts ගොඩක් තියෙනවා.

Flow matching **සරලයි**. Idea එක මේකයි:

1. **Random noise** point එකක් ගන්නවා: `x₀ ~ N(0, I)` (Gaussian noise)
2. **Real data** point එකක් ගන්නවා: `x₁` (dataset එකෙන්, e.g., CIFAR-10 image එකක්)
3. එකේ ඉඳලා එකට **straight line** එකක් draw කරනවා
4. Neural network එකට train කරනවා: **මේ line එකේ ඕනම point එකක direction** (velocity) predict කරන්න

```
Noise x₀ ---------> Data x₁
         straight line (OT path)
```

**Time variable** `t` ∈ [0, 1]:
- t = 0 → pure noise
- t = 1 → real data
- Intermediate t → noise සහ data mix එකක්

**Linear interpolation (OT path):**

```
x_t = (1 - t) × x₀ + t × x₁
```

**Training target:**

Network `v_θ(x_t, t)` predict කරන්න ඕන velocity `x₁ - x₀`. ඒ කියන්නේ noise point එකේ ඉඳලා data point එකට direction එක.

**CFM Loss:**

```
L(θ) = E_{t, x₀, x₁} [ ‖v_θ(x_t, t) - (x₁ - x₀)‖² ]
```

මේක **simple regression loss** එකක්! MSE (Mean Squared Error) එකක් විතරයි. Compare කරන්න:
- **GANs** = minimax game (unstable, mode collapse)
- **Diffusion** = variance schedules, score matching (complex)
- **Flow matching** = just MSE regression (simple, stable)

**Mathematical Framework (ගැඹුරට):**

Flow matching theory එක **Continuous Normalizing Flows (CNFs)** මත පදනම් වෙනවා.

**Vector field** `v_t(x)` එකක් define කරනවා, ඒකෙන් **flow** `φ_t(x)` එකක් generate වෙනවා:

```
d/dt φ_t(x) = v_t(φ_t(x)),    φ₀(x) = x        (ODE)
```

මේ flow එක simple distribution `p₀` (Gaussian noise) එකෙන් complex distribution `p₁` (real data) එකට transform කරනවා.

**Problem:** Target vector field `u_t(x)` directly compute කරන්න **intractable** (impossible practically).

**Solution - Conditional Flow Matching (CFM):**

Per-sample **conditional** paths use කරනවා:

```
L_CFM(θ) = E_{t, q(x₁), p_t(x|x₁)} ‖v_t(x) - u_t(x|x₁)‖²
```

**Theorem 2 (crucial result):** FM සහ CFM objectives වල gradients **identical**! ∇_θ L_FM = ∇_θ L_CFM

මේකෙන් තේරෙන්නේ: intractable FM optimize කරනවා වෙනුවට, tractable CFM optimize කළාම **exactly same result** එක ලැබෙනවා.

**Optimal Transport (OT) paths:**

Mean සහ standard deviation linearly change වෙනවා:

```
μ_t(x) = t × x₁
σ_t(x) = 1 - (1 - σ_min) × t
```

σ_min → 0 වෙනකොට, target simplify වෙනවා: `u_t ≈ x₁ - x₀`

OT paths **straight lines**. Diffusion paths **curved**. Straight paths = **fewer ODE steps** at inference. Photonic chip එකේ step එකක් sub-nanosecond. Steps අඩු = faster generation.

**ඇයි Flow Matching PhotonFlow-ට perfect?**

1. **Architecture-agnostic**: Loss එක `v_θ` ඇතුළේ මොනවා තියෙනවද කියලා care කරන්නේ නෑ. U-Net, Transformer, Monarch - ඕනම architecture එකක් use කරන්න පුළුවන්.
2. **Stable training**: GAN minimax game එකක් නෑ. Noise injection (shot noise, thermal crosstalk) training harder කරන නිසා, stable objective එකක් **critical**.
3. **Few ODE steps**: Straight OT paths = fewer steps = faster photonic inference.
4. **Simple**: MSE regression. No variance schedules. No score functions.

---

### 📄 Reference 4: Peebles 2023 - Scalable Diffusion Models with Transformers (DiT)

**Paper:** Peebles, Xie. "Scalable diffusion models with transformers." *ICCV 2023*.

**මේ paper එක ඇයි important?**

DiT කියන්නේ **අපි use කරන්න බැරි architecture** එක - ඒ වගේම **අපි compare කරන baseline** එකත්. DiT තමයි modern diffusion/flow models වල **standard backbone**. GPU එකක run කරනවනම්, DiT best choice එක. Photonic chip එකක run කරනවනම්, DiT **impossible**.

**DiT Architecture:**

DiT එක Vision Transformer (ViT) based:

1. **Patchify**: Image (e.g., 256×256) → VAE latent (32×32×4) → patches → tokens
2. **DiT Blocks** (N blocks):
   - Multi-head **softmax self-attention**
   - **LayerNorm** (adaptive - adaLN-Zero)
   - **MLP** with **GELU** activation
   - Residual connections
3. **Decoder**: tokens → image

**adaLN-Zero Conditioning (ගැඹුරට):**

Time step `t` සහ class label `c` inject කරන්නේ normalization layer එකට:

```
(γ₁, β₁, α₁, γ₂, β₂, α₂) = MLP(t_emb + c_emb)

x = x + α₁ × Attention(γ₁ × LayerNorm(x) + β₁)
x = x + α₂ × FFN(γ₂ × LayerNorm(x) + β₂)
```

**α parameters zero-initialized** → Training start එකේදී **each block = identity function**. Deep models stable train වෙනවා.

**Scaling result: FID 2.27** on ImageNet 256×256 (state-of-the-art at the time).

**ඇයි DiT photonic chip එකක run කරන්න බෑ?**

| DiT Component | Problem |
|---|---|
| **Softmax attention** | `softmax(QK^T/√d)V` - exponential function (`e^x`), sum, division. Optical domain එකේ implement කරන්න බෑ. |
| **LayerNorm** | Mean, variance compute → division, square root. Not photonic. |
| **GELU** | `x × Φ(x)` where Φ is Gaussian CDF - `erf()` function. No optical analog. |

**PhotonFlow එකට DiT එකෙන් ගන්නේ මොනවාද?**

1. **Time conditioning via norm modulation** - adaLN idea එක. අපි divisive power norm එකට time embedding add කරනවා.
2. **Zero-initialized residuals** - α = 0 trick. Monarch layers වලටත් apply කරනවා.

**DiT vs PhotonFlow:**

| DiT | PhotonFlow |
|---|---|
| Softmax attention | Monarch layer pair (L, R) |
| LayerNorm / adaLN-Zero | Divisive power norm x/(‖x‖₂ + ε) |
| GELU | Saturable absorber tanh(αx)/α |
| VAE latent space | Direct pixel space (small benchmarks) |
| FID 2.27 (ImageNet) | Within 8-10% of GPU baseline |

---

### 📄 Reference 5: Dao 2022 - Monarch: Expressive Structured Matrices

**Paper:** Dao, Chen, et al. "Monarch: Expressive structured matrices for efficient and accurate training." *ICML 2022*.

**මේ paper එක ඇයි important?**

**මේක PhotonFlow එකේ bridge paper එක.** Flow matching (ML side) සහ photonic hardware (physics side) connect කරන key paper එක මේකයි. Monarch matrices තමයි attention replace කරන layer type එක, **ඒ වගේම** MZI mesh එකේ computation graph එකට **exactly match** වෙන layer type එකත් එකමයි!

**Monarch Matrix Definition:**

n×n matrix එකක් (n = m²):

```
M = P L Pᵀ R
```

මෙතන:
- **R** = block diagonal: `diag(R₁, R₂, ..., R_m)` - m blocks, each m×m
- **L** = block diagonal: `diag(L₁, L₂, ..., L_m)` - m blocks, each m×m
- **P** = **fixed permutation** (stride/perfect shuffle) - reshape m×m, transpose, flatten

**Operations:**

```
1. Reshape x → m×m matrix X
2. Multiply each row by R_i:    X' = R × X         (m independent m×m multiplies)
3. Transpose:                    X'' = (X')ᵀ         (permutation P - FREE!)
4. Multiply each row by L_i:    Y = L × X''          (m independent m×m multiplies)
5. Flatten Y → y
```

**Parameters:** 2×n×√n (dense matrix එකක n² ට compare කරන්න - ගොඩක් less!)
**FLOPs:** O(n^{3/2})

**ඇයි Monarch = MZI Mesh ද? (Critical Insight!)**

මේක PhotonFlow **entire project එක depend වෙන** insight එක. Monarch paper එකේ authors මේක mention කරන්නේ නෑ - ඔවුන් GPU efficiency ගැන focus කළා. ඒත් coincidence එකක් විදිහට, **Monarch computation graph = MZI mesh computation graph**.

Compare කරන්න:

| Monarch | MZI Mesh |
|---|---|
| Block diagonal R | Column of MZIs (small matrix multiplies on input slices) |
| Permutation P | Waveguide routing (physical wiring - **zero cost!**) |
| Block diagonal L | Another column of MZIs |

```
Monarch:    x → [Block-diag R] → [Permute P] → [Block-diag L] → y
MZI Mesh:   x → [MZI column 1] → [Waveguide routing] → [MZI column 2] → y
```

**EXACTLY SAME STRUCTURE!** එක Monarch layer එකක් = MZI columns දෙකක් + permutation එකක් (free). මේක "happy accident" එකක් ඒත් **PhotonFlow whole project එක depend වෙන්නේ මේ accident එක මත**.

**MM* Class (Pair of Monarch Layers):**

Monarch layers දෙකක product (MM*) represent කරන්න පුළුවන්:
- Convolutions
- Hadamard transform
- Toeplitz matrices

(MM*)² (Monarch layers 4 product) represent කරන්න පුළුවන්:
- Fourier transform
- Discrete sine/cosine transforms

PhotonFlow block එකක Monarch L සහ R use කරනවා → MM* class → attention replace කරන්න ප්‍රමාණවත් expressiveness.

**GPU Results (Monarch original paper):**

- ViT/MLP-Mixer on ImageNet: **2x faster**, same accuracy
- GPT-2 on Wikitext-103: **2x faster**, same perplexity
- PDE solving, MRI reconstruction: Monarch beats hand-crafted Fourier transforms

**Projection (Theorem 1):**

Dense matrix A → closest Monarch matrix M closed-form solution එකක් තියෙනවා (rank-1 SVD per batch slice). මේකෙන් **pretrained DiT weights Monarch-ට project** කරන්නත් පුළුවන් theoretically.

---

### 📄 Reference 6: Meng 2022 - ButterflyFlow

**Paper:** Meng, Zhou, Choi, Dao, Ermon. "ButterflyFlow: Building invertible layers with butterfly matrices." *ICML 2022*.

**මේ paper එක ඇයි important?**

මේක **validation paper** එක. "Structured matrices generative model එකක use කරන්න පුළුවන්ද?" කියන ප්‍රශ්නයට "ඔව්, පුළුවන්" කියලා answer දෙන paper එක මේකයි. Reviewer කෙනෙක් "structured matrices generation වලට කලින් use කරලා තියෙනවද?" කියලා ඇහුවොත්, මේ paper එක point කරන්න පුළුවන්.

**Butterfly vs Monarch:**

Butterfly matrices සහ Monarch matrices **close relatives**:

| Feature | Butterfly | Monarch |
|---|---|---|
| Factors | log(n) sparse factors | **2** block-diagonal factors |
| Parameters | O(n log n) | O(n√n) |
| GPU efficiency | Bad (many sequential sparse ops) | **Good** (batched matrix multiply) |
| MZI mapping | log(n) columns of MZIs | **2 columns** of MZIs |

Tri Dao **co-author** on both papers! Same Stanford research group.

**ButterflyFlow Results:**

| Dataset | ButterflyFlow (bpd) | Glow (bpd) |
|---|---|---|
| MNIST | **1.05** | 1.05 |
| CIFAR-10 | **3.33** | 3.35 |
| ImageNet 32 | **4.09** | 4.09 |

MIMIC-III (structured medical data): **-27.92 NLL** vs Glow's worst performance → **2.4x improvement, half the parameters!**

**PhotonFlow ඇයි Monarch choose කළේ Butterfly නෙවෙයි?**

1. Monarch = 2 factors → **2 MZI columns** (practical, simple chip layout)
2. Butterfly = log(n) factors → **log(n) MZI columns** (complex, more loss)
3. Monarch GPU-friendly (batched multiply) → training efficient
4. Both are expressive enough for generation

---

### 📄 Reference 7: Jacob 2018 - Quantization and Training of Neural Networks

**Paper:** Jacob, Kligys, et al. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." *CVPR 2018*.

**මේ paper එක ඇයි important?**

MZI phase shifters bit 4-6 ක precision එකක් විතරයි. Float32 weights bit 4 ට round කළොත් accuracy **dramatically drop** වෙනවා. QAT (Quantization-Aware Training) use කරලා training time එකේදීම quantization simulate කළොත්, model එක low precision එකට **adapt** වෙනවා. මේක Google එකේ paper එකක් - TensorFlow Lite quantization standard මේකෙන් ආවා.

**Quantization Scheme:**

Real value `r` සහ quantized value `q` අතර mapping:

```
r = S × (q - Z)
```

- **S** (scale): positive real number
- **Z** (zero-point): integer - real zero exactly representable
- **q**: unsigned 8-bit integer (original paper), **4-bit** (PhotonFlow)

**QAT Training Process:**

1. Weights/activations **float32** store කරනවා, normal backprop
2. Forward pass එකේ **fake quantization nodes** insert: float → quantize → dequantize → float
3. Backprop: **straight-through estimator** - quantize/dequantize operation through identity gradient
4. Batch norm **fold** into conv weights

**4-bit Results (Critical Warning!):**

| Precision | Accuracy Loss |
|---|---|
| 8-bit weights, 8-bit activations | -0.9% to -1.3% |
| **4-bit weights**, 8-bit activations | **-11.4% to -14.0%** |
| 8-bit weights, **4-bit activations** | -3.1% to -3.7% |

**4-bit weight quantization = MAJOR accuracy drop!** නිවැරදිව -14% accuracy gap එකක්.

**PhotonFlow QAT Strategy:**

මේ reason එක නිසා, PhotonFlow **directly 4-bit QAT train කරන්නේ නෑ**. Instead:

```
Stage 1: Float32 + photonic noise training → convergence (100K-200K steps)
Stage 2: 4-bit QAT fine-tuning (10K steps only)
```

Model එක පළමුව noise-robust වෙනවා, **ඊට පස්සේ** quantization apply කරනවා. Both simultaneously add කළොත් training too noisy වෙලා converge වෙන්නේ නෑ.

---

### 📄 Reference 8: Ning 2025 - StrC-ONN (Structured Compression for Optical NNs)

**Paper:** Ning, Zhu, Feng, et al. "Hardware-efficient photonic tensor core: Accelerating deep neural networks with structured compression." *Optica*, vol. 12, 2025.

**මේ paper එක ඇයි important?**

මේක **independent validation** එක PhotonFlow approach එකට. UT Austin group එකම (Ning 2024 survey paper එකත් ලිව්වේ) structured compression + hardware-aware training photonic chips වල works කියලා prove කරනවා.

**Block-Circulant Compression:**

StrC-ONN use කරන්නේ **block-circulant** matrices:

```
C = [c₀   c_{k-1}  c_{k-2}  ...  c₁  ]
    [c₁   c₀       c_{k-1}  ...  c₂  ]
    [c₂   c₁       c₀       ...  c₃  ]
    [...]
```

Matrix එක **first row** එකෙන් fully define වෙනවා. Parameters: **O(n)** - Monarch (O(n√n)) ට වඩාත් compressed!

**Monarch vs Block-Circulant:**

| Feature | Monarch (PhotonFlow) | Block-Circulant (StrC-ONN) |
|---|---|---|
| Parameters | O(n√n) | **O(n)** (more compressed) |
| Expressiveness | **Higher** | Lower |
| Use case | Generation (need expressiveness) | Classification (compression sufficient) |
| Approach | Train from scratch with structure | Compress pretrained model |

**Results:**

- Parameter reduction: **74.91%**
- Power efficiency: **3.56x improvement**
- Accuracy: comparable to uncompressed models

**Key Validation for PhotonFlow:**

StrC-ONN සහ PhotonFlow **independently** same conclusion එකට එනවා:

1. Dense layers photonic hardware වල impractical
2. **Structured matrices** = solution
3. **Hardware-aware training** (noise, quantization simulate) essential
4. Accuracy cost acceptable

Difference එක: StrC-ONN classification වලට (block-circulant sufficient), PhotonFlow generation වලට (Monarch expressiveness ඕන).

---

### 📄 Reference 9: Zhu/Jiang 2026 - Optical Neural Network for Generative Models

**Paper:** Jiang, Wu, Cheng, Dong. "A fully real-valued end-to-end optical neural network for generative model." *Frontiers of Optoelectronics*, 2026.

**මේ paper එක ඇයි important?**

මේක **අපේ primary competitor** - photonic hardware එකක generative model එකක් run කරන **first ever demonstration**. Real fabricated chip එකක actual GAN run කරලා images generate කළා.

**Hardware:**

- **4×4 MZI mesh** (first linear layer)
- **42 phase shifters** (20 mW each for π phase shift)
- **Dual micro-ring modulators (MRMs)** - real-valued encoding
- **2×4 linear layer** (second mesh)
- Real fabricated **silicon photonic chip**

**Key Innovation - Real-Valued Encoding:**

Previous optical NNs complex domain (amplitude + phase) එකේ work කළා. Electronic post-processing ඕන වුණා real values get කරන්න. Jiang et al. **dual MRMs** use කරනවා:
- λ_n+ wavelength = positive component
- λ_n- wavelength = negative component
- Differential photocurrent = real-valued output

**Results:**

- Iris classification: **98% accuracy** on chip
- GAN: MNIST digit "7", **8×8 resolution** (very small!)
- Latency: **~1.76 ns** per inference
- Energy: **37 pJ** per operation

**GAN vs Flow Matching (PhotonFlow advantage):**

| Optical GAN (Jiang 2026) | PhotonFlow |
|---|---|
| GAN = minimax game (unstable) | CFM = regression (stable) |
| Scale: 8×8 images | Scale: 32×32 (CIFAR), 64×64 (CelebA) |
| No noise-aware training | Shot noise + thermal crosstalk regularization |
| No QAT | 4-bit QAT fine-tuning |
| No FID reported (quality?) | FID within 10% of GPU baseline |

**GAN Training on Noisy Hardware = BAD IDEA:**

GAN training GPU එකේ වුණත් unstable. Generator සහ discriminator fight කරනවා - mode collapse risk. **Photonic noise add කරනකොට** situation **worse** වෙනවා.

Flow matching = simple regression. Loss monotonically decreases. **Noise injection tolerance high**.

**PhotonFlow vs Optical GAN - Clear Improvements:**

1. **Stable training** (flow matching vs GAN)
2. **Larger scale** (CIFAR-10, CelebA-64 vs 8×8 MNIST)
3. **Hardware-aware training** (noise regularization)
4. **Quantization-aware** (4-bit QAT)
5. **Quantitative quality** (FID reported vs not)

---

## 4. References එකිනෙකට Connect වෙන්නේ කොහොමද?

මේ section එක super important. References **isolated papers** නෙවෙයි - ඒවා **interconnected story** එකක්.

### Flow Diagram:

```
                    Lipman 2023
                  (Flow Matching Loss)
                        |
                        v
              PhotonFlow Vector Field Network v_θ
              /          |           \
             /           |            \
      Dao 2022      Saturable      Divisive Power
      Monarch       Absorber         Normalization
        |          (Shen 2017)      (Microring)
        |
        +-- Meng 2022 validates structured matrices for generation
        |
        v
   MZI Mesh Array
    (Shen 2017)
        |
        +-- Ning 2024 provides hardware specs & noise models
        +-- Ning 2025 confirms structured compression works on MZI
        |
        v
   4-bit QAT Fine-tune
    (Jacob 2018)
        |
        v
   Compare against:
     - DiT baseline on GPU (Peebles 2023)
     - Optical GAN baseline (Zhu/Jiang 2026)
```

### Stages සහ Papers:

| Stage | What Happens | Key Papers |
|---|---|---|
| 1. MZI Hardware Enumeration | List photonic primitives | Shen 2017, Ning 2024 |
| 2. Architecture Co-design | Build PhotonFlowBlock | Dao 2022, Meng 2022, Peebles 2023 |
| 3. Training | CFM loss + noise + QAT | Lipman 2023, Jacob 2018 |
| 4. Photonic Simulation | Profile in torchonn | Ning 2024, Ning 2025 |
| 5. Evaluation | FID, latency, energy | Peebles 2023, Zhu 2026 |

---

## 5. Key Concepts Deep Dive

### 5.1 MZI (Mach-Zehnder Interferometer) ගැඹුරට

MZI එකක් තේරුම් ගන්න, water pipe analogy එකක් use කරමු:

හිතන්න, water pipe එකක් දෙකට split වෙනවා (Y-junction). එක pipe එකක "valve" එකක් තියෙනවා flow control කරන්න. ආයේ pipes දෙක join වෙනවා. Valve setting එක change කරලා, output එකේ water distribution control කරන්න පුළුවන්.

MZI එකත් exactly එහෙමයි - **light** pipe (waveguide) එකක්:

```
           Phase shifter θ
              ↓
Input → [BS₁] → Path A → [BS₂] → Output 1
              → Path B →        → Output 2
```

BS = beamsplitter (50/50 light splitter)

**θ** (internal phase shift) = splitting ratio control
**φ** (external phase shift) = output phase control

**2×2 Unitary Matrix:**

```
MZI(θ, φ) = e^{iφ} × [sin(θ/2)   cos(θ/2)]
                      [cos(θ/2)  -sin(θ/2)]
```

**Cascading:** MZI multiple cascade කරලා **ඕනම N×N unitary matrix** එකක් implement කරන්න පුළුවන් (Reck decomposition theorem).

**Key properties:**
- Computation happens at **speed of light** (no clock cycles!)
- **Passive** - once phases set, no power consumed during computation
- **Parallel** - all matrix elements computed simultaneously
- Energy = only phase shifter heaters (~10 mW each)

### 5.2 Monarch Matrix = MZI Mesh (Visual Explanation)

```
Monarch Layer:
              Input x (length n = m²)
                    |
                    v
              [Reshape to m×m]
                    |
    ┌───┬───┬───┬───┤
    │R₁ │R₂ │R₃ │R₄ │  ← Block-diagonal R (m independent m×m multiplies)
    └───┴───┴───┴───┘
                    |
              [Transpose = Permutation P]  ← FREE! Just waveguide routing
                    |
    ┌───┬───┬───┬───┤
    │L₁ │L₂ │L₃ │L₄ │  ← Block-diagonal L (m independent m×m multiplies)
    └───┴───┴───┴───┘
                    |
              [Flatten to vector y]


MZI Mesh on Chip:
              Input light modes
                    |
    ┌───┬───┬───┬───┤
    │MZI│MZI│MZI│MZI│  ← Column 1 of MZIs (independent 2×2 unitaries)
    └───┴───┴───┴───┘
                    |
              [Waveguide crossing pattern]  ← Physical routing = FREE!
                    |
    ┌───┬───┬───┬───┤
    │MZI│MZI│MZI│MZI│  ← Column 2 of MZIs (independent 2×2 unitaries)
    └───┴───┴───┴───┘
                    |
              Output light modes
```

**Structure EXACTLY matches!** මේ coincidence එක මත PhotonFlow entire project එක build වෙලා තියෙන්නේ.

### 5.3 Saturable Absorber (Optical Nonlinearity)

Graphene layer එකක් waveguide එකට integrate කරනවා. Low light intensity එකේදී graphene **absorb** කරනවා. High intensity එකේදී absorption **saturate** වෙනවා (transparent වෙනවා).

```
Output
  ↑
  │         _________
  │       /
  │     /
  │   /
  │ /
  │/________________→ Input
  
  Looks like tanh(x)!
```

PhotonFlow approximation:

```
σ(x) = tanh(0.8x) / 0.8
```

α = 0.8 → tanh ට වඩා ටිකක් gentler slope. Exact shape එක critical නෑ - smooth, monotonic, saturating function එකක් enough.

### 5.4 Divisive Power Normalization (Photonic LayerNorm)

LayerNorm: `(x - μ) / √(σ² + ε)` → mean, variance compute ඕන → electronic!

PhotonFlow alternative:

```
N(x) = x / (‖x‖₂ + ε)
```

**‖x‖₂** (L2 norm) = total power of the optical signal. **Photodetector** එකකින් measure කරන්න පුළුවන් (light intensity sum).

Division by power = **microring resonator** feedback loop. Photodetector total power measure කරනවා → feedback signal → microring resonator attenuation control කරනවා.

**Fully photonic!** Mean/variance compute කරන්න ඕන නෑ. Total power measure + divide = photonic primitives.

### 5.5 Conditional Flow Matching (CFM) - Intuition

Simple analogy: හිතන්න airport එකකට drive කරන්න ඕන කියලා.

**Diffusion model approach:**
- Random walk: start point එකේ ඉඳලා random directions වලට walk කරනවා. Eventually airport එකට reach වෙනවා (hopefully). Path curved, uncertain, ගොඩක් steps.

**Flow matching approach:**
- GPS: start point එකේ ඉඳලා airport එකට **straight line** එකක draw කරනවා. ඒ direction එකට drive කරනවා. Path straight, efficient, **fewer steps**.

```
Diffusion:                    Flow Matching (OT):
  
  x₀ ----\                    x₀ --------→ x₁
          |---\                (straight line!)
               |--→ x₁        
  (curved path!)
```

Training: "මේ point එකේ ඉඳලා airport එකට direction එක මොකක්ද?" predict කරන්න network train කරනවා.

### 5.6 QAT (Quantization-Aware Training) - Practical Example

Float32 weight: `w = 0.7823456789...`

4-bit quantization: 2⁴ = 16 possible values only!

```
4-bit range: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
Mapped to:   -1.0, -0.867, -0.733, ..., 0.0, ..., 0.733, 0.867, 1.0
```

`w = 0.7823...` → closest 4-bit value: `0.733` or `0.867`

**Error introduced!** ±0.05 to ±0.1 per weight.

**Without QAT (post-training quantization):**
Model float32 train කරනවා → train වෙලා ඉවර → weights 4-bit ට round → **accuracy drops 11-14%!**

**With QAT:**
Training **forward pass** එකේදී quantization simulate කරනවා:

```
Forward:  w_float → quantize(w_float) → dequantize → use for computation
                    ↑ introduces error that loss sees
Backward: straight-through estimator (gradient flows through as if identity)
```

Model **quantization error see** කරන නිසා, adapt වෙනවා. Low-precision-friendly weight distribution learn කරනවා.

---

## 6. Experiment Plan විග්‍රහය

Paper draft එකේ Table I:

| Exp | Configuration | Baseline | Target Metric |
|---|---|---|---|
| **1** | Standard CFM + attention on GPU (200K steps) | -- | FID (reference) |
| **2** | PhotonFlow on MNIST (sanity check, 100K steps) | Exp. 1 | FID delta |
| **3** | Exp. 2 + shot noise + thermal crosstalk | Exp. 2 | FID, training curve |
| **4** | Exp. 3 + 4-bit QAT (10K fine-tune) | Exp. 3 | FID, precision |
| **5** | Best config on CelebA-64 | Exp. 1 | FID, IS |
| **6** | Photonic hardware sim. via torchonn | Exp. 1 | ns/step, fJ/MAC |

**Progressive complexity:**

Exp 1 → baseline establish
Exp 2 → "does Monarch work for generation?" (MNIST sanity check)
Exp 3 → "does noise-aware training help?" (shot + thermal noise)
Exp 4 → "does 4-bit QAT preserve quality?" (hardware precision)
Exp 5 → "does it scale to harder datasets?" (CelebA-64)
Exp 6 → "what are the actual photonic performance numbers?" (torchonn simulation)

---

## 7. Success Criteria සහ ඒවා එන්නේ කොහෙන්ද?

| Criterion | Target | Source |
|---|---|---|
| FID within 10% of GPU baseline | ≤ 1.1 × DiT FID | DiT paper (Peebles 2023) sets reference |
| Latency < 1 ns per ODE step | Sub-nanosecond | Shen 2017 (100 GHz photodetection) + Ning 2024 (hardware specs) |
| Energy < 1 pJ per sample | Sub-picojoule | Shen 2017 (sub-fJ/MAC) + Ning 2024 (system energy) |

---

## 8. Open Questions සහ Future Work

1. **FID gap (8-10%) close වෙන්න පුළුවන්ද longer training වලින්?** - Paper draft එක commit කරන්නේ නෑ.

2. **Divisive power normalization 4-bit precision එකේ stable ද?** - Dao paper එක test කරන්නේ නෑ.

3. **Real chip fJ/MAC vs torchonn simulation fJ/MAC?** - Exp 6 results ඕන.

4. **StrC-ONN block-circulant vs native Monarch training FID comparison?** - Ablation study possibility.

5. **Zhu 2026 optical GAN noise-aware training වලින් improve වෙනවද? GAN instability fundamental ද noisy hardware එකේ?** - Open research question.

---

## 9. Datasets සහ Targets

| Dataset | Resolution | Purpose |
|---|---|---|
| **MNIST** | 28×28 | Sanity check (Exp 2) |
| **CIFAR-10** | 32×32 | Primary benchmark (Exp 3, 4) |
| **CelebA-64** | 64×64 | Higher resolution test (Exp 5) |

Baselines:
1. **Standard CFM + attention** on GPU (primary baseline)
2. **Optical GAN** of Zhu/Jiang 2026
3. **Ablated PhotonFlow** without noise regularization

---

## 10. Technical Tools සහ Libraries

| Tool | Purpose |
|---|---|
| **PyTorch** | Core deep learning framework |
| **torchcfm** | Conditional Flow Matching implementation |
| **torchonn** | MZI mesh profiling, photonic simulation |
| **photontorch** | Optical circuit modeling |

---

## 11. Summary - ඇයි PhotonFlow Special ද?

මෙතනදී, whole story එක summarize කරමු:

**Problem Statement:**
- AI inference GPU-intensive, power-hungry
- Photonic chips = fast + efficient alternative
- ඒත් modern generative models (transformers) photonic chips වල run කරන්න බෑ (softmax, LayerNorm non-photonic)
- O-E-O conversion speed advantage kill කරනවා

**PhotonFlow Solution:**
1. **Monarch layers** replace attention → MZI mesh native (Dao 2022)
2. **Saturable absorber** replaces ReLU/GELU → graphene waveguide native (Shen 2017)
3. **Divisive power norm** replaces LayerNorm → microring + photodetector native
4. **CFM training** → stable regression loss, architecture-agnostic (Lipman 2023)
5. **Noise regularization** → shot noise + thermal crosstalk during training (Shen 2017, Ning 2024)
6. **4-bit QAT** → matches MZI precision limits (Jacob 2018, Ning 2024)

**Result:**
- FID within 8-10% of GPU attention baseline
- Sub-nanosecond per ODE step (estimated)
- Sub-picojoule per generated sample (estimated)
- **First generative model co-designed for photonic hardware**

**Why Each Reference Matters (One-Liner Summary):**

| Ref | One-line Role |
|---|---|
| Shen 2017 | MZI mesh = compute unit, noise model source, energy/latency targets |
| Ning 2024 | Hardware specs encyclopedia (precision, loss, noise parameters) |
| Lipman 2023 | CFM = training loss (architecture-agnostic, stable) |
| Peebles 2023 | DiT = what we can't use (softmax, LayerNorm) + FID baseline |
| Dao 2022 | Monarch = attention replacement = MZI mesh native |
| Meng 2022 | ButterflyFlow = validates structured matrices for generation |
| Jacob 2018 | QAT = training for 4-bit MZI precision |
| Ning 2025 | StrC-ONN = independent validation of structured + noise-aware approach |
| Zhu 2026 | Optical GAN = photonic generation competitor (GAN instability, small scale) |

---

## 12. Contribution එකේ Novelty

PhotonFlow **novel** ඇයි?

1. **First** generative model designed for photonic hardware from scratch
2. **Monarch-to-MZI mapping** discovery - Monarch matrices MZI mesh computation graph exactly match (nobody saw this before)
3. **All-photonic pipeline** - zero O-E-O conversions during inference
4. **Noise-aware CFM training** - photonic noise as training regularizer (not just evaluation noise)
5. **Integrated approach** - architecture + training + hardware simulation in one pipeline

**ප්‍රාසාදයක් (bonus):** මේ project එක undergraduate research! Hasinthaka Piyumal (University of Kelaniya, Sri Lanka) සහ Senumi Costa (University of Plymouth, UK).

---

## 13. Glossary - Technical Terms සිංහලෙන්

| English Term | සිංහල Explanation |
|---|---|
| **MZI** (Mach-Zehnder Interferometer) | ආලෝක කිරණ දෙකට බෙදලා, phase shift එකක් apply කරලා, ආයේ combine කරන optical device |
| **Beamsplitter** | ආලෝක කිරණයක් දෙකට බෙදන device |
| **Phase shifter** | ආලෝක තරංගයේ phase (position in wave cycle) change කරන element |
| **Saturable absorber** | Light intensity වැඩි වෙනකොට transparent වෙන material (nonlinearity) |
| **Photodetector** | ආලෝකය electricity බවට convert කරන sensor |
| **Microring resonator** | Ring-shaped waveguide - specific wavelength light trap කරන device |
| **Waveguide** | ආලෝකය guide කරන channel (optical fiber වගේ, chip එක ඇතුළේ) |
| **Flow matching** | Noise → data direction predict කරන generative training method |
| **CFM** (Conditional Flow Matching) | Per-sample conditional paths use කරන tractable flow matching variant |
| **ODE** (Ordinary Differential Equation) | dx/dt = f(x,t) - vector field integrate කරන equation |
| **FID** (Frechet Inception Distance) | Generated images real images වලට compare කරන quality metric (lower = better) |
| **QAT** (Quantization-Aware Training) | Training time quantization simulate කරන training method |
| **Monarch matrix** | M = PLPᵀR structured matrix - two block-diagonals + permutation |
| **Butterfly matrix** | log(n) sparse factors product - Monarch parent family |
| **Block-circulant** | First row define කරන shift-structured matrix |
| **SVD** (Singular Value Decomposition) | M = UΣV† matrix decomposition |
| **adaLN-Zero** | Adaptive LayerNorm with zero-initialized gates (DiT conditioning) |
| **O-E-O conversion** | Optical → Electronic → Optical signal conversion (bottleneck!) |
| **Shot noise** | Quantum photon counting noise |
| **Thermal crosstalk** | Adjacent phase shifters heat interference |
| **DAC** (Digital-to-Analog Converter) | Digital signal → analog voltage (for phase shifters) |
| **ADC** (Analog-to-Digital Converter) | Analog signal → digital value (from photodetectors) |
| **WDM** (Wavelength Division Multiplexing) | Different wavelengths use කරලා parallel computation |

---

> **Final Note:** මේ document එක PhotonFlow research paper එකේ සියලුම references deep understanding එකක් ලබා දීමට සකස් කරන ලදී. Technical accuracy maintain කරමින්, concepts understandable way එකකින් explain කරන්න උත්සාහ කරා. Questions තිබ්බොත්, original papers refer කරන්න - paper/lit-review/pdfs/ folder එකේ PDF files හැමදේම available.
