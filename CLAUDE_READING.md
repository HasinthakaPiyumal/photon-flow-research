# PhotonFlow: සම්පූර්ණ පර්යේෂණ විග්‍රහය

> මේ document එක PhotonFlow research paper එකේ සහ එහි references වල ගැඹුරු විග්‍රහයක්.
---

## 1. PhotonFlow කියන්නේ මොකක්ද?

මචං, PhotonFlow කියන්නේ **generative AI model** එකක්. ඒත් මෙතන special එක මොකක්ද කියනවනම්, මේ model එක design කරලා තියෙන්නේ **silicon photonic hardware** (ආලෝක පාදක chip) එකක run වෙන්න පුළුවන් විදිහට.

සාමාන්‍යයෙන් AI models run කරන්නේ **GPU** (Graphics Processing Unit - originally game graphics render කරන chip, ඒත් parallel computation ගොඩක් කරන්න පුළුවන් නිසා AI training/inference වලට use කරනවා) එකක. GPU power ගොඩක් කනවා, heat ගොඩක් generate කරනවා, speed එකටත් limits තියෙනවා. ඒත් **photonic chips** වල ගණනය කරන්නේ **ආලෝකය (light)** use කරලා. ආලෝකය speed of light එකෙන් ගමන් කරන නිසා, electronic chips වලට වඩා **100x - 1000x faster** වෙන්න පුළුවන්. Energy consumption එකත් **femtojoule** level එකේ - GPU එකට වඩා **100,000x** අඩුයි!

### ප්‍රශ්නය මොකක්ද?

හරි, photonic chips හොඳයි කියලා අපි දන්නවා. ඒත් ප්‍රශ්නය මොකක්ද කියනවනම්, දැනට තියෙන generative models (flow matching, diffusion models) වල components කිහිපයක් photonic chip එකක run කරන්න බෑ:

| Problem Component | ඇයි බැරි? |
|---|---|
| **Softmax attention** | Softmax කියන්නේ `e^x / sum(e^x)` - exponential function (e^x කියන්නේ "e" කියන special number එක x වෙනි බලයට ඔසවන ගණිත function එක - ගොඩක් ඉක්මනට grow වෙනවා, x=10 නම් e^10 ≈ 22026!) එකක්. **Attention** කියන්නේ neural network එකට input එකේ **වැදගත්ම parts focus** කරන්න පුළුවන් mechanism එකක් - sentence එකක "cat sat on mat" කියනකොට "sat" process කරනගමන් "cat" කෙරෙහි වැඩි attention යොමු කරනවා වගේ. Softmax attention එකේදී හැම element pair එකක්ම compare කරලා "මොකද වැදගත්?" decide කරනවා. ඒත් මේක optical domain එකේ implement කරන්න බෑ - exponential, sum, division ඕන. Electronic circuits වලට යන්නම ඕන. |
| **LayerNorm** | Layer Normalization - neural network layer එකක output values "normalize" (සාමාන්‍ය range එකකට ගේන්නවා) කරන technique එකක්. Mean (සාමාන්‍ය/average), variance (values distribute වෙන range එක) calculate කරලා divide කරන්න ඕන. Optical domain එකේ variance compute කරන්න photonic primitive එකක් නෑ. |
| **ReLU / GELU** | මේවා **activation functions** (neural network layer එකක output එක non-linear බවට convert කරන functions - මේවා නැතිව network එකට complex patterns learn කරන්න බෑ, simple straight line fit එකක් විතරක් වෙනවා). **ReLU** (Rectified Linear Unit) = `max(0, x)` - negative values 0 කරනවා, positive values එලෙසම තියනවා. සරලම activation function එක. **GELU** (Gaussian Error Linear Unit) = `x × Φ(x)` - ReLU ට වඩා smooth version එකක්, modern transformers වල use කරනවා. මේ activation functions photonic chip එකක directly implement කරන්න බෑ. |

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

Training **loss** (loss = model එකේ error/mistake measure කරන number, training goal = මේ number minimize කරනවා) එක **Conditional Flow Matching (CFM)**:

```
L(θ) = E_{t, x₀, x₁} [ ‖v_θ(x_t, t) - (x₁ - x₀)‖² ]
```

මේකට extra **regularizers** දෙකක් add කරනවා (regularizer = model එක **overfitting** (training data විතරක් memorize කරලා new data වලට generalize කරන්න බැරි වීම) වලින් ආරක්ෂා කරන extra constraint/penalty, මෙතන hardware noise simulate කරනවා model robust කරන්න):
- **Shot noise** injection: σ_s = 0.02 (quantum photon noise - photon counting එකේ inherent randomness)
- **Thermal crosstalk** injection: σ_t = 0.01 (adjacent phase shifter heating - එක component එකේ heat ළඟ components වලට affect වීම)

ඊට පස්සේ **4-bit Quantization-Aware Training (QAT)** fine-tuning stage එකක් (steps 10K). (**Fine-tuning** = දැනටමත් train කරපු model එකක් additional training small amount එකකින් specific task එකකට/condition එකකට adapt කරනවා - from scratch train කරන්නේ නැතුව, learned weights ටිකක් adjust කරනවා.)

### Stage 4: Photonic Simulation

`torchonn` library එකෙන් trained weights photonic chip එකක simulate කරනවා:
- MZI phase quantization (4-bit)
- Optical loss (0.1 dB per beamsplitter)
- Detector noise

### Stage 5: Evaluation

- **FID** (Frechet Inception Distance) - generated images real images වලට compare කරලා quality measure කරනවා (lower = better, 0 = perfect). Inception neural network එකෙන් feature distributions compare කරනවා.
- **Photonic latency** - ns/step (nanosecond = 10⁻⁹ seconds per ODE step)
- **Energy** - fJ/MAC (femtojoule per **MAC** = Multiply-Accumulate, neural network එකේ basic computation unit: multiply + add), pJ/sample (picojoule per generated sample)

Success criteria:
- FID: GPU baseline එකට 10% ඇතුළත
- Latency: < 1 ns per ODE step
- Energy: < 1 pJ per generated sample

---

## 3. References ගැඹුරු විග්‍රහය

PhotonFlow කියන්නේ අලුතින් හදපු invention එකක් නෙවෙයි — existing research papers 9 ක ලැබුණු discoveries, techniques, සහ insights **combine** කරලා build කරපු system එකක්. හැම paper එකක්ම PhotonFlow puzzle එකට specific piece එකක් contribute කරනවා. මේ section එකේදී, එක එක paper එක **මොකක්ද**, **මොනවද discover/propose කළේ**, **ඒක PhotonFlow ට connect වෙන්නේ කොහොමද** කියලා deeply explain කරනවා.

Papers 9 broadly categories 3 කට බෙදන්න පුළුවන්:

- **Hardware foundations** (References 1, 2, 8): photonic chip එකක මොනවා possible ද, specs මොනවාද, limitations මොනවාද
- **ML/Architecture** (References 3, 4, 5, 6): training method, baseline architecture, Monarch matrices, structured matrix validation
- **Hardware adaptation** (References 7, 9): quantization training, photonic generation competitor

හරි, එකින් එක ගමු.

---

### 📄 Reference 1: Shen 2017 — Deep Learning with Coherent Nanophotonic Circuits

**Paper:** Shen, Harris, et al. "Deep learning with coherent nanophotonic circuits." *Nature Photonics* 11(7), 441-446, 2017.

**Simple summary:** "ආලෝකයෙන් neural network එකක් run කරන්න ඇත්තටම පුළුවන්ද?" කියන ප්‍රශ්නයට **"ඔව්, පුළුවන්!"** කියලා ලෝකයට පළමුවෙන්ම prove කළ landmark paper එක මේකයි.

**PhotonFlow එකට role එක:** Hardware foundation. MZI mesh = compute unit. Noise model source. Energy/latency targets.

**Background:** 2017 ට කලින්, "photonic computing" එක mainly theoretical concept එකක් විතරයි. Papers වල "theoretically possible" කියලා ලියලා තිබුණා, ඒත් ඇත්තටම **real chip** එකක් හදලා, neural network එකක් load කරලා, real task එකක් successfully run කරපු කෙනෙක් **හිටියේ නෑ**. MIT (Massachusetts Institute of Technology) එකේ Yichen Shen ලා ඒක first time change කළා — silicon photonic chip එකක් fabricate කරලා, MZI mesh 56 ක්, vowel recognition (ස්වර හඳුනාගැනීම) task එකක් chip එකේම run කළා.

**Core Idea: MZI Mesh එකෙන් Matrix Multiplication**

**Neural network** layer එකක basic operation එක මොකක්ද? **Matrix multiplication** (matrix = numbers table එකක්, rows × columns; matrix multiplication = two matrices combine කරලා new matrix එකක් ලබා ගන්න ක්‍රමය) එකක් (**linear transform** - data straight line/plane එකකට map කරන operation) + **nonlinearity** (straight-line නොවන function එකක්, complex patterns learn කරන්න essential) එකක්. Shen et al. prove කළේ මේ දෙකම optics වලින් කරන්න පුළුවන් කියලා.

**Matrix decomposition with SVD:**

ඕනම real-valued matrix එකක් M, **SVD (Singular Value Decomposition)** එකෙන් decompose කරන්න පුළුවන්. SVD කියන්නේ **matrix එකක් සරල කොටස් 3 කට කැඩීමේ ක්‍රමයක්** - ඕනම complex matrix එකක් "rotate → scale → rotate" operations 3 කට break කරන්න පුළුවන්:

```
M = U Σ V†
```

මෙතන:
- **U** සහ **V†** → **unitary matrices** (unitary matrix කියන්නේ data "rotate" කරන matrix එකක් - information එක නැති වෙන්නේ නැතුව direction විතරක් change කරනවා, like spinning an object without changing its size). මේවා **cascaded MZI arrays** වලින් implement කරන්න පුළුවන්. හැම MZI එකක්ම 2×2 unitary rotation එකක්.
- **Σ** → **diagonal matrix** (diagonal matrix = හරස් line එකේ values විතරක් තියෙන matrix, scaling/stretching represent කරනවා, අනිත් හැම position එකක්ම zero). Optical attenuators වලින් implement කරනවා.

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

Linear transform එකට පස්සේ nonlinearity එකක් ඕන. Electronics වල ReLU (negative values zero කරන function), sigmoid (values 0-1 range එකට squeeze කරන S-shaped function) use කරනවා. Photonics වල **saturable absorber** එකක් use කරනවා.

Saturable absorber කියන්නේ material එකක් (graphene වගේ) - light intensity වැඩි වෙනකොට **transparent** වෙනවා. Low intensity එකේදී absorb කරනවා, high intensity එකේදී pass through කරනවා. මේක mathematically tanh-like function එකක්:

```
σ(x) ≈ tanh(αx) / α     (α = 0.8 PhotonFlow වලදී)
```

**Experiment — ඇත්තටම Chip එකේ මොනවා වුණාද?**

MIT team එක fabricate කළ chip එකේ **programmable MZIs 56** ක් තිබුණා. මේ MZIs program කරලා (phase angles set කරලා), neural network weight matrix එකක් chip එකට load කළා. ඊට පස්සේ vowel sounds (අ, ඉ, උ, ඒ, ඕ වගේ) chip එකට input කරලා recognize කරන්න try කළා.

Results: chip එකේ **76.7% accuracy** ලැබුණා. ඒ කාලෙම digital computer එකෙන් **91.7%** ලැබුණා. Accuracy gap එක 15% ක් විතර — ඒත් මේක **fundamental limitation** එකක් නෙවෙයි! Gap එක එන්නේ **thermal crosstalk** (ළඟ ඉන්න phase shifters එකිනෙකට heat leak වෙන එක) සහ **limited precision** නිසා. Better chip fabrication එකෙන් මේ gap එක close කරන්න පුළුවන්. MZI fidelity (එක MZI එකක් කොච්චර accurately work කරනවද?) **99.8%** ± 0.003 — nearly perfect!

**Noise Model — PhotonFlow Training එකට ඇයි Critical ද?**

Real chip එකක measurements ගත්තම, **noise sources 4 ක්** identify වුණා. මේ noise sources photonic computing වල **unavoidable** (මඟහරින්න බැරි) — chip design better කළත් completely eliminate කරන්න බෑ. මේ noise sources මොනවාද, ඒවා ඇයි එන්නේ, PhotonFlow ඒවා handle කරන්නේ කොහොමද explain කරමු:

1. **Phase encoding noise** (σ_φ ≈ 5×10⁻³ radians): Phase shifter එකක "θ = 45.000°" set කරන්න try කරනවා. ඒත් actual value 44.95° හෝ 45.04° වෙන්න පුළුවන් — tiny heater එකේ temperature **exactly** control කරන්න physical limitation එකක් තියෙනවා. Thermometer එකක± 0.1° error වගේ.

2. **Thermal crosstalk**: Phase shifter #1 heat කරනකොට, **adjacent** (ළඟම ඉන්න) phase shifter #2 ටත් heat ටිකක් leak වෙනවා. ඒ extra heat එකෙන් shifter #2 එකේ phase value unintentionally shift වෙනවා. Oven එකේ cookies bake කරනකොට ළඟ ඉන්න cookie එකටත් heat affect වෙනවා වගේ.

3. **Photodetection noise** (σ_D ≈ 0.1%): Photodetector output signal එකේ ±0.1% ක variation. Detector එකේ "dynamic range" (minimum to maximum measure කරන්න පුළුවන් range) limit නිසා.

4. **Shot noise**: Quantum physics level එකේ inherent randomness. ආලෝකය **photons** (ආලෝක කණිකා) වලින් හැදිලා — rain drops window එකක hit වෙනවා වගේ, photons detector එකට hit වෙන rate එක perfectly constant නෑ. Second එකකට photons 1000 ක් average ඒත් actual value 997, 1004, 998... random variation. මේක **quantum randomness** — remove කරන්න physically impossible.

**PhotonFlow මේ noise handle කරන්නේ කොහොමද?** Training time එකේදී, Gaussian noise (bell-curve shaped random numbers) model එකට inject කරනවා: shot noise simulate කරන්න σ_s = 0.02, thermal crosstalk simulate කරන්න σ_t = 0.01. මේකෙන් model එක "noisy conditions expect කරන්න" learn කරනවා — real chip එකට deploy කරනකොට noise එක surprise එකක් වෙන්නේ නෑ.

**Energy සහ Speed — ඇයි Photonic Computing Exciting ද?**

Shen 2017 paper එකේ measurements බලමු:

- **Energy:** එක FLOP (Floating Point Operation — multiply/add වගේ ගණිත operation එකක්) එකකට **~5/(m×N) femtojoules**. GPU එකකට compare කරනවනම්, photonic chip එක **100,000x more energy efficient**! Femtojoule = 10⁻¹⁵ joules — unfathomably tiny energy amount.
- **Speed:** Photodetection rate **100 GHz ට වැඩි** — ඒ කියන්නේ **තත්පරයට operations 100 billion**. Electronic networks ට වඩා **100x faster**.

මේ numbers suggest කරන්නේ, photonic chip එකක AI run කරන්න පුළුවන් නම්, GPU එකට වඩා orders of magnitude better performance ලැබෙනවා කියලා.

**PhotonFlow එකට connection — summary:**

Shen 2017 paper එකෙන් PhotonFlow ගන්නේ:
1. **MZI mesh = compute unit** — architecture design Monarch layers force කරන්නේ MZI mesh match වෙන්නයි
2. **Saturable absorber = only allowed optical activation** — ReLU/GELU replace
3. **Noise model = training regularization parameters** — σ_s, σ_t values
4. **Energy/latency targets = success criteria** — sub-fJ, sub-ns benchmarks

---

### 📄 Reference 2: Ning 2024 — Photonic-Electronic Integrated Circuits for AI Accelerators

Shen 2017 paper එකෙන් "photonic computing possible" කියලා prove වුණා. ඒත් ඒක 2017 — ඊට පස්සේ field එක ගොඩක් develop වුණා. **2024 වෙනකොට field එක කොහොමද?** Ning 2024 survey paper එකෙන් ඒ picture එක ලැබෙනවා.

**Paper:** Ning, Zhu, Feng, et al. "Photonic-electronic integrated circuits for high-performance computing and AI accelerators." *J. Lightwave Technol.*, vol. 42, pp. 7834-7859, 2024.

**Simple summary:** Photonic computing field එකේ **encyclopedia** එක වගේ paper එකක්. UT Austin group (University of Texas at Austin) එකෙන් ලියපු, **pages 137 ක්** තියෙන massive survey. Photonic chips ගැන: මොනවා possible ද, specs මොනවාද, limitations මොනවාද, future මොකක්ද — හැමදේම cover කරනවා. PhotonFlow එකේ **hardware parameters** (precision bits, noise levels, energy numbers) හැමදේම **මේ survey එකෙන්** ගන්නේ.

**Key Hardware Specs — PhotonFlow Simulation Parameters එන්නේ මෙතනින්:**

Survey එක photonic hardware specifications exhaustively document කරනවා. PhotonFlow simulation එකට **directly** use කරන numbers මේවා:

- **MZI effective precision: 4-6 bits.** Phase shifter එකකට set කරන්න පුළුවන් distinct angles ගණන 2⁴=16 (4-bit) to 2⁶=64 (6-bit). Float32 (values billions ක්) ට compare කරන්න — massive precision drop. මේ reason නිසා PhotonFlow **4-bit QAT** target කරනවා — worst case එකට prepare.
- **Loss per beamsplitter: ~0.1 dB.** dB (decibel) කියන්නේ signal power loss measure unit එකක් — logarithmic scale (0.1 dB ≈ 2.3% power loss). ආලෝකය beamsplitter එකක් pass through කරනකොට, **ටිකක් absorb/scatter** (ගිලගෙන/විසිරී) වෙනවා. Beamsplitters cascade (ගොඩගැසෙන) කරනකොට total loss accumulate වෙනවා — MZI columns 10 ක් = ~10% total power loss.
- **Detector shot noise: σ = 0.01 - 0.03.** Shen 2017 paper එකේ explain කළ quantum photon counting noise. PhotonFlow σ_s = 0.02 (range එකේ middle) select කරනවා.
- **Thermal crosstalk: nearest-neighbor coupling.** Adjacent phase shifters affect වෙන ප්‍රමාණය. PhotonFlow σ_t = 0.01.
- **Energy per MAC: sub-fJ (optical only).** MAC = Multiply-Accumulate (multiply + add — neural network එකේ basic compute operation). Optical computation alone **sub-femtojoule** — incredibly efficient. ඒත් total system (including DAC/ADC electronic converters) **1-10 fJ/MAC**. PhotonFlow zero O-E-O → DAC/ADC ඕන නෑ → pure optical sub-fJ target.

**O-E-O Bottleneck — Photonic Computing එකේ #1 ප්‍රශ්නය:**

මේ survey එකේ **most important finding** මේකයි: photonic computing speed advantage මුළුමනින්ම **kill** කරන bottleneck එක **O-E-O (Opto-Electronic-Opto) conversion**. මේක මොකක්ද?

Photonic chip එකේ MZI cascade ආලෝකයෙන් matrix multiply **picoseconds** (10⁻¹² seconds) වලින් කරනවා. ඒත් softmax, LayerNorm වගේ operations optically implement කරන්න බෑ — ඒවාට ආලෝකය electricity බවට convert (O→E), electronic circuit එකෙන් process, ආයේ electricity ආලෝකයට convert (E→O) කරන්න ඕන. මේ **conversion alone nanoseconds** (10⁻⁹ seconds) ගන්නවා — MZI computation එකට වඩා **1000x slower!** Super-fast car එකක drive කරනවා ඒත් හැම kilometer එකකම toll booth එකක minutes ගණන් wait කරනවා වගේ.

Survey එක conclude කරනවා: **O-E-O elimination = photonic computing success key**. PhotonFlow **exactly** මේක කරනවා — zero O-E-O conversions. Monarch layers, saturable absorber, divisive power norm — **හැමදේම** optically implement.

**Survey එකේ Broader Findings:**

Ning 2024 field එක analyze කරලා important conclusions 4 ක් draw කරනවා:

1. **Toy-scale problem:** 2024 වෙනකොට publish වෙලා තියෙන photonic NN demonstrations බොහොමයක **neurons කිහිපයක් විතරයි** (Shen 2017: 56 MZIs, Zhu 2026: 4×4 mesh). Real-world models (millions of parameters) scale කරන්න ගොඩක් research ඕන. PhotonFlow CIFAR-10 (32×32) සහ CelebA-64 (64×64) target කරනවා — previous optical demos ට වඩා ලොකු, ඒත් still moderate scale.

2. **Co-design essential:** Standard neural network architecture එකක් "as-is" photonic chip එකකට port කරන්න **බෑ**. Architecture එක hardware constraints **සමඟ** design (co-design) කරන්න ඕන. PhotonFlow **exactly** මේ philosophy follow කරනවා — Monarch layers MZI match, saturable absorber = only activation, divisive power norm = only normalization.

3. **Limited precision acceptance:** Analog photonic computing = limited precision (4-6 bits). Accept it, train for it. QAT (Quantization-Aware Training) = essential technique. PhotonFlow 4-bit QAT adopt කරනවා.

4. **Application-specific ≫ general-purpose:** General-purpose photonic processor බලාගෙන ඉන්නවා වෙනුවට, **specific application** එකකට optimize කළ design better. PhotonFlow = specific application (flow matching generative model) + specific hardware (MZI mesh) = application-specific co-design.

**Simulation Methodology — PhotonFlow Experiment 6 එකට:**

Survey එක describe කරන simulation pipeline එක PhotonFlow directly adopt කරනවා. Steps:

1. **Weight → MZI phases:** Trained weight matrix SVD/Clements decomposition (Reck decomposition එකේ improved version — MZI mesh depth minimize කරනවා, ඒ කියන්නේ light pass through කරන්න ඕන MZI layers අඩු = less optical loss) use කරලා MZI phase angles බවට convert.
2. **Phase quantization:** Continuous phase angles (e.g., 45.37°) → nearest allowed discrete angle (e.g., 45.0° or 47.5° for 4-bit = 16 steps).
3. **Optical loss injection:** Each beamsplitter stage 0.1 dB loss add.
4. **Detector noise:** Gaussian distributed random noise add.
5. **Thermal crosstalk:** Correlated noise — neighboring elements affect (random ඒත් neighbors **related**, independent noise නෙවෙයි).

---

### 📄 Reference 3: Lipman 2023 — Flow Matching for Generative Modeling

References 1 සහ 2 එකෙන් **hardware side** එක cover වුණා — photonic chip එකක මොනවා possible ද, specs මොනවාද, bottleneck මොකක්ද. දැන් **ML (machine learning) side** එකට යමු. PhotonFlow **train** කරන්නේ කොහොමද? ඒ training method එක එන්නේ මේ paper එකෙන්.

**Paper:** Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow matching for generative modeling." *ICLR 2023*.

**Simple summary:** AI model එකක් images generate කරන්න train කරන **new method** එකක් propose කරනවා — existing methods (GANs, diffusion) ට වඩා **simple, stable, efficient**. Meta AI (Facebook AI Research / FAIR) එකෙන් ආවා.

**PhotonFlow එකට role එක:** Training objective — model train කරන loss function/method එක provide කරනවා.

**Flow Matching - Basic Idea:**

**Diffusion models** ගැන අහලා ඇති නේද? (Diffusion model කියන්නේ image එකකට step-by-step noise add කරලා, ඊට පස්සේ "noise එක ඉවත් කරන්නේ කොහොමද" කියලා neural network එකකට teach කරන generative model type එකක්. DALL-E, Stable Diffusion, Midjourney - මේවා හැමදේම diffusion-based.) ඒවාට **variance schedules** (noise add කරන rate/speed එක control කරන schedule - step 1 ට noise ටිකක්, step 100 ට noise ගොඩක් වගේ), **score functions** (data distribution එකේ gradient/slope - "data likelihood වැඩි වෙන direction එක" point කරන function), reverse process කියලා complicated concepts ගොඩක් තියෙනවා.

Flow matching **සරලයි**. Idea එක මේකයි:

1. **Random noise** point එකක් ගන්නවා: `x₀ ~ N(0, I)` (**Gaussian noise** / **Normal distribution** - statistics වල "bell curve" එක. වැඩිම values මැද, extreme values දෙපැත්තට - dice roll එකක 7 ගොඩක් එනවා, 2 හෝ 12 ටිකක් එනවා වගේ pattern එකක්)
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

මේක **simple regression loss** එකක්! (**Regression** කියන්නේ "predict කරන value එක actual value එකට කොච්චර close ද" measure කරන task type එකක් - house price predict කරනවා වගේ continuous value predict කිරීම.) **MSE (Mean Squared Error)** එකක් විතරයි (predict value සහ actual value අතර difference square කරලා average ගන්න ක්‍රමය - difference ² = ලොකු errors වලට ලොකු penalty). Compare කරන්න:
- **GANs** (Generative Adversarial Networks - generator එකක් images හදනවා, discriminator එකක් "real ද fake ද?" judge කරනවා, දෙන්නා fight කරනවා) = **minimax game** (එක player minimize කරනවා, අනිත් player maximize කරනවා - zero-sum game එකක්) → unstable, **mode collapse** (generator එකට variety generate කරන්න බැරි වෙලා, එකම type එකේ output එකක් repeatedly generate කරනවා - e.g., හැම face එකම same face වගේ)
- **Diffusion** = variance schedules, **score matching** (data distribution එකේ gradient estimate කරන training method) → complex
- **Flow matching** = just MSE regression (simple, stable)

**Mathematical Framework (ගැඹුරට):**

Flow matching theory එක **Continuous Normalizing Flows (CNFs)** මත පදනම් වෙනවා. (**Normalizing Flow** කියන්නේ simple distribution එකක් - like Gaussian bell curve - complex distribution එකකට step-by-step transform කරන invertible function chain එකක්. "Normalizing" = probability conservation, "Flow" = continuous transformation.)

**Vector field** `v_t(x)` එකක් define කරනවා (**vector field** = space එකේ හැම point එකකටම direction arrow එකක් assign කරන function, wind map එකක් වගේ - හැම location එකේදීම wind direction එක පෙන්නනවා). ඒකෙන් **flow** `φ_t(x)` එකක් generate වෙනවා:

```
d/dt φ_t(x) = v_t(φ_t(x)),    φ₀(x) = x        (ODE)
```

මේක **ODE (Ordinary Differential Equation)** එකක් - "time එක ටිකක් ගතවෙනකොට x කොච්චරක් change වෙනවද?" describe කරන equation. GPS navigation එකේ "current position එකේදී velocity මොකක්ද?" කියන එක solve කරනවා වගේ.

මේ flow එක simple distribution `p₀` (Gaussian noise) එකෙන් complex distribution `p₁` (real data) එකට transform කරනවා.

**Problem:** Target vector field `u_t(x)` directly compute කරන්න **intractable** (mathematically possible ඒත් practically compute කරන්න **impossible** - compute time/memory ප්‍රමාණවත් නෑ).

**Solution - Conditional Flow Matching (CFM):**

Per-sample **conditional** paths use කරනවා:

```
L_CFM(θ) = E_{t, q(x₁), p_t(x|x₁)} ‖v_t(x) - u_t(x|x₁)‖²
```

**Theorem 2 (crucial result):** FM සහ CFM objectives වල **gradients** (gradient = function එකේ slope/direction - "loss decrease වෙන direction එක" point කරන arrow; ∇ symbol = gradient; training = gradient direction එකට weights update කරනවා) **identical**! ∇_θ L_FM = ∇_θ L_CFM

මේකෙන් තේරෙන්නේ: intractable FM **optimize** (minimize/maximize - loss minimize කිරීම = best weights find කිරීම) කරනවා වෙනුවට, **tractable** (practically compute කරන්න possible) CFM optimize කළාම **exactly same result** එක ලැබෙනවා.

**Optimal Transport (OT) paths:**

Mean සහ standard deviation linearly change වෙනවා:

```
μ_t(x) = t × x₁
σ_t(x) = 1 - (1 - σ_min) × t
```

σ_min → 0 වෙනකොට, target simplify වෙනවා: `u_t ≈ x₁ - x₀`

OT paths **straight lines**. Diffusion paths **curved**. Straight paths = **fewer ODE steps** at **inference** (inference = trained model එක use කරලා **new data generate/predict** කරන process එක - training = ඉගෙනීම, inference = ඉගෙන ගත් දේ use කිරීම). Photonic chip එකේ step එකක් sub-nanosecond. Steps අඩු = faster generation.

**ඇයි Flow Matching PhotonFlow-ට perfect?**

1. **Architecture-agnostic**: Loss එක `v_θ` ඇතුළේ මොනවා තියෙනවද කියලා care කරන්නේ නෑ. U-Net, Transformer, Monarch - ඕනම architecture එකක් use කරන්න පුළුවන්.
2. **Stable training**: GAN minimax game එකක් නෑ. Noise injection (shot noise, thermal crosstalk) training harder කරන නිසා, stable objective එකක් **critical**.
3. **Few ODE steps**: Straight OT paths = fewer steps = faster photonic inference.
4. **Simple**: MSE regression. No variance schedules. No score functions.

---

### 📄 Reference 4: Peebles 2023 — Scalable Diffusion Models with Transformers (DiT)

Flow matching training method ලැබුණා (Reference 3). ඒත් training method **alone** model එකක් නෙවෙයි — architecture (model එකේ internal structure/design) එකක්ත් ඕන. GPU එකේ generative models වල **best architecture** එක මොකක්ද? DiT. ඒක **PhotonFlow compare කරන baseline** (reference point) එක, ඒ වගේම DiT එකෙන් design ideas කිහිපයක් **borrow** කරනවා.

**Paper:** Peebles, Xie. "Scalable diffusion models with transformers." *ICCV 2023*.

**Simple summary:** Generative image models (diffusion/flow based) වල backbone එක U-Net (older CNN architecture) එකෙන් **Transformer** එකකට change කළා. Results: state-of-the-art image generation quality. ඒත් Transformer architecture එක photonic chip එකක run කරන්න **impossible** — ඒ ප්‍රශ්නය solve කරන්නයි Monarch matrices (Reference 5) ඕන.

**PhotonFlow එකට role එක:** (1) GPU baseline — compare against. (2) Design ideas source — time conditioning, zero-initialized residuals. DiT තමයි modern diffusion/flow models වල **standard backbone**. GPU එකක run කරනවනම්, DiT best choice එක. Photonic chip එකක run කරනවනම්, DiT **impossible**.

**DiT Architecture:**

DiT එක **Vision Transformer (ViT)** based (Transformer architecture = attention mechanism use කරන neural network type එකක්, originally text processing වලට හදලා, ViT = vision/images වලට adapt කළ version):

1. **Patchify**: Image (e.g., 256×256) → **VAE latent** (VAE = Variational Autoencoder - image එක compressed representation එකකට encode කරන network; "latent" = compressed/hidden representation, 256×256 image එක 32×32×4 latent space එකකට compress කරනවා) → **patches** (latent image එක small squares/කොටු වලට කපනවා) → **tokens** (හැම patch එකක්ම vector එකකට convert කරනවා - NLP වල word = token, vision වල patch = token)
2. **DiT Blocks** (N blocks):
   - Multi-head **softmax self-attention** (input එකේ හැම token එකක්ම අනිත් **හැම token** එකක් සමඟ compare කරනවා - "මොනවා related ද?" find කරනවා. "Multi-head" = එකම process එක parallel heads කිහිපයකින් - එකක් color focus, එකක් shape focus වගේ)
   - **LayerNorm** (adaptive - **adaLN-Zero**: time step එකට adapt වෙන LayerNorm version එකක්)
   - **MLP** (Multi-Layer Perceptron - basic "fully connected" neural network layers කිහිපයක්, every input neuron → every output neuron connected) with **GELU** activation (GELU = smooth version of ReLU, `x × Φ(x)` where Φ = **Gaussian CDF** - Gaussian distribution එකේ "x ට වඩා අඩු values එන probability එක" දෙන function)
   - **Residual connections** (layer එකේ input එක, output එකට **directly add** කරනවා - shortcut path එකක්. මේකෙන් deep networks train කරන්න ලේසි වෙනවා, gradient vanishing problem එක avoid කරනවා)
3. **Decoder**: tokens → image

**adaLN-Zero Conditioning (ගැඹුරට):**

Time step `t` සහ class label `c` inject කරන්නේ normalization layer එකට:

```
(γ₁, β₁, α₁, γ₂, β₂, α₂) = MLP(t_emb + c_emb)

x = x + α₁ × Attention(γ₁ × LayerNorm(x) + β₁)
x = x + α₂ × FFN(γ₂ × LayerNorm(x) + β₂)
```

**α parameters zero-initialized** → Training start එකේදී **each block = identity function** (identity function = input එකම output - `f(x) = x`, කිසිම change එකක් නෑ). මේකෙන් deep models (layers ගොඩක් තියෙන models) stable train වෙනවා - start එකේ blocks "do nothing" කරනවා, ඊට පස්සේ slowly learn කරනවා.

**Scaling result: FID 2.27** on ImageNet 256×256 (state-of-the-art at the time).

**ඇයි DiT photonic chip එකක run කරන්න බෑ?**

| DiT Component | Problem |
|---|---|
| **Softmax attention** | `softmax(QK^T/√d)V` - Q (Query), K (Key), V (Value) = input එකෙන් derive කරන matrices 3 ක්. QK^T = "queries සහ keys match වෙනවද?" check කරනවා. Softmax = probabilities බවට convert. V = actual information extract කරනවා. exponential function (`e^x`), sum, division ඕන. Optical domain එකේ implement කරන්න බෑ. |
| **LayerNorm** | Mean, variance compute → division, square root. Not photonic. |
| **GELU** | `x × Φ(x)` where Φ is **Gaussian CDF** - `erf()` function (**erf** = error function - Gaussian bell curve එකට related mathematical function එකක්, closed-form solution (simple equation එකකින් directly calculate කරන්න පුළුවන් answer එකක්) නැති special function). No optical analog. |

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

### 📄 Reference 5: Dao 2022 — Monarch: Expressive Structured Matrices

DiT එක GPU එකේ best ඒත් photonic chip එකක impossible — softmax attention, LayerNorm, GELU non-photonic. ප්‍රශ්නය: **attention replace** කරන්නේ **මොනවාට** ද? Softmax attention නැතුව, MZI mesh එකේ directly run කරන්න පුළුවන්, ඒත් generation quality maintain කරන layer type එකක් ඕන. **Monarch matrices** ඒ answer එක.

**Paper:** Dao, Chen, et al. "Monarch: Expressive structured matrices for efficient and accurate training." *ICML 2022*.

**Simple summary:** Stanford researchers (Tri Dao et al.) new matrix type එකක් propose කරලා — **Monarch matrix** — GPU training **2x faster** ඒත් accuracy **same**. ඔවුන්ගේ intention GPU efficiency. ඒත් **coincidentally**, Monarch matrix එකේ computation structure MZI mesh එකේ physical structure එකට **exactly match** — මේ coincidence එක PhotonFlow entire project depend වෙන key insight එක.

**PhotonFlow එකට role එක:** Bridge paper. Attention replacement = Monarch layers. MZI mesh native structure. Monarch matrices තමයි attention replace කරන layer type එක, **ඒ වගේම** MZI mesh එකේ computation graph එකට **exactly match** වෙන layer type එකත් එකමයි!

**Monarch Matrix Definition:**

n×n matrix එකක් (n = m²):

```
M = P L Pᵀ R
```

මෙතන:
- **R** = **block diagonal** (**block-diagonal matrix** = matrix එක diagonal line එකේ small matrix blocks වලට split වෙනවා, blocks අතර values = zero. E.g., 8×8 matrix එකක්, 2×2 blocks 4 කින් හැදිලා - off-diagonal areas හැමදේම zero): `diag(R₁, R₂, ..., R_m)` - m blocks, each m×m
- **L** = block diagonal: `diag(L₁, L₂, ..., L_m)` - m blocks, each m×m
- **P** = **fixed permutation** (**permutation** = elements වල order/sequence change කරන operation - cards shuffle කරනවා වගේ; "stride permutation" / "perfect shuffle" = specific pattern එකකට reorder, reshape m×m, transpose, flatten)

**Operations:**

```
1. Reshape x → m×m matrix X
2. Multiply each row by R_i:    X' = R × X         (m independent m×m multiplies)
3. Transpose:                    X'' = (X')ᵀ         (permutation P - FREE!)
4. Multiply each row by L_i:    Y = L × X''          (m independent m×m multiplies)
5. Flatten Y → y
```

**Parameters:** 2×n×√n (**dense matrix** (හැම element එකක්ම non-zero, fully filled matrix එකක්) එකක n² parameters ට compare කරන්න - ගොඩක් less!)
**FLOPs:** O(n^{3/2}) (**FLOPs** = Floating Point Operations - computer එකකට කරන්න ඕන ගණිත operations ගණන, multiply/add operations count එක. Operations අඩු = faster + efficient)

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
- **Convolutions** (image processing වල sliding filter/window එක - image එකේ edge detection, blur, sharpen වගේ operations. CNN = Convolutional Neural Network)
- **Hadamard transform** (±1 values only matrix transform - fast, simple, signal processing වල use කරනවා)
- **Toeplitz matrices** (diagonal line එකක values constant වෙන matrix - time series, signal processing වල naturally appear වෙනවා)

(MM*)² (Monarch layers 4 product) represent කරන්න පුළුවන්:
- **Fourier transform** (signal එකක් frequency components වලට decompose කරන transform - music එකේ "bass, treble, mid" separate කරනවා වගේ, image එකේ "patterns at different scales" separate කරනවා)
- **Discrete sine/cosine transforms** (Fourier transform එකේ real-valued versions - JPEG image compression එකේ core math එක මේකයි)

PhotonFlow block එකක Monarch L සහ R use කරනවා → MM* class → attention replace කරන්න ප්‍රමාණවත් expressiveness.

**GPU Results (Monarch original paper):**

- ViT/MLP-Mixer on ImageNet: **2x faster**, same accuracy
- GPT-2 on Wikitext-103: **2x faster**, same **perplexity** (language model එකක quality measure කරන metric - "next word predict කරන්නේ කොච්චර uncertain ද?" measure කරනවා. Perplexity 10 = model එකට next word options 10 ක් equally likely. Lower = better)
- PDE solving, MRI reconstruction: Monarch beats hand-crafted Fourier transforms

**Projection (Theorem 1):**

Dense matrix A → closest Monarch matrix M **closed-form solution** (step-by-step iterate කරන්නේ නැතුව, **equation එකකින් directly answer එක calculate** කරන්න පුළුවන්, like quadratic formula) එකක් තියෙනවා (rank-1 SVD per batch slice). මේකෙන් **pretrained DiT weights Monarch-ට project** කරන්නත් පුළුවන් theoretically.

---

### 📄 Reference 6: Meng 2022 — ButterflyFlow

Monarch matrices attention replace කරන්න propose කළා (Reference 5). ඒත් important ප්‍රශ්නයක් ඉතිරියි: **"structured matrices use කරලා generative model එකක ඇත්තටම image generate කරන්න පුළුවන්ද?"** Attention (dense, softmax-based) replace කළාම generation quality maintain වෙනවද? ButterflyFlow paper එක මේ ප්‍රශ්නයට **experimental evidence** දෙනවා.

**Paper:** Meng, Zhou, Choi, Dao, Ermon. "ButterflyFlow: Building invertible layers with butterfly matrices." *ICML 2022*.

**Simple summary:** Stanford group එක (Tri Dao **co-author** on both Monarch and Butterfly papers!) **butterfly matrices** (Monarch matrices වල close relative) use කරලා **normalizing flow** (simple distribution → complex distribution transform කරන invertible generative model type එකක්) generative model එකක් build කළා. Results: dense matrix baselines **match** කරනවා ඒත් parameters **half**, speed **better**. මේකෙන් "structured matrices generation-capable" kියලා prove වෙනවා.

**PhotonFlow එකට role එක:** Validation — "structured matrices generative models වලට work කරනවා" prior art evidence. Reviewer එක "structured matrices generation වලට use කරලා තියෙනවද?" ඇහුවොත්, මේ paper point කරන්න පුළුවන්.

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

**bpd (bits per dimension)** = generative model quality metric එකක්. Image එකේ **එක pixel එකක්** represent කරන්න average bits කීයක් ඕනද measure කරනවා. Lower = better compression = better model.

| Dataset | ButterflyFlow (bpd) | Glow (bpd) |
|---|---|---|
| MNIST | **1.05** | 1.05 |
| CIFAR-10 | **3.33** | 3.35 |
| ImageNet 32 | **4.09** | 4.09 |

MIMIC-III (structured medical data): **-27.92 NLL** (**NLL** = Negative Log-Likelihood - model එක data "predict කරන්නේ කොච්චර හොඳට ද" measure කරන metric. More negative = better. Log = logarithm, likelihood = probability) vs Glow's worst performance → **2.4x improvement, half the parameters!**

**PhotonFlow ඇයි Monarch choose කළේ Butterfly නෙවෙයි?**

ButterflyFlow prove කරනවා structured matrices generation-capable. ඒත් PhotonFlow **Monarch** choose කරනවා **Butterfly** නෙවෙයි. ඇයි?

1. **Chip layout simplicity:** Monarch = factors **2 ක් විතරයි** → MZI columns **2 ක්** (chip එකේ physically compact, simple routing). Butterfly = factors **log(n)** → MZI columns log(n) ක් (n=256 නම් columns 8, n=1024 නම් columns 10 — chip layout complex, more physical space).

2. **Optical loss accumulation:** හැම MZI column/beamsplitter stage එකකටම 0.1 dB loss. Columns 2 ක = loss 0.2 dB ≈ 4.5% total. Columns 8 ක = loss 0.8 dB ≈ 17% total! Monarch = **significantly less optical power loss**.

3. **GPU training efficiency:** Monarch operations = **batched matrix multiplies** (GPU එකේ highly optimized). Butterfly = sequential sparse operations (GPU unfriendly, slow). Training time Monarch **2x faster**.

4. **Generation expressiveness:** ButterflyFlow results confirm — both Butterfly and Monarch generation quality **similar**. Expressiveness difference small. ඒ නිසා simpler, more efficient Monarch sufficient.

---

### 📄 Reference 7: Jacob 2018 — Quantization and Training of Neural Networks

References 1-6 එකෙන් hardware, training method, architecture, validation cover වුණා. ඒත් critical challenge එකක් ඉතිරියි: MZI phase shifters **4-6 bit precision** විතරයි. Normal neural network **32-bit** precision. 32 → 4 bits ට drop කළොත් accuracy **dramatically crash** වෙනවා. මේ ප්‍රශ්නයට solution එක **QAT (Quantization-Aware Training)** — ඒ technique එක introduce කරපු paper මේකයි.

**Paper:** Jacob, Kligys, et al. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." *CVPR 2018*.

**Simple summary:** Google එකේ paper එකක්. Neural network weights **low-precision integers** (full precision floating point වෙනුවට) ලෙස represent කරන්න, accuracy **minimal loss** එකෙන් — ඒක කරන්නේ training time එකේදීම precision loss simulate කරලා, model එකට "low precision එකට adapt වෙන්න" ඉගැන්වීමෙන්. TensorFlow Lite quantization standard මේ paper එකෙන් ආවා. **PhotonFlow** මේ same technique MZI phase shifter precision limit එකට adapt කරන්න use කරනවා.

**PhotonFlow එකට role එක:** QAT methodology. 4-bit hardware precision adapt කරන training strategy.

**Quantization Scheme:**

Real value `r` සහ quantized value `q` අතර mapping:

```
r = S × (q - Z)
```

- **S** (scale): positive real number
- **Z** (zero-point): integer - real zero exactly representable
- **q**: unsigned 8-bit integer (original paper), **4-bit** (PhotonFlow)

**QAT Training Process:**

1. Weights/activations **float32** store කරනවා, normal **backprop** (backpropagation - neural network training එකේ core algorithm: output error එක calculate කරලා, ඒ error එක **backwards** network layers හරහා propagate කරනවා - හැම weight එකක් error එකට contribute කළේ කොච්චරද calculate කරලා, weights adjust කරනවා)
2. **Forward pass** එකේ (input → layers හරහා → output යන direction එක, prediction generate කරන step එක) **fake quantization nodes** insert: float → quantize → dequantize → float
3. Backprop: **straight-through estimator** (quantize operation එකේ gradient mathematically **zero** (step function). ඒත් zero gradient = learning stop! Straight-through trick එක = backward pass එකේ quantize operation එක **ignore** කරලා gradient "straight through" pass කරනවා, as if quantization happened නැති වගේ) - quantize/dequantize operation through identity gradient
4. **Batch norm** (Batch Normalization - training batch එකේ statistics use කරලා values normalize කරන technique) **fold** into conv weights (batch norm parameters convolution weights වලට **merge** කරනවා inference time efficiency වලට)

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

Model එක පළමුව noise-robust වෙනවා, **ඊට පස්සේ** quantization apply කරනවා. Both simultaneously add කළොත් training too noisy වෙලා **converge වෙන්නේ නෑ** (convergence = training process එකේ loss value stable වෙලා, model එක "ඉගෙන ගෙන ඉවරයි" කියන state එකට reach වෙනවා. Converge නොවීම = loss keep fluctuating/increasing = model learn කරන්නේ නෑ).

---

### 📄 Reference 8: Ning 2025 — StrC-ONN (Structured Compression for Optical NNs)

PhotonFlow approach එක valid ද? Structured matrices + noise-aware training + photonic hardware = works ද? මේ ප්‍රශ්නයට **independent validation** — completely separate research group එකක්, separately same conclusion එකට arrive වුණා. StrC-ONN paper එක ඒ validation.

**Paper:** Ning, Zhu, Feng, et al. "Hardware-efficient photonic tensor core: Accelerating deep neural networks with structured compression." *Optica*, vol. 12, 2025.

**Simple summary:** UT Austin group (Ning 2024 survey paper එකත් ලිව්වේ **same group**) pretrained neural networks **structured matrices** (block-circulant) use කරලා **compress** කරලා photonic hardware එකේ efficient ව run කරන method එකක් develop කළා. PhotonFlow වගේම, ඔවුන්ගේ core insight: **dense (full) matrices photonic hardware වල impractical → structured matrices = solution**. PhotonFlow වලින් independently, same conclusion!

**PhotonFlow එකට role එක:** Independent validation. "Structured + noise-aware = works" confirm කරනවා. Paper review එකේදී "is your approach reasonable?" ප්‍රශ්නයට "another group independently validated it" answer.

**Block-Circulant Compression — මොකක්ද?**

StrC-ONN use කරන structured matrix type එක **block-circulant**. "Circulant" meaning තේරුම් ගන්න, simple example එකක් බලමු. හිතන්න first row `[a, b, c]` කියලා. Circulant matrix එකේ **හැම row** එකක්ම first row එකේ **shifted version** එකක්:

```
Row 1: [a, b, c]     ← original
Row 2: [c, a, b]     ← shift right by 1
Row 3: [b, c, a]     ← shift right by 2
```

Full circulant matrix:
```
C = [c₀   c_{k-1}  c_{k-2}  ...  c₁  ]
    [c₁   c₀       c_{k-1}  ...  c₂  ]
    [c₂   c₁       c₀       ...  c₃  ]
    [...]
```

**Key insight:** Matrix එක **first row එකෙන් fully define** වෙනවා! n×n matrix එකකට normally n² parameters ඕන, ඒත් circulant matrix එකකට **n parameters** විතරයි. Massive compression! "Block-circulant" = matrix එක blocks වලට divide කරලා, **each block** circulant. Parameters: **O(n)** — Monarch O(n√n) ට වඩාත් compressed!

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

### 📄 Reference 9: Zhu/Jiang 2026 — Optical Neural Network for Generative Models

References 1-8 එකෙන් PhotonFlow **build** කරන්න ඕන building blocks හැමදේම cover වුණා. ඒත් important ප්‍රශ්නයක්: **වෙනත් කවුරුවත් photonic chip එකක generative model එකක් run කරලා තියෙනවද?** Answer: ඔව්, **Jiang/Zhu 2026** — **first ever demonstration** of a generative model on real photonic hardware. **මේක PhotonFlow එකේ primary competitor**.

**Paper:** Jiang, Wu, Cheng, Dong. "A fully real-valued end-to-end optical neural network for generative model." *Frontiers of Optoelectronics*, 2026.

**Simple summary:** Chinese research group එකක් **real fabricated silicon photonic chip** එකක **GAN** (Generative Adversarial Network) run කරලා MNIST handwritten digit "7" **8×8 pixel resolution** එකේ generate කළා. First ever optical generative model on real hardware! ඒත් — **8×8 resolution** only, GAN instability, no noise-aware training, no QAT. PhotonFlow **significantly improve** කරනවා මේ baseline එක මත.

**PhotonFlow එකට role එක:** Primary comparison baseline. "Before us, the best photonic generation was this — we improve on it in these ways..."

**Hardware — ඇත්තටම Chip එක කොහොමද?**

Jiang et al. fabricate කළ chip එක relatively small: **4×4 MZI mesh** (first linear layer — inputs 4 × outputs 4), **42 phase shifters** (each needing 20 mW for a full π phase shift), **2×4 MZI mesh** (second linear layer). Real **silicon photonic chip** — simulation නෙවෙයි, actual fabricated hardware.

**Key Innovation — Real-Valued Encoding:**

Previous optical neural networks **complex domain** (amplitude + phase) එකේ work කළා. Light signal එකක amplitude (strength) **සහ** phase (wave position) දෙකම compute result represent කරනවා. ඒත් final answer real-valued number එකක් ගන්න electronic post-processing ඕන වුණා. Jiang et al. clever trick එකක් use කරනවා: **dual micro-ring modulators (MRMs)**. Wavelength දෙකක් use: λ_n+ = positive component represent, λ_n- = negative component represent. Final result = difference (differential photocurrent) = **real-valued number**. Electronic post-processing minimize!

**Results:**

ඔවුන්ගේ chip එකේ tasks 2 ක් run කළා. **Iris classification** (flower measurements classify කරන classic ML dataset) — chip එකේ **98% accuracy**. Impressive! ඒත් generative task එකේ results more modest: **GAN** training (off-chip, GPU එකේ) + inference (on-chip) — MNIST digit **"7" only**, resolution **8×8 pixels** (ගොඩක් tiny! Compare: PhotonFlow target CIFAR-10 = 32×32 = 16x more pixels). Latency: **~1.76 ns** per inference. Energy: **37 pJ** per operation.

**GAN vs Flow Matching (PhotonFlow advantage):**

| Optical GAN (Jiang 2026) | PhotonFlow |
|---|---|
| GAN = minimax game (unstable) | CFM = regression (stable) |
| Scale: 8×8 images | Scale: 32×32 (CIFAR), 64×64 (CelebA) |
| No noise-aware training | Shot noise + thermal crosstalk regularization |
| No QAT | 4-bit QAT fine-tuning |
| No FID reported (quality?) | FID within 10% of GPU baseline |

**GAN Training on Noisy Hardware = BAD IDEA:**

GAN training GPU එකේ වුණත් unstable. **Generator** (fake images හදන network) සහ **discriminator** (real vs fake judge කරන network) fight කරනවා - **mode collapse** risk (generator එක **diversity අහිමි** වෙලා, හැම වෙලාවෙම එකම type output generate කරනවා - e.g., 10 digit types ඉගෙන ගන්න ඕන ඒත් "7" විතරක් generate කරනවා). **Photonic noise add කරනකොට** situation **worse** වෙනවා.

Flow matching = simple regression. **Loss** (loss = model එකේ prediction error measure කරන number එක - training goal = loss minimize කිරීම, lower loss = better model) **monotonically decreases** (monotonically = එක direction එකටම, up-down fluctuation නැතුව steady decline). **Noise injection tolerance high**.

**PhotonFlow vs Optical GAN - Clear Improvements:**

1. **Stable training** (flow matching vs GAN)
2. **Larger scale** (CIFAR-10, CelebA-64 vs 8×8 MNIST)
3. **Hardware-aware training** (noise regularization)
4. **Quantization-aware** (4-bit QAT)
5. **Quantitative quality** (FID reported vs not)

---

## 4. References එකිනෙකට Connect වෙන්නේ කොහොමද?

References 9 individually deeply explain කළා. ඒත් මේවා **isolated (වෙන වෙනම) papers නෙවෙයි** — ඒවා **interconnected story** (එකිනෙකට බැඳුණු කතාවක්) එකක්. එක paper එකක contribution එක තවත් paper එකක result එක මත build වෙනවා. මේ section එකේදී ඒ connections visualize කරමු.

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

**Cascading:** MZI multiple cascade කරලා **ඕනම N×N unitary matrix** එකක් implement කරන්න පුළුවන් (**Reck decomposition theorem** - ඕනම unitary matrix එකක් 2×2 rotation sequences එකකට decompose කරන්න පුළුවන් කියන mathematical proof. හැම 2×2 rotation එකක්ම = එක MZI එකක්, ඒ නිසා MZI cascade = ඕනම unitary matrix).

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
| **5** | Best config on CelebA-64 | Exp. 1 | FID, **IS** (Inception Score - generated image quality + diversity measure; higher = better; Inception network එකෙන් classify කරලා confidence check කරනවා) |
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
| **SVD** (Singular Value Decomposition) | M = UΣV† - ඕනම matrix එකක් "rotate → scale → rotate" operations 3 කට decompose කරන ක්‍රමය |
| **adaLN-Zero** | Adaptive LayerNorm with zero-initialized gates - time/class info inject කරන conditioning method |
| **Gradient** | Function එකක slope/direction - "loss decrease වෙන direction එක" - backprop එකේ core concept |
| **Epoch** | Training data set එක සම්පූර්ණයෙන් එක වරක් process කිරීම |
| **Inference** | Trained model එක use කරලා new predictions/generations කිරීම |
| **Latent space** | Data එකේ compressed/hidden representation - image 256×256 → latent 32×32 වගේ |
| **Token** | Data එකේ basic unit - NLP වල word, vision වල image patch |
| **Loss function** | Model error measure කරන function - training goal = loss minimize කිරීම |
| **Convergence** | Training process එක stable වෙලා model "learned" state එකට reach වීම |
| **Fine-tuning** | Pre-trained model එක additional training ටිකකින් new task එකකට adapt කිරීම |
| **Regression** | Continuous value predict කරන ML task type (classification = category predict) |
| **O-E-O conversion** | Optical → Electronic → Optical signal conversion (bottleneck!) |
| **Shot noise** | Quantum photon counting noise |
| **Thermal crosstalk** | Adjacent phase shifters heat interference |
| **DAC** (Digital-to-Analog Converter) | Digital signal → analog voltage (for phase shifters) |
| **ADC** (Analog-to-Digital Converter) | Analog signal → digital value (from photodetectors) |
| **WDM** (Wavelength Division Multiplexing) | Different wavelengths use කරලා parallel computation |

---

> **Final Note:** මේ document එක PhotonFlow research paper එකේ සියලුම references deep understanding එකක් ලබා දීමට සකස් කරන ලදී. Technical accuracy maintain කරමින්, concepts understandable way එකකින් explain කරන්න උත්සාහ කරා. Questions තිබ්බොත්, original papers refer කරන්න - paper/lit-review/pdfs/ folder එකේ PDF files හැමදේම available.
