# Lipman 2023 - Flow Matching for Generative Modeling

**Citation:** Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow matching for generative modeling." ICLR 2023. arXiv:2210.02747

**Affiliations:** Meta AI (FAIR) and Weizmann Institute of Science.

**Why it matters for us:** This is the paper that gave us the training objective. We use it as is. The clever part of PhotonFlow is the network architecture, not the loss. Flow matching is the loss.

## The one big idea

Diffusion models work, but their training is awkward. You have to think about variance schedules, score functions, and a noisy reverse process. Flow matching skips all of that.

Instead of learning to denoise, you learn a velocity field. Pick a starting point `x_0` from random noise, pick an endpoint `x_1` from your dataset, and draw a straight line between them. Train a neural network `v_theta(x_t, t)` to predict the direction of travel along that line at every point in time. That is the whole training signal.

At sample time you start from random noise and integrate the learned velocity field forward in time using a standard ODE solver.

## The mathematical framework

### Continuous Normalizing Flows

A time-dependent vector field `v_t(x)` defines a flow `phi_t(x)` via the ODE:

```
d/dt phi_t(x) = v_t(phi_t(x)),    phi_0(x) = x           (eqs. 1-2)
```

This flow transports a simple prior density `p_0` (e.g., standard Gaussian) to a data-approximating density `p_1` via the push-forward equation:

```
p_t = [phi_t]* p_0                                         (eq. 3)
```

where the push-forward is defined by the change of variables:

```
[phi_t]* p_0(x) = p_0(phi_t^{-1}(x)) * |det (d phi_t^{-1} / dx)(x)|    (eq. 4)
```

### The Flow Matching (FM) objective

Given a target probability path `p_t(x)` and a corresponding vector field `u_t(x)` that generates `p_t`, the FM objective is:

```
L_FM(theta) = E_{t, p_t(x)} ||v_t(x) - u_t(x)||^2        (eq. 5)
```

where `t ~ U[0,1]` and `x ~ p_t(x)`. This is simple: regress the neural network vector field `v_t` onto the target `u_t`. But it is **intractable** because we do not know `u_t` in closed form.

### The Conditional Flow Matching (CFM) objective

The key insight: construct the intractable marginal path from tractable per-sample conditional paths. For each data sample `x_1`, define a conditional path `p_t(x|x_1)` that starts at noise and concentrates around `x_1` at `t=1`. The marginal path is then:

```
p_t(x) = integral p_t(x|x_1) q(x_1) dx_1                 (eq. 6)
```

The CFM objective replaces the intractable marginal with tractable conditionals:

```
L_CFM(theta) = E_{t, q(x_1), p_t(x|x_1)} ||v_t(x) - u_t(x|x_1)||^2    (eq. 9)
```

**Theorem 2 (the crucial result):** The FM and CFM objectives have identical gradients:

```
nabla_theta L_FM(theta) = nabla_theta L_CFM(theta)
```

This means optimizing the tractable CFM is equivalent to optimizing the intractable FM. We never need access to the marginal vector field or the marginal probability path.

### Gaussian conditional probability paths

The paper considers a general family of Gaussian conditional paths:

```
p_t(x|x_1) = N(x | mu_t(x_1), sigma_t(x_1)^2 I)         (eq. 10)
```

with boundary conditions: `mu_0(x_1) = 0`, `sigma_0(x_1) = 1` (start at standard Gaussian) and `mu_1(x_1) = x_1`, `sigma_1(x_1) = sigma_min` (concentrate around data at t=1).

The corresponding flow map is:

```
psi_t(x) = sigma_t(x_1) * x + mu_t(x_1)                  (eq. 11)
```

and the conditional vector field that generates this path (Theorem 3):

```
u_t(x|x_1) = (sigma'_t(x_1) / sigma_t(x_1)) * (x - mu_t(x_1)) + mu'_t(x_1)    (eq. 15)
```

where primes denote time derivatives.

### Optimal Transport (OT) conditional paths -- what we use

The OT path is the simplest choice: mean and std change linearly in time:

```
mu_t(x) = t * x_1
sigma_t(x) = 1 - (1 - sigma_min) * t                      (eq. 20)
```

Plugging into Theorem 3 gives the OT conditional vector field:

```
u_t(x|x_1) = (x_1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t)    (eq. 21)
```

When `sigma_min -> 0`, this simplifies to the straight-line target that our paper draft uses:

```
u_t(x|x_1) ~ x_1 - x_0        (simplified OT target)
```

This is the form in our CFM loss (eq. 1 of PhotonFlow paper draft). The OT paths form **straight line trajectories** from noise to data, whereas diffusion paths are curved. Straight paths need fewer ODE integration steps at inference, which matters when each step runs on a photonic chip.

## The loss as used in PhotonFlow

In PhotonFlow the network `v_theta` is a stack of 6 to 8 PhotonFlow blocks, each containing:

1. Monarch L
2. Monarch R
3. Optical activation `sigma(x) = tanh(alpha*x)/alpha`
4. Divisive power normalization `x / (||x||_2 + eps)`
5. Time embedding added in

The CFM loss in PhotonFlow (eq. 1 of our paper draft):

```
L(theta) = E_{t, x_0, x_1} [ ||v_theta(x_t, t) - (x_1 - x_0)||^2 ]
```

where `x_t = (1 - t) * x_0 + t * x_1` is the linear interpolation (the OT path with sigma_min = 0).

The loss does not know or care about the photonic architecture inside `v_theta`. We can add shot noise and thermal crosstalk during the forward pass, and the loss remains the same simple regression.

## Why we picked this over diffusion or GANs

- **vs GANs:** flow matching has a stable training objective. There is no minimax game and no mode collapse. The optical GAN of [[Zhu 2026 - Optical NN for Generative Models]] that we benchmark against has exactly the kind of instability we want to avoid.
- **vs diffusion:** the loss is simpler and there is no schedule to tune. The straight-line OT paths mean fewer integration steps at sample time, which is critical when each step runs on a photonic chip.
- **vs likelihood-based models:** flow matching gives you a generative model without a tractable density, but we do not care about densities. We care about samples and FID.

## What flow matching does NOT specify

This is the gap PhotonFlow fills. Flow matching tells you:

- what loss to minimize
- what the network's input and output shapes are
- how to sample at the end (integrate the learned vector field with an ODE solver)

It tells you nothing about what the network should look like inside. The original paper used a U-Net. [[Peebles 2023 - DiT]] later showed you can use a transformer. We show you can use stacked Monarch layers, which is what makes our model run on photonic hardware.

So the network architecture is a free design choice and that is exactly the choice we are exploiting.

## What we keep, what we change

| Lipman 2023 element | What we do with it |
|---|---|
| CFM loss (eq. 9, simplified to OT form) | Keep exactly as is |
| OT linear interpolation paths (eq. 20) | Keep |
| ODE sampler at inference | Keep |
| U-Net or transformer backbone | **Replace** with PhotonFlow blocks |
| Standard layers (LayerNorm, attention, ReLU) | **Replace** with Monarch + saturable absorber + divisive power norm |

## How we use it

- The training script computes the CFM loss exactly as written above. We add the photonic noise injection inside the forward pass of `v_theta`, before the loss is computed, so the network learns to be noise robust.
- We use the simplest Gaussian source `x_0 ~ N(0, I)`. No fancy couplings.
- At sample time we use a standard fixed-step ODE solver. The number of steps is one of the things we report in our latency table.

## See also

- [[Index]]
- [[Peebles 2023 - DiT]] for an example of swapping the backbone of a diffusion / flow model. They use a transformer. We use Monarch.
- [[Dao 2022 - Monarch]] for the structured matrix that lets us run `v_theta` on a photonic chip
- [[Zhu 2026 - Optical NN for Generative Models]] for why a GAN objective is worse than CFM on noisy photonic hardware
