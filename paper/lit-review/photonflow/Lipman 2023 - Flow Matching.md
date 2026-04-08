# Lipman 2023 - Flow Matching for Generative Modeling

**Citation:** Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow matching for generative modeling." ICLR 2023.

**Why it matters for us:** This is the paper that gave us the training objective. We use it as is. The clever part of PhotonFlow is the network architecture, not the loss. Flow matching is the loss.

## The one big idea

Diffusion models work, but their training is awkward. You have to think about variance schedules, score functions, and a noisy reverse process. Flow matching skips all of that.

Instead of learning to denoise, you learn a velocity field. Pick a starting point `x_0` from random noise, pick an endpoint `x_1` from your dataset, and draw a straight line between them. Train a neural network `v_theta(x_t, t)` to predict the direction of travel along that line at every point in time. That is the whole training signal.

At sample time you start from random noise and integrate the learned velocity field forward in time using a standard ODE solver.

## The loss

The conditional flow matching (CFM) loss in its simplest form is:

```
L(theta) = E_{t, x0, x1} [ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ]
```

where:

- `t` is sampled uniformly from [0, 1]
- `x_0` is sampled from the source distribution (Gaussian noise)
- `x_1` is sampled from the data distribution
- `x_t = (1 - t) * x_0 + t * x_1` is the linear interpolation between them
- `v_theta` is the network we are training

That is it. No noise schedule. No score matching. No KL divergence. Just a regression loss that asks the network to predict a direction.

The target `(x_1 - x_0)` is constant along each interpolated path. It does not depend on `t`. This is why the paths are "straight" and why flow matching tends to need fewer ODE steps at inference time than diffusion.

## Why we picked this over diffusion or GANs

- **vs GANs:** flow matching has a stable training objective. There is no minimax game and no mode collapse. The optical GAN of Zhu et al. that we benchmark against in the paper draft has exactly the kind of instability we want to avoid.
- **vs diffusion:** the loss is simpler and there is no schedule to tune. The straight-line paths usually mean fewer integration steps at sample time, which is a good thing when each step has to run on a photonic chip.
- **vs likelihood-based models:** flow matching gives you a generative model without a tractable density, but we do not care about densities. We care about samples and FID.

## What flow matching does NOT specify

This is the gap PhotonFlow fills. Flow matching tells you:

- what loss to minimize
- what the network's input and output shapes are
- how to sample at the end

It tells you nothing about what the network should look like inside. The original paper used a U-Net. [[Peebles 2023 - DiT]] later showed you can use a transformer. We show you can use stacked Monarch layers, which is what makes our model run on photonic hardware.

So the network architecture is a free design choice and that is exactly the choice we are exploiting.

## The CFM loss in our PhotonFlow block

In PhotonFlow the network `v_theta` is a stack of 6 to 8 PhotonFlow blocks, each containing:

1. Monarch L
2. Monarch R
3. Optical activation `sigma(x) = tanh(alpha*x)/alpha`
4. Divisive power normalization `x / (||x||_2 + eps)`
5. Time embedding added in

The CFM loss does not know or care about any of this. It just feeds in `x_t` and `t`, asks for a velocity, and compares against `x_1 - x_0`. We can add as much photonic noise as we want during the forward pass, the loss does not change.

## What we keep, what we change

| Lipman 2023 element | What we do with it |
|---|---|
| CFM loss | Keep exactly as is |
| Linear interpolation paths | Keep |
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
