# PhotonFlow Lit Review

This vault holds the focused literature notes for the PhotonFlow project. The aim is not to summarize each paper completely. The aim is to capture, in plain English, the one or two ideas from each paper that we actually use, and why.

## The four papers we build on

PhotonFlow sits on the intersection of two worlds. From the **photonics side** we take the idea that an MZI mesh can do matrix multiplication at the speed of light. From the **generative modeling side** we take flow matching, because it has stable training and a simple loss. The middle piece is Monarch matrices, which let us write a useful linear layer in a way that an MZI mesh can actually run.

Each note below covers one paper and ends with a short "How we use it" section.

- [[Shen 2017 - Coherent Nanophotonic Circuits]]  
  The first paper to actually run a neural network on a silicon photonic chip. Shows that an MZI mesh can implement a matrix-vector multiply, and uses a saturable absorber as the nonlinearity. This is the hardware foundation for everything we do.

- [[Lipman 2023 - Flow Matching]]  
  Introduces conditional flow matching. We keep the loss as is and only redesign the network that learns the vector field.

- [[Peebles 2023 - DiT]]  
  Shows that a transformer can replace the U-Net in diffusion. This is the model we are NOT allowed to use, because it relies on softmax attention. It tells us what we are competing against.

- [[Dao 2022 - Monarch]]  
  Defines Monarch matrices, a class of structured linear layers that are both expressive and hardware friendly. We use these to replace attention. The key insight is that a Monarch layer is the same kind of computation graph that an MZI mesh runs.

## How they fit together

```
Lipman 2023 (flow matching loss)
        |
        v
   PhotonFlow vector field network
   /        |          \
  /         |           \
Dao 2022    saturable    divisive
Monarch     absorber     power norm
  |         (Shen 2017)  (microring)
  v
MZI mesh array
(Shen 2017)
```

We keep Lipman's training objective, drop the attention from Peebles, replace it with Dao's Monarch layers, and run the whole thing on the kind of MZI hardware Shen built. Peebles is in the vault as the baseline we compare against, not as something we extend.

## What is intentionally NOT here

- Detailed math derivations. Read the original papers for those.
- Anything about diffusion sampling tricks (DDIM, classifier-free guidance, etc.) that we are not using.
- Long related-work sections from each paper. Only the parts that touch our design.

## Open questions to come back to

- Does the 8 to 10 percent FID gap close if we train longer? The paper draft does not commit on this yet.
- Is divisive power normalization stable enough at 4-bit precision? The Dao paper does not test this case.
- What is the actual fJ-per-MAC number on a real chip vs the estimate we get from `torchonn`? Need exp6 to answer.
