# Activation functions

The computation inside the MLP that bends the widened values. The baseline and the first bundle both use GELU. The candidates are:

- **ReLU²:** squares positive values and sets negative values to 0. Adds no weights.
- **SwiGLU:** adds one matrix and multiplies by a gate. The standard in LLaMA and others.
- **xIELU, xIPReLU:** give ReLU²'s negative side a gradient too, with two values learned per layer.

## What is known

The loss differences are our own calculations, comparing within the same table. [Primer](../sources/primer.md) compared the log of perplexity.

**GELU to ReLU²:**

| Source | Scale | Loss difference |
|---|---|---|
| Primer | 110 million, C4 | About −0.7% (read from a figure) |
| Primer | About 35 million, LM1B | In the better direction. Large in T5, very small in T2T (no numbers) |
| [The xIELU paper](../sources/xielu.md) | 1.1 billion, 4 billion tokens | −0.84% (against SiLU, which is close to GELU) |
| [Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md) | 150 million to 280 million | 0 (added after QK-norm) |
| [Record 5's code](../sources/modded-nanogpt-records.md) | About 120 million (12 layers, width 768) | A comment says "about 1–2%". How it was measured is not written |

**Results are split between ReLU² and SwiGLU:**

| Source | Scale | Result |
|---|---|---|
| [nanochat's LOG](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/swiglu.md) | d12, d24 | ReLU² wins on steps, time and compute alike |
| Primer | 110 million | ReLU² wins by about 0.2% (figure) |
| The xIELU paper | 1.1 billion, 125 billion tokens | ReLU² wins by 0.68% |
| [ReLU² Wins](../sources/relu2-wins.md) | 1.3 billion, 100 billion tokens | About the same. SwiGLU is slightly ahead partway through |
| The xIELU paper | 1.1 billion, 4 billion tokens | SwiGLU wins by 1.04% |

Many reports find SwiGLU better than GELU (−2.6% in [GLU Variants](../sources/glu-variants.md), −1.37% in [BTLM](../sources/btlm.md), slightly better in [BLOOM's comparison](../sources/bloom-ablations.md)).

**There are reports that xIELU is better than ReLU²:** −0.60% (125 billion tokens) and −1.16% (4 billion tokens) in the xIELU paper. xIPReLU: −0.43% and −1.16%. [Apertus](../sources/apertus.md) adopted it for 70B, and at 1.5 billion it gave −2.0% from SwiGLU. All these results are at 1.1 billion or more, with AdamW-type optimizers, and involve xIELU's authors. We have not found results at the current scale or with Muon. LeakyReLU(0.5)², a close idea that also gives the negative side a gradient, was slightly better per step in [nanochat](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/leaky-relu-squared.md) but slower, and lost on time.

**Speed:** At the xIELU paper's 1.1 billion, ReLU² took 534 milliseconds against GELU's 544 milliseconds (1.8% faster). xIELU took 560 milliseconds, 4.9% slower than ReLU² (unoptimized). xIPReLU took 542 milliseconds, 1.5% slower. By the current conversion, these come to a gain of about 0.1%, a loss of about 0.27% and a loss of about 0.08% respectively. Measured on the H100 in [the second bundle](../experiments/second-bundle.md), ReLU² was 2.6% faster per update than GELU (0.615 seconds and 0.632 seconds; about a 0.15% gain converted).

Under PyTorch's automatic 16-bit computation (autocast), only the squaring should run in 32 bits on the GPU (`square` calls `pow`, and `pow` is in the 32-bit list in `autocast_mode.h`; `F.rms_norm` is in the same list). The code of nanochat, autoresearch and modded-nanogpt is written the same way, and ReLU² was faster in the GPU measurement too, so it has not been a problem.

**Initialization:** ReLU²'s output changes with the square of its input's size, so it depends on the size of the MLP's input weights. With inputs whose mean square is 1, the standard deviation of the input weights' output is 0.4 in nanochat, about 0.58 in modded-nanogpt and 1.0 in autoresearch; the first bundle's initialization gives about 0.51 at width 640 (all our own calculations).

**Other uses:** ReLU² is easy to make sparse at inference (ReLU² Wins). [Stochastic activations](../sources/stochastic-activations.md) is also about making inference faster, and is not a candidate for the current BPB.

## Current view

In the view as of 2026-11, the second bundle uses ReLU². We think its effect at the same cost is 0–1%. The reasons:

- Comparisons with GELU include some showing 0, but none report it worse. It is slightly faster.
- Only ReLU² has examples run with Muon at a similar scale (nanochat, autoresearch; code from the same lineage, not independent confirmations). A bundle is measured once, so harm from one change is hard to notice.
- It is a one-line change and adds no weights.

The input weights' initialization stays unchanged, since it is within the range of the other projects.

xIELU stays a candidate. Its expected effect is about 0.3–1% better than ReLU², but there are no results at the current scale.

## What would change the view

- Results comparing xIELU or xIPReLU with ReLU² at the current scale.
- Training becoming unstable. ReLU²'s gradient grows in proportion to the value. Revisit the input weights' initialization too.
- Measuring the effect of each change separately.
- Making it a task that also counts inference cost. ReLU² might gain further.
