# Speed and cost

Candidates that increase the tokens that can be read at the same cost. The goal is to make a better model at the same cost, so they are treated the same as model improvements. These are estimates as of 2026-11; the only thing measured is the speed measurement of [the first bundle](../experiments/first-bundle.md).

The effects are rough values from converting at about 3.9% lower loss per doubling of compute (Delphi's formula, [Marin's post](../sources/marin-moe.md)).

## How the baseline's training spends its time

The breakdown is in [the baseline's training](../experiments/baseline.md). Training takes about 68% of a rental, and most of the rest is the time from renting until the machine is usable, plus the data transfer. GPU utilization was 99–100% during training, so data loading was not a bottleneck.

Efficiency, by the same formula as autoresearch's (6N + 12·L·H·Q·T, against the H100's bf16 peak of 989.5 TFLOPs), is about 40%. autoresearch displays 39.8%, and nanochat's LOG about 46%. nanoGPT's display is based on the A100 and cannot be compared directly.

## Candidates

| Candidate | Expected effect | Notes |
|---|---|---|
| Price (the lists of [10-31](../experiments/gpu-prices.md) and [11-01](../experiments/h100-prices-2026-11-01.md)) | About 1.2% if the rented $2.41/hour becomes $1.95; 3.9% at half price | Cheap machines have old drivers. The model does not change |
| Interruptible rentals | About 6.9% at $0.71 | In `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/interruptible-gpu.md), the H100 stopped after 1 minute and did not come back. The model does not change |
| Preparation time | About 1.6% if cut to 100 seconds | Smaller when runs follow one another. Irrelevant at large scale |
| Method for computing attention (attention-kernel.md) | About 0.2% | Measured. The time per update differed by at most 1.6%; the only difference is the preparation when a shape is new |
| Overall efficiency | About 0.8% if raised to 46% | The cause of the gap has not been investigated. The first bundle is about 39%, unchanged when the attention method was switched. [The second bundle](../experiments/second-bundle.md) is about 35% (computation not counted in the formula grew). For the precision of the values flowing through the layers, see stream-precision.md |
| Rows per pass | About 0.1% at 128 rows | Measured. At width 640, 128 rows are 1.6–2.4% faster than 64, and 32 rows are 6.6% slower. 256 rows do not fit in memory even when the values flowing through the layers are bf16 |
| FP8 | 0 or a loss at the current scale | In `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/fp8.md), it is 1.17× faster at 26 layers but about 5% at the same loss, and at 12 layers it is actually slower. [Record #92](../sources/modded-nanogpt-anvil2.md) wrote its own dedicated kernels for −7% in time |
| CUDA graphs | Should be small (not measured) | −4% in time in record #92. There, each update takes about 30 milliseconds, so Python's share of the time is large. Here it is about 600 milliseconds |

In [Auto Research](../sources/auto-research.md), nanochat's quality score rose 38.7%, and the largest gain came from a change that found wasted speed and spent the time saved on more tokens. Speed can be measured without running training to the end, so it is cheaper to check than a model comparison.

## Things to consider separately

- Works at large scale too: the attention computation, efficiency, the choice of GPU.
- Works only at small scale: preparation time. It cuts research costs but is not an improvement that carries over to the frontier.
- Leaves the model unchanged: price. The training amount grows, but it is not an improvement to the model or the training method itself.

## Conditions likely to change the view

- When the model's shape or the context is changed greatly. Attention's share and the number of rows that fit in memory change.
- When prices change greatly, or when the GPUs available to rent change.
