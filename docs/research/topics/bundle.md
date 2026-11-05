# The first bundle

The first set of changes, measured in a single training run. What it contains and how to remove each change are in `configs/bundle.py`, and the view on the optimizer is in optimizer.md. [The measured result](../experiments/first-bundle.md) was that, overall, validation BPB fell 18.5% and correct-answer BPB 16.5%. The reasons and estimates per item below date from 2026-11, before measuring, and the items were not measured one by one. Rewrite them when new results come in.

## Value embeddings

A per-token table is placed in every other layer, counting from the last, and added to the attention's values ([the `language-model` note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/value-embeddings.md)). The form follows nanochat's `gpt.py`. The gate's form differs from autoresearch's (2·sigmoid, 32 dimensions, weights starting from 0).

The evidence is [ResFormer](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/value-residual.md) (20.3% fewer tokens and 16.1% fewer parameters needed for the same loss; AdamW, 82 million to 470 million), [Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md) (−1.1% with the gate included) and nanochat's LOG (it won on steps, compute and time alike). ResFormer reuses the first layer's values; no paper on the form with a table itself has been found.

The tables' learning rate of 0.1 is untuned and was set lower than nanochat's value at width 768 (0.15) and autoresearch's converted to width 640 (about 0.66), because we judged the baseline model's values to be smaller in magnitude than nanochat's. The tables add almost no computation, but at 10 layers and width 640 they add 26.2 million parameters and use memory.

## 128-dimensional heads

The only evidence is one row of Marin's table (−1.0%) and nanochat's LOG: "Head dim definitely prefers to be 128 instead of 64". A single search found no paper that compares 128 and 64 directly. LEMA (arXiv 2609.25802) compares with widths smaller than 64 but was not read. It is included because it takes little effort.

## 10 layers, width 640

A comparison with the compute (6ND, about 2.7e17) held equal to the baseline's. The changes in loss follow the [Chinchilla](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/chinchilla.md) formula, and the right-hand column multiplies in the coefficient from [Volkova et al.](../sources/optimizer-scaling-laws.md) (an effect on tokens of 1.41–2.08×). It is a rough calculation that combines separate studies, and the model sizes are also below the formula's range.

| Shape | Parameters | Tokens | Per parameter | Formula | Corrected for Muon |
|---|---|---|---|---|---|
| 12 layers, width 768 (baseline) | 91 million | 500 million | 5.5 | Baseline | Baseline |
| 12 layers, width 640 | 64 million | 710 million | 11 | −1.0% | −0.4 to −0.7% |
| 10 layers, width 640 | 54 million | 840 million | 15 | −1.2% | −0.3 to −0.8% |
| 8 layers, width 640 | 45 million | 1.02 billion | 23 | −1.3% | −0.1 to −0.7% |
| 8 layers, width 512 | 29 million | 1.56 billion | 53 | −0.8% | +0.1 to +1.1% |

nanochat's LOG (2026-01-27), with a form that uses Muon and the tables, put the optimal ratio for 1e18–1e19 at about 10.5, excluding embeddings. Extended to the current compute, that gives about 55 million parameters, close to this shape. Width ÷ layers is 64, the same as the baseline, and nanochat's LOG says 64 is better than 128.

The differences are small, and the fine ordering cannot be relied on ([the problems with Approach 2](../sources/problems-with-chinchilla-approach-2.md), [an individual's check](../sources/chinchilla-check.md)). There is no result that actually compared them at the current scale: nanochat's smallest is 1e18, and [Delphi](../sources/delphi.md) starts at 3e18.

## Conditions likely to change the view

- When items are removed and measured, and some turn out to have no effect or to do harm.
- When the untuned learning rates are tuned and the results change greatly.
- When measurements of different shapes at the same compute differ from the table above.
- When the task is changed to one that also counts the cost of inference. Smaller models gain the advantage.
