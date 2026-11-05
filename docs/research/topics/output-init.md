# Initialization of the output weights

The initial values of the last weights of attention and the MLP (`c_proj`). The values coming out of them are added to the residual. `lm_head`'s initialization is a separate matter, considered together with separating the embeddings.

- **Small values:** The baseline and the first bundle. 0.02/√(2×number of layers), as in GPT-2; 0.0045 for 10 layers.
- **0:** At the start of training, each layer passes its input through unchanged. [nanochat](../sources/nanochat-log.md), [autoresearch](../sources/autoresearch.md) and [modded-nanogpt](../sources/modded-nanogpt-records.md) all use this form, and all run with Muon. They are code of the same lineage, so not independent confirmations, and none of them measured its effect alone. Adoption in large models has not been checked. `language-model` also has a short [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/zero-output-init.md).
- **0.02 without scaling down:** The default in HF's Llama.

## What is known

We have not found a direct comparison of the current small values and 0. Close comparisons follow. The loss differences are our own calculations.

| Source | Conditions | Compared | Result |
|---|---|---|---|
| [Multi-Head Low-Rank Attention](../sources/multi-head-low-rank-attention.md) | 2.9 billion, AdamW, 98.3 billion tokens | 0.02 to 0 | −0.63%. Better in 8 of 9 attention forms (+0.06% to −1.30%) |
| [Optimal Scaling](../sources/optimal-scaling-norm.md) | Width 256, 2–64 layers, Scion, 270 million tokens | No scaling down to 1/√(2×number of layers) | −0.24% to −0.55%. The authors call it negligible |
| [Physics of LM 4.1](../sources/physics-of-language-models-4-1.md) | Llama and others | 1/√N to no scaling down | No scaling down is "slightly better". No numbers |
| [μ-Transfer](../sources/mu-transfer.md) | Up to 1.2 billion, Adam | Q's weights starting from 0 | −0.04% to −0.14% |
| [ReZero](../sources/rezero.md) | 12 layers, character by character, no LayerNorm | The multiplier on the whole layer from 1 to 0 | Steps to reach BPB 1.2 about 0.6×. Final BPB the same |
| [LayerScale](../sources/layerscale.md) | Images, 12–36 layers | The multiplier on the whole layer from 0 to small values | Small values are better. Accuracy 78.3 to 80.4 at 12 layers |

[SkipInit](../sources/skipinit.md) finds 1/√depth and 0 about the same, in image CNNs. [Transformers Are Born Biased](../sources/born-biased.md) says random initial values create a bias in token preferences that weakly persists after training. With 0 this bias should not appear at the start, but that is our own reading, and the effect on quality is unknown.

**The difference from the current values should be small:** The −0.63% was against values about 4.5× larger than ours. Muon's update size is 0.04 per update regardless of the gradient, and it catches up with the initial size of the MLP's output weights (0.11–0.34 in singular values) in 3–9 updates (our own calculation).

**Cost:** The compute does not change. No weights are added.

**Does it work with Muon:** At the first update, the gradient of the weights just before the output is 0. The current Muon then just makes a zero update, and starts moving from the next update (checked by reading `muon.py`, and by a test).

**The first loss:** Starting from 0 with the embeddings still shared makes the first loss high, because the input reaches the output unchanged and strongly predicts the same token. Tried in the second bundle's shape, the first loss was 12.8 against 9.1 with the current values, fell to 8.2 at the second update, and over the next 30 updates showed no difference from the current values ([record of the check](../experiments/second-bundle-code-check.md)). [Record 5](../sources/modded-nanogpt-records.md) (width 768) also falls from 16.0 to 9.5 at the second update. This is a check of the start, not a comparison of quality.

## Current view

In the view as of 2026-11, the second bundle uses 0. We think its effect at the same cost is small, 0–0.3%. The reasons:

- It costs nothing, is a one-line change, and adds no weights.
- A bundle is measured once, so it follows the examples that ran with Muon. autoresearch's set of learning rates also runs in this form.
- We have not found reports of 0 making a language model worse.

The other forms were not chosen. Multipliers on the whole layer (ReZero, LayerScale) have no comparison close to our shape, and are considered separately under the related idea of the x0 and resid weights. CompleteP was worse than no multiplier in Optimal Scaling's comparison (+1.5–2.4%).

## What would change the view

- A direct comparison of the current values and 0.
- Measuring the effect of each change separately.
- Changing the learning rates a lot. Moving away from the set that ran with 0 weakens the reason to follow the examples.
