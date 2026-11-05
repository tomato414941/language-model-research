# The model's shape

[The first bundle](../experiments/first-bundle.md) brought the training method up to current methods, but much of the model's shape is still nanoGPT's. [nanochat](../sources/nanochat-log.md), [autoresearch](../sources/autoresearch.md) and [modded-nanogpt](../sources/modded-nanogpt-records.md) all change the shape. The three are code of the same lineage and inherit methods from one another, so even when all three share a shape, that is not three independent confirmations. "Current view" is provisional, based on what was read as of 2026-11, and not measured ourselves.

| Part | First bundle | autoresearch | Reported effect |
|---|---|---|---|
| Position | Learned table | RoPE | rope.md |
| Normalization | LayerNorm (learned weights) | RMSNorm (no weights) | normalization.md |
| Normalizing Q and K | No | Yes | qk-norm.md |
| Activation function | GELU | ReLU² | activation.md |
| Output initialization | Small values | 0 | output-init.md |
| Embeddings | Shared between input and output | Separate, large learning rate | embeddings.md |
| x0 and resid weights | No | Yes | residual-weights.md |
| Logit softcap | No | 15 | logit-softcap.md |
| Precision of the values flowing through the layers | fp32 | Only the embedding table in bf16 | stream-precision.md |
| How data is read | Cut out from the middle | Packed from the start of documents | document-rows.md |
| Attention window | Full | Half in 3 of 4 layers | Estimated to cut compute by about 7%. attention-kernel.md |
| Context | 1,024 | 2,048 | An update is 7.5% slower. Validation BPB cannot be compared directly |

Record 5 is worth 1.47× in time, including 128-dimensional heads (put into the bundle). Record 9 is worth 1.32×, including value residual and momentum warmup (put into the bundle).

autoresearch's standard reaches BPB 0.998 with 300 seconds of training, and the first bundle 1.024 with 810 seconds. But the amount of validation data, the context and how rows are cut differ, so they cannot be compared as is. GPU utilization is about 39% for both autoresearch and the first bundle, and the compute per token is about 240 million and 410 million (our own calculation).

## Current view

The second bundle centered on bringing the shape up to the current standard. [The second bundle](../experiments/second-bundle.md)'s result against the first bundle: correct-answer BPB −4.4%, and validation BPB read from the start of documents −4.2% (validation BPB as currently measured is +2.2%; the reason is in document-rows.md). The effect of each change was not measured separately.

- There are records of adding them one at a time, at a similar scale, starting from a model close to nanoGPT; this is the evidence closest to our conditions.
- Most add no compute or reduce it.
- All are methods that do not depend on a particular model or task. But only RoPE, RMSNorm (the form with weights) and separate embeddings can be called standard at large scale too; RMSNorm without weights, ReLU² and zero-initialized outputs come mainly from the three lineages above. Whether they also help at large scale has not been checked.
- autoresearch runs with this combination of shape and learning rates.

A rough estimate is that the ten changes down to how data is read lower BPB by 2–6% (from each note's expectations). It comes from the conversion of 2× compute to −3.9% and from numbers under different conditions, and part of the records' effects is already included in the first bundle. Nor do the effects add up. It is only a guide for setting the order.

The window and the context are postponed, since they also change how things are measured.

## What would change the view

- Measurements far off the estimate.
- The learning rates no longer fitting after the shape changes. nanochat says values tuned at one scale get worse at another.
- Separate embeddings becoming a disadvantage through table size when the vocabulary grows.
- Changing the task so that inference cost counts too.
