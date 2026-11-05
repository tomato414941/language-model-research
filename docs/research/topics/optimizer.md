# Optimizer

What to update the weights with. "Current view" is provisional, based on what was read as of 2026-11. Our only measurement is one run of the whole bundle ([the first bundle](../experiments/first-bundle.md)), and Muon's own effect was not measured. Rewrite it when new results come in.

## What reading showed

- **AdamW:** the experiments in [the paper](../sources/adamw.md) are image classification, not comparisons with other methods on language models.
- **How Muon relates to AdamW:** it is used alongside AdamW, not as a replacement for all of it. [Muon](../sources/muon.md) is used for the hidden layers' matrices, and the embeddings, the output, normalization and scalars are updated with AdamW. [Kimi](../sources/muon-is-scalable.md) and DeepSeek-V4 (arXiv 2606.19348) split them the same way; for the baseline's 12 layers and width 768, about 92% of the parameters fall on Muon's side.
- **Size of the difference:** the reports are split.

| Source | Conditions | Muon's difference (vs AdamW) |
|---|---|---|
| [The original post](../sources/muon.md) | Width 768, 524,288 tokens per update, the FineWeb speed record | 1.35× |
| [Fantastic](../sources/fantastic-pretraining-optimizers.md) | 130 million, 1× the training amount | About 1.4× |
| Same as above | 1.2 billion | About 1.1× |
| [Volkova et al.](../sources/optimizer-scaling-laws.md) | ClimbMix, 50 million to 1.5 billion, 1.5–10× the training amount | An effect on tokens of 2.08× |
| Same as above | FineWeb, the same range | An effect on tokens of 1.41× |
| [Kimi](../sources/muon-is-scalable.md) | 399 million to 1.5 billion, the optimal number of tokens | 52% of the compute (about 2×) |
| [Essential AI](../sources/practical-efficiency-of-muon.md) | 100 million to 4 billion | 10–15% fewer tokens |
| [The Polar Express paper](../sources/polar-express.md) | 124 million, 1 billion tokens, 32,768 tokens per update, no weight decay | Loss 4.197 with AdamW, 3.639 with Muon (−13%) |

- **Improvements to Muon:** [Polar Express](../sources/polar-express.md), with no change in cost, lowered loss by 1.4–1.7% at 1 billion tokens and 0.3% at 10 billion tokens. NorMuon ([note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/normuon.md)) needs 6% fewer updates than Muon for the same loss at 124 million and 15% fewer at 350 million. At 1.1 billion it needs 21.74% fewer than AdamW, against 10.43% for Muon. Its memory is about half of Adam's, and its update time is 2.9% longer than AdamW's (at 5.4 billion, arXiv 2510.05491).
- **Cautious weight decay:** at 338 million with 20 tokens per parameter, AdamW went from 3.0136 to 3.0059 (−0.26%); at 1 billion with 100 tokens, AdamW went from 2.65 to 2.56 and Muon from 2.51 to 2.42 (about −3.5%; arXiv 2510.12402, `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/cautious-weight-decay.md)). At [IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md)'s 70 million it is about −1%, with an inconsistency in the baseline's values. We think the effect is on the small side when the training amount is small.
- **Whether to split Q, K and V:** [the original post](../sources/muon.md) says splitting is better, while NorMuon's authors say splitting brought no improvement, so the reports are split. On splitting per head, [T-Muon](../sources/t-muon.md) says that with plain Muon it gives 0.6–0.7% lower loss at 60 million to 350 million. In NorMuon's [speedrun](../sources/modded-nanogpt-paired-head-muon.md), the gain from treating K and V two heads at a time is about 0.04%, and in [Group Muon](../sources/group-muon.md) splitting fully per head is slightly worse at the end.
- **Improvements from 2026:** [Muown](../sources/muown.md) is about 1% better than plain Muon at 1 billion and above, but against NorMuon it is about 0.1% at 124 million. [FARO](../sources/faro.md) is about 0.4% at 800 steps. As far as was read, none of them is compared directly with the combination of NorMuon and cautious weight decay now in use, or else the difference is small. In [record #92](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/anvil.md), even with the update rule reverted the difference is 4.9 millinats, and most of the difference comes from how the final weights are made. In [the reports of what did not work](../sources/modded-nanogpt-wr-blog.md), optimizers with slow state, such as AdEMAMix, lose below 1,500 steps. The current 1,281 updates are also on the short side.
- **Powers of the gradient:** [GradPower](../sources/gradpower.md) raises the gradient to the power p element-wise before passing it to the optimizer. The paper's p=1.2 is a result at 131,072 tokens per update; in [Parameter Golf](../sources/parameter-golf-gradpower.md), with the same 524,288 tokens as now and Muon, it made things worse, and 0.9 was slightly better.

- **Other methods:** the matrix-based SOAP and Kron match or beat Muon as the training amount grows ([Fantastic](../sources/fantastic-pretraining-optimizers.md), [Volkova et al.](../sources/optimizer-scaling-laws.md), [NVIDIA](../sources/soap-muon-and-beyond.md)). [Semenov et al.](../sources/benchmarking-optimizers.md) rank AdEMAMix and MARS best for long training, which conflicts with Fantastic. The cause is unsettled. In [AlgoPerf](../sources/algoperf.md)'s track that allows tuning, the matrix-based Distributed Shampoo won. In [modded-nanogpt's optimization track](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/modded-nanogpt-optimization-track.md), SOAP-Muon takes fewer steps than Muon. However, that comparison does not count time.
- **Learning rate:** Fantastic says AdamW with a peak learning rate of 6e-4 is a weak baseline. nanoGPT uses the same 6e-4. In a tuned example at 130 million with the same update size, it was 0.008. The model shapes differ, so it does not necessarily carry over as is.
- **Cost:** at width 768 and 524,288 tokens per update, Muon's extra computation is at most 0.7% (the original post's calculation). Its memory is less than AdamW's.
- **Weight decay size:** [nanochat](../sources/nanochat-log.md)'s best values are about 0.40 at width 512 and about 0.22 at width 768, and it says the value is inversely proportional to the square of the width. At width 640 that is about 0.3, so the current 0.2 is on the small side. However, nanochat trains longer than we do, and autoresearch uses 0.2 at width 512.
- **Limits of fine adjustment:** nanochat says values tuned at one scale do worse at another. Measuring also costs $0.55 per run, so it cannot be decided by reading alone.

## Current view

As the first thing to try on the baseline (91 million parameters, 500 million tokens, 524,288 tokens per update, a training amount of about 0.27×), the first bundle used Muon and AdamW together.

The settings started from autoresearch's values, but differ in the following points.

- Embedding and output learning rates: [autoresearch](../sources/autoresearch.md) uses 0.6 and 0.004 for separate tables. The first bundle shares one table, so these cannot be used as is. It used the output's 0.004, and the same value for the position encoding and the normalization weights (the view on separation is in embeddings.md). This is close to the 0.0036 of [the first Muon record](../sources/modded-nanogpt-muon-record.md).
- Matrix learning rate 0.04: the number and size of updates are the same as autoresearch's, but the width differs (512 and 768), and it is untuned. nanochat's default is 0.02. The first Muon record has the same shape with 6.5× longer training, and its update magnitude is about a quarter of the current value.
- Polar Express, NorMuon, cautious weight decay: autoresearch's values include them, so all are used at first. Each can be turned off in the settings, so they can be checked one by one if the results are bad.
- Q, K, V: the model keeps them as one matrix unchanged, and the optimizer splits it into three to orthogonalize. The baseline's weights can be loaded as is.

The reasons for this view are these.

- At 1×, the smallest training amount compared, Muon came first. However, 0.27× is outside the compared range, so this is extrapolation.
- Its cost in computation and memory is small.
- autoresearch runs it with the same 953 updates. Other changes are included too, so Muon's own effect is unknown.

A rough estimate of the effect: up to 2× from tuning the learning rate and a further 1.4× from Muon, up to 2.8× in all. Converted at about 3.9% lower loss per doubling of compute (Delphi's formula `loss = 1.65 + 117.8·C^−0.0966` in [Open Athena's post](https://www.openathena.ai/blog/pretraining-speedup/)), that is about 6%. It multiplies figures from different conditions and is only a guide for deciding the order. The Polar Express paper's comparison showed a 13–15% drop, so with a small training amount the effect may be larger than estimated. However, its update size is about 1/16 of ours, and it has no weight decay. The first bundle fell 18.5% overall, closer to the paper's comparison than to this rough estimate. However, that includes changes other than Muon and the share from the baseline AdamW's weak settings.

## Conditions likely to change the view

- When our measurements differ greatly from the estimates. Especially if the Muon bundle cannot beat a tuned AdamW.
- When the training amount is increased. SOAP or Kron may become better.
- When the number of tokens per update is cut greatly. Revisit Semenov et al.'s AdEMAMix and MARS.
- When results come out comparing SOAP-Muon and others at the current scale, time included.
- When the model is made larger. In Fantastic the difference fades, and in NVIDIA's work a difference appears with larger batches.

## Related

- `language-model`'s notes: [NorMuon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/normuon.md), [Muon+](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muon-plus.md), [MuonEq](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muoneq.md), [SOAP-Muon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/soap-muon.md), [IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md).
