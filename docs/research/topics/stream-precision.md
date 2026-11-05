# Precision of the values flowing through the layers

How many bits to hold the values flowing through each layer (the activations) in. Neither the model's shape nor its weights change; it is a matter of speed. On the precision of the weights, `language-model` has a [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/training-precision.md).

Under autocast, matrix products and attention run in bf16, and the values flowing through the layers stay in fp32 by default. In [PyTorch's code](../sources/pytorch-autocast.md), `rms_norm` returns fp32 under GPU autocast, so even if the flowing values are made bf16, they should return to fp32 at every RMSNorm (not checked on the GPU). The larger the values, the more memory is read and written.

## What is known

- **Measured result:** On the H100 in [the second bundle](../experiments/second-bundle.md), making the flowing values bf16 made an update 3.7% faster (0.639 seconds and 0.615 seconds; about −0.21% converted). The checks of the output and the gradients passed too.
- **[modded-nanogpt's record 10](../sources/modded-nanogpt-records.md):** At 12 layers, width 768 and vocabulary 50,304, it made the activations bf16 and dropped autocast. An update became 6.5% shorter, the updates needed to reach 3.28 rose by 1.3% (val 3.2791 and 3.2781, so not matched), and overall it was 5.3% shorter. About −0.31% converted.
- **[nanochat](../sources/nanochat-log.md) (03-04):** Dropped autocast and made the types explicit. No speed numbers.
- **GPU utilization:** About 46% in nanochat's LOG, and about 39% for autoresearch and the first bundle, which keep autocast. Even with the same formula for it, the shapes differ, and the cause has not been looked into. This may be part of the difference (our own reading).
- **Quality:** Rounding increases. nanochat says it is "probably fine". Whether record 10's increase in updates comes from rounding or from seed-to-seed variation is unknown.
- **Gradients:** Compared with fp32 on the same weights, at width 640 and 4 layers the cosine similarity of the matrices' gradients is 0.9999 or more, and 0.98 for the x0 weights. In a small model of width 32, the gradients of the x0 and resid weights are buried in rounding (0.82, 0.56; [record of the check](../experiments/second-bundle-code-check.md)). A comparison under CPU autocast.
- **Ease of checking:** The speed shows in a short measurement with `speed.py`, and whether the output and the gradients match can be checked the same way as in `attention_check.py`. There is no need to decide by reading alone.

## Current view

In the view as of 2026-11, use it. Make the flowing values bf16 throughout, and have RMSNorm return the same type as its input. The weights and the saved format do not change, and autocast stays. Each time the shape changes, measure with and without it using `speed.py`, and use it only when it is faster and the output matches. The effect at the same cost is about 0.2%.

Dropping autocast and making the types explicit (record 10, nanochat) is a large change, so it is postponed. We have not reached record 10's 6.5% or nanochat's utilization of about 46%, so there may be room to get faster still.

## What would change the view

- It no longer being faster after the shape changes.
- Training becoming unstable. Suspect rounding. The gradients of the x0 and resid weights are the most easily affected.
- Measuring the form without autocast and finding it faster still.
