# Logit softcap

Smoothly bounds the predicted values (logits) with `c * tanh(logits / c)`. Adds no weights. `language-model` also has a short [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/logit-softcap.md).

## What is known

**Quality:** There is only one comparison against none, and the difference is small. The differences are our own calculations.

| Source | Compared | Result |
|---|---|---|
| [modded-nanogpt's record 9](../sources/modded-nanogpt-records.md) | None → 30 (with QK-norm, 3,200 steps) | 3.2844 to 3.2791 (−0.16%) |
| Record 18 | 30 → 15 | Updates needed for the same loss went from 1,490 to 1,390 (about −0.4% converted) |
| [nanochat](../sources/nanochat-log.md) (24 layers) | 5 to 30 | 5 is terrible, the others about the same. About −0.14% at the best, 20 |

Record 18's author sees it as an imposed structure that helps at small scale.

**Stability:** In [the comparison of divergence](../sources/training-stability-methods.md) (830 million), it was the most stable along with QK-norm, and combining the two allows a further 1.5× learning rate. The second bundle includes QK-norm.

**Cost:** On the H100 in [the second bundle](../experiments/second-bundle.md), an update was 2.7% slower (0.599 seconds and 0.615 seconds; +0.15% converted). Record 9 (vocabulary 50,304) had 2.0%, so the guess that it would be even smaller with a smaller vocabulary was wrong.

**Start of training:** Tried in the second bundle's shape with a batch about 1/500 of the real one, the loss spike at the fourth update (9.36) seen with the x0 and resid weights did not appear with the softcap (8.44; [record of the check](../experiments/second-bundle-code-check.md)). This is not a comparison of quality.

**Adoption:** [Gemma 2](../sources/gemma-soft-capping.md) uses 30 on the last layer, and Gemma 3 replaced it with QK-norm. autoresearch and modded-nanogpt's optimization track use 15, and the optimization track keeps it even after removing its unusual parts.

**Other forms:**

- `15 * x / √(x² + 15²)` in modded-nanogpt's optimization track. Lighter than tanh, but we have not found a comparison.
- `23 * sigmoid((x + 5) / 7.5)` was slightly worse in nanochat, and may help only with FP8.
- [z-loss](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/z-loss.md): bounds the size with a penalty in the loss. IMU-1 and others use it, but comparisons of its effect have not been read.

## Current view

In the view as of 2026-11, it goes into the second bundle, as `15 * tanh(logits / 15)`, the same as autoresearch. We think its effect at the same cost is 0–0.4%. The reasons:

- Every comparison points the better way, and we have not found reports of it making things worse.
- It is a one-line change, and the measured cost is under the cutoff given below.
- A bundle is measured once, so it follows the examples that ran with Muon.

However, the evidence comes almost entirely from the modded-nanogpt lineage, and Gemma 3 in effect judged it unnecessary when QK-norm is present.

`tanh` lacks precision in bf16, so, as in nanochat and autoresearch, it is applied after casting back to fp32.

## What would change the view

- A change of shape changing how much it slows things down. If it is more than 3% slower, consider removing it. 3% is about 0.17% converted, close to the middle of the expected range. The measured 2.7% is near this.
- Results comparing with and without the softcap at the current scale. If Gemma 3's judgment holds at small scale too, it is not needed.
- A larger vocabulary. The cost grows.
- Measuring the effect of each change separately.
