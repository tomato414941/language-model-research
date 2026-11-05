# MoE

Splits the MLP into several experts and runs only some of them for each token. `language-model` also has a [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/moe.md). In the view as of 2026-11, its effect at the current scale is unclear and the implementation is heavy, so its priority is lowered. This is not a measured result.

- **Reports of large effects:** [Marin](../sources/marin-moe.md): worth about 4× the compute at 1e18 to 3e20 (6.7× at 1e23, 3.6× in wall-clock time). In [the Engram paper](../sources/engram.md)'s comparison, validation loss is 7.6% lower at the same compute. [DeepSeekMoE](../sources/deepseekmoe.md): with models of the same 200 million active parameters, The Pile loss is 6–9% lower.
- **Difference in the amount of training:** DeepSeekMoE's comparison is at 500 tokens per parameter. The current baseline is at 5–15 tokens, on the side short of tokens. We think MoE, which adds capacity, is unlikely to help in the same way (our own interpretation).
- **Wall-clock time at small scale:** In nanochat's LOG (2026-02-19), it got better per step, but efficiency fell from about 46% to about 35% and it lost on time.
- **Adoption:** It is standard at the frontier, and its priority rises if we aim for large models.

## What would change the view

- A larger budget that greatly increases the tokens per parameter.
- Finding an implementation that does not lose speed.
- Putting more weight on improvements that also work at large scale. The current order centers on the effect at the current scale.
