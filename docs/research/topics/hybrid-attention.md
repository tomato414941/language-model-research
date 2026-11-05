# Hybrids with linear attention

Replacing the attention in some layers with linear attention such as Gated DeltaNet or Kimi Delta Attention. The view as of 2026-11 is that the effect is about 1–2%, and it was not put in the first bundle because the implementation is heavy. This is not a measured result.

- **Effect:** [Kimi Linear](../sources/kimi-linear.md) has about 1.2% lower validation loss at the same compute (our conversion). By the scaling law, that is worth 1.16× the compute. [Olmo Hybrid](../sources/olmo-hybrid.md) is worth 1.3× at 1B, more at larger sizes. In a comparison at a nominal 100 million, the loss is 4–6% lower, but the parameter counts, about 103 million and 130 million, are not matched. Allowing for that, we think it is 1–3%.
- **Speed:** the main advantage is inference at long contexts. In [the Gated DeltaNet paper](../sources/gated-deltanet.md), ordinary attention is faster up to a context of 2,000 tokens. Olmo Hybrid adjusts its shape to match speeds. We do not think it becomes cheaper at the current context (1,024).
- **Why keep attention:** In [Zoology](https://arxiv.org/abs/2312.04927) (abstract), the latest attention-free convolutional models are up to 2.1 worse than attention in Pile perplexity, and 82% of the gap was explained by the ability to recall information that appeared earlier. On recall tasks, a 70-million attention model beats a 1.4-billion convolutional one.
- **Adoption:** used at the frontier, in Qwen3-Next, Qwen 3.5 and 3.6, Kimi K3 and others.

## Conditions likely to change the view

- When the context is lengthened greatly, or when the cost of inference is also counted.
- When a comparison at the current scale that includes wall-clock time comes out.
