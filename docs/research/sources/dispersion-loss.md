# Dispersion Loss

Liu et al., Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models (arXiv 2602.00217, ICML 2026).

## Condensation

- The cosine similarities between the vectors of tokens in the same sentence gather near 1 (Definition 1.1). They are measured on the vectors after each layer, over every pair in the sentence (§2.2).
- It is strong in small models such as GPT-2 and Qwen3-0.6B, and less likely to occur in large models such as GPT2-xl and Qwen3-32B. In experiments that vary the size in GPT-2's shape as well, the larger the model, the less the vectors gather (§3.1.2). It does not show at what size it disappears.
- It says that distillation from a larger model does not fix it.

## Method

- The loss `log Σ_{i≠j} exp(−arccos(cos(z_i, z_j)) / (π τ))` is computed on each sentence's token vectors, aggregated over the layers, and added to the next-token loss with a weight of 0.1 (τ=1, §3.4, §4.2). Applying it to later layers works better than to earlier ones (§4.2).
- The computation is `O(N²F)` per batch (N is the sentence length, F the width). Tokens may be subsampled (§3.4).

## Results

| Model | Standard | With the dispersion loss |
|---|---|---|
| GPT-2 | 34.95 ± 0.11 | 35.61 ± 0.12 |
| GPT-2-m | 38.55 ± 0.18 | 39.35 ± 0.15 |
| Qwen3-0.6B | 45.93 ± 0.37 | 47.45 ± 0.16 |
| Qwen3-1.7B | 54.57 ± 0.17 | 55.68 ± 0.21 |

- The above are the results of trained models reading 200 million more tokens (wikitext-103, 3 seeds), averaged over 10 tasks. The time is 1.03–1.05× (Table 2, A100 time).
- Training from scratch was Qwen3-0.6B on 156 billion tokens of C4 (640 GPUs): 34.84→36.01 (PIQA +4.0, TruthfulQA +7.4, Table 5). It does not show the variability.
- It does not report validation loss or perplexity.

About the first 80% of the HTML main text was read, through summaries and quotations. The appendix's settings (Table S1) and the optimizer were not read.
