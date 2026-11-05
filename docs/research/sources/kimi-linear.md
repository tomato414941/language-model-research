# Kimi Linear

Kimi Team (arXiv 2510.26692, 2025-10). It extends Gated DeltaNet with finer-grained gates per head (KDA) and mixes it with full attention at 3:1.

- Comparison at the same compute (650 million active parameters, MoE, Muon). Full attention has a training perplexity of 9.45 and a validation perplexity of 5.77. 3:1 has 9.23 and 5.65. 7:1 has 5.70 on validation, and 15:1 has 5.82.
- Scaling laws (650 million to 1.7 billion active, compute-optimal token counts): full attention 2.3092·C^−0.0536, Kimi Linear 2.2879·C^−0.0527. The difference is worth about 1.16× in compute. Full attention was tuned by grid search, and Kimi Linear used the same settings.
- 48 billion in total (3 billion active) on 1.4 trillion tokens. MMLU is 71.6 for full attention and 73.8 for Kimi Linear. MMLU-Pro is 47.2 and 51.0.
- The main advantage is at long context: the KV cache shrinks by up to 75%, and generation at a 1 million token context is up to 6× faster. At a 4,000-token context, it shows about the same speed.

The abstract, the introduction, the experiment tables, the scaling laws and the start of the efficiency section were read. The details of KDA's algorithm were not read.
