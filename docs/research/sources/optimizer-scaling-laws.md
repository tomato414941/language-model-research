# Towards Robust Scaling Laws for Optimizers

Volkova et al. (arXiv 2602.07712, 2026-02).

- Compares AdamW, Muon, Scion, Shampoo and SOAP in two families: OLMo-shaped (ClimbMix) and Llama-shaped (FineWeb). 50 million to 1.5 billion parameters, at 30, 50, 100 and 200 tokens per parameter. Batch 512, context 1,024 (524,288 tokens per update). WSD with 5% warmup and 20% decay. The settings are tuned at 50 million, and at the other sizes the learning rate is tuned. More than 250 training runs.
- Fitting a separate Chinchilla formula for each method is unstable. Sharing the exponents and varying only the per-method coefficients ρN (the effect on model size) and ρD (the effect on tokens) more than halves the extrapolation error.

The coefficients, with AdamW as 1.

| Method | OLMo ρN | OLMo ρD | Llama ρN | Llama ρD |
|---|---|---|---|---|
| Muon | 0.96 | 2.08 | 1.02 | 1.41 |
| Scion | 0.95 | 1.99 | 0.95 | 1.73 |
| Shampoo | 0.95 | 1.55 | 0.99 | 1.16 |
| SOAP | 0.95 | 2.57 | 0.98 | 1.75 |

Read: the abstract, the introduction, the experimental setup, Tables 3 and 4, and the conclusion. The theory sections and the appendices were not read.
