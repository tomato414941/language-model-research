# Benchmarking Optimizers

Semenov et al. (EPFL and others), Benchmarking Optimizers for Large Language Model Pretraining (arXiv 2509.01440, 2025-09).

- A Llama architecture (weight tying, SwiGLU, RMSNorm, RoPE) on FineWeb; each method is tuned before the comparison.
- 124 million parameters, 16,000 or 131,000 tokens per update, trained on 1 billion to 16.8 billion tokens.
- At 16,000 tokens per update and 1 billion tokens, SOAP, ADOPT, AdEMAMix, D-Muon, Prodigy and SF-AdamW beat AdamW, and D-Muon is the best. With more tokens, AdEMAMix takes the lead and AdamW closes the gap.
- At 720 million parameters and 1 million tokens per update, AdEMAMix and MARS are the best.
- Overall, AdEMAMix is consistently good. Signum, Lion and MARS improve with large batches. Sophia diverges when trained long with small batches. SOAP is stable at both batch sizes.

The abstract, the figure captions and part of the results were read. The results at 210 million and the appendix were not read.
