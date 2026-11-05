# Importance of Embedding Layer Learning Rate

Quantifying Hyperparameter Transfer and the Importance of Embedding Layer Learning Rate (arXiv 2605.21486). On top of nanoGPT, it studies whether the learning rate carries over to other widths, changing the differences between the standard parametrization (SP) and μP one at a time.

12 layers, context 1,024, vocabulary 50,304, widths 128 to 2,048, FineWeb-Edu, AdamW. 10 billion tokens, or 20 tokens per parameter. About 160,000 H100 hours.

- Of the 16 combinations of the four differences between SP and μP, the one that matters is the embedding learning rate. Setting only SP's embedding learning rate to μP's brings it level with μP, and lowering only μP's embedding learning rate makes it worse.
- The effect of `lm_head`'s initial scale is negligible. The learning rate of the LayerNorm weights can also be small.
- A low embedding learning rate not only slows training but also makes it unstable near the best learning rate.
- It matters most early in training. Lowering the embedding learning rate partway through under μP leaves a larger gap the earlier it is lowered, ending with a loss gap of about 0.1–0.2.
- With shared embeddings, theory requires multiplying `lm_head`'s output by a factor of 1/width (Appendix H.15).
- The results are for AdamW only, and extending them to Muon is left to future work.

The abstract, the introduction, Sections 4 and 5, part of Section 8, Appendix A and the start of H.15 were read. The numbers in the figures were not read.
