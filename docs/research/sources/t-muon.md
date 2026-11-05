# T-Muon

Li et al., T-MUON: Tensorizing Muon via t-SVD for Large Language Model Pre-training (arXiv 2610.24938, 2026-10).

It stacks several gradient matrices into a tensor, applies a transform along the stacking direction, and then orthogonalizes each slice. There are four ways of stacking: two adjacent layers, Q/K/V, heads, and all the heads of Q/K/V. The transforms are identity, DFT, DCT and DST. Stacking per head with the identity transform (Head-I) is the same as splitting Q, K and V into blocks of rows per head and orthogonalizing them separately. Per-head orthogonalization itself was studied earlier by Group Muon (arXiv 2605.08933), and the paper says its contribution is the formulation as tensors and the comparison.

## Conditions

- Only Q, K and V have their split changed. The attention output and the MLP use ordinary Muon, and the embeddings, the output and the like use AdamW.
- Context 1,024, 512 sequences per update (524,288 tokens), cosine, 10% warmup, weight decay 0.1, 5 Polar Express iterations. 1 billion tokens of FineWeb (about 1,900 updates).
- The learning rate is searched over the same grid for Muon and T-Muon (0.005–0.08 for the LLaMA type) and chosen by the mean over 3 seeds.
- In the LLaMA type, Q, K and V are separate matrices, and the baseline Muon also orthogonalizes them separately. In the GPT-2 type, Q, K and V are one matrix; the baseline Muon orthogonalizes it as a whole, while T-Muon splits it first in every form.

## Results (LLaMA type, FineWeb 1 billion tokens, 3 seeds)

| Size | AdamW | Muon | Head-I | Difference from Muon (loss) |
|---|---|---|---|---|
| 60 million (8 layers, width 512, 8 heads) | 32.17 | 26.56 ± 0.02 | 25.95 ± 0.03 | −0.71% |
| 130 million (12 layers, width 768, 12 heads) | 28.15 | 22.36 ± 0.01 | 21.92 ± 0.00 | −0.65% |
| 350 million (24 layers, width 1,024, 16 heads) | 24.03 | 19.39 ± 0.00 | 19.06 ± 0.00 | −0.58% |

- The numbers are validation perplexity. The differences are converted to loss (our own calculation).
- Stacking adjacent layers or Q/K/V with the identity transform gives the same update as ordinary Muon in the LLaMA type, and nearly the same values (26.59 and 26.57 at 60 million). In the main comparison, every transform other than identity is worse than identity.
- Taking 130 million to 2.2 billion tokens gives 19.35→19.00 (−0.62%). On FineWeb-Edu (1 seed, with the learning rate set at a small scale and carried up), −0.5 to −0.7% at 60 million–1.3 billion.
- In the GPT-2 type the difference from Muon is larger, at −1.1 to −1.3%, but the authors say part of it comes from splitting Q, K and V. Even the form that only splits them (QKV-I) gives 42.24→40.60 at 50 million.
- With long contexts (1 seed), the best form changes with the model (Head-I, Layer-F, Head-stack-I). For GPT-Base with long contexts, the DFT form is the best.
- The time per update and the scores on downstream tasks are not reported.

The HTML main text and Appendices D and E were read (the tables directly from the HTML). The proofs (Appendices B, C) were not read.
