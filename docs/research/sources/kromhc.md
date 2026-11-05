# KromHC

KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices (arXiv 2601.21579). A reduced-parameter form of mHC, which splits the residual stream into several streams and mixes them.

It replaces nanochat's residual and compares the forms (Muon for matrices; AdamW for the residual streams, the embedding and the output). The original nanochat form has weights on x0 and resid, and the mHC forms replace them (Appendix G). FineWeb-Edu, about 20 tokens per parameter, 4 streams.

| Form | Added parameters (6 layers) | Validation BPB, 6 layers (about 60 million) | Validation BPB, 12 layers (about 186 million) |
|---|---|---|---|
| Original residual | | 1.047 | 0.864 |
| mHC | 460,000 | 1.042 | 0.861 |
| mHC-lite | 610,000 | 1.045 | 0.864 |
| KromHC | 240,000 | 1.047 | 0.862 |

The CORE scores are 6.477, 7.971, 8.208 and 9.018 (6 layers) and 14.774, 16.023, 13.217 and 16.872 (12 layers), with larger differences than BPB. The comparison of compute and time was not read.

Read: the abstract, the results in Section 4, and the beginnings of Appendices G and H.
