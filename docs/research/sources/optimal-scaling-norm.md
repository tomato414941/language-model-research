# Optimal Scaling Needs Optimal Norm

Optimal Scaling Needs Optimal Norm (arXiv 2510.03871). A paper that studied, with Scion (an optimizer that sets the size of an update by the matrix's norm, close to Muon), how the best norm does not change as the scale of the model or the data changes.

Appendix A.10 has a comparison of the initial size of the output weights of attention and SwiGLU: whether or not the semi-orthogonal initialization is multiplied by 1/√(2×number of layers). Width 256, RMSNorm, RoPE, separate input and output embeddings. 2,048 updates of 32 sequences × 4,096 tokens (270 million tokens).

The final loss at the best learning rate. The differences are our own calculation.

| Layers | Not multiplied | 1/√(2×number of layers) | Difference |
|---|---|---|---|
| 2 | 4.10 | 4.09 | −0.24% |
| 16 | 3.74 | 3.73 | −0.27% |
| 64 | 3.62 | 3.60 | −0.55% |

- The authors call the difference negligible. The numbers have only two decimal places, so there is rounding of about ±0.3%.
- The CompleteP form, which multiplies the residuals by a factor, is worse than not multiplying (2 layers 4.15, 16 layers 3.82, 64 layers 3.67).
- It does not compare 0.

Appendix A.10 and the part on the model's conditions were read. Of the main text, only the introduction and the key points were read.
