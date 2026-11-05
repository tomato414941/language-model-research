# GLU Variants

Shazeer, GLU Variants Improve Transformer (arXiv 2002.05202). Where SwiGLU comes from.

- The output of one matrix passed through the activation is multiplied elementwise by the output of another matrix. To keep the parameters and compute equal, the inner width is cut from 4× to 8/3×.
- T5's fill-in-the-blank task; held-out log-perplexity of pretraining on C4 (lower is better).

| MLP | 65,536 updates | 524,288 updates |
|---|---|---|
| ReLU | 1.997 | 1.677 |
| GELU | 1.983 | 1.679 |
| Swish | 1.994 | 1.683 |
| GLU | 1.982 | 1.663 |
| Bilinear | 1.960 | 1.648 |
| GEGLU | 1.942 | 1.633 |
| SwiGLU | 1.944 | 1.636 |
| ReGLU | 1.953 | 1.645 |

GELU to SwiGLU: −2.6% (our own calculation). ReLU² is not compared. The model is not an autoregressive language model, and the paper says the downstream task results vary widely.

The main text and Table 1 were read.
