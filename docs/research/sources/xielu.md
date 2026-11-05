# xIELU

Huang and Schlag, Deriving Activation Functions Using Integration (arXiv 2411.13010).

- Builds activation functions from the shape of their gradient. On the positive side the gradient grows linearly, as with ReLU², and on the negative side the gradient can be negative.
- xIELU is `αp x² + 0.5x` on the positive side and `αn (e^x − 1) − αn x + 0.5x` on the negative side. αp and αn are values learned per layer, with softplus setting their range. xIPReLU is a lighter form that also uses `αn x² + 0.5x` on the negative side.
- It argues that they are better than ReLU² and GELU in terms of the gradient, but says its explanation of why is a hypothesis and that a more rigorous framework is needed.

Llama-shaped, with RMSNorm, RoPE, AdamW and FineWeb Edu. The weights and compute are matched by making the inner width of the non-GLU MLPs 1.5×. The loss differences are our own calculation.

**Main comparison (context 4,096, 125 billion tokens, about 114 tokens per parameter):**

| Activation function | Validation loss at 1.1 billion | Validation loss at 3 billion |
|---|---|---|
| SwiGLU | 2.353 | 2.273 |
| ReLU² | 2.337 | |
| xIPReLU | 2.327 | |
| xIELU | 2.323 | 2.258 |

The error is about ±0.004. From ReLU², −0.43% with xIPReLU and −0.60% with xIELU. Early in training xIELU is worse than SwiGLU, and it catches up at 20 billion tokens.

**Perplexity in the comparison of components (1.1 billion, context 1,024, 4 billion tokens, about 3.6 tokens per parameter):**

| Activation function | perplexity |
|---|---|
| xIELU | 17.52 |
| xIPReLU | 17.52 |
| SwiGLU | 17.58 |
| xSiLU | 17.91 |
| ReLU² | 18.12 |
| SiLU | 18.57 |

From SiLU to ReLU², −0.84%; from ReLU², −1.04% with SwiGLU and −1.16% with xIELU.

**Time per update (1.1 billion, 4 GH200s):**

| Activation function | Milliseconds |
|---|---|
| ReLU² | 534 |
| SiLU | 540 |
| xIPReLU (unoptimized) | 542 |
| GELU | 544 |
| SwiGLU | 549 |
| xIELU (unoptimized) | 560 |

- Downstream evaluation at 3 billion is about the same, with SwiGLU at 39.6 and xIELU at 39.7. Which one wins depends on the task.
- As limitations, it lists that speed needs a dedicated kernel, that checks on larger models are needed, and that training at low precision may need changes. There are no results with Muon or for small models.

Read: the whole body and the settings in the appendix. The numbers in the figures were not read.
