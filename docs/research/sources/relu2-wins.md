# ReLU² Wins

Zhang et al., ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs (arXiv 2402.03804).

- The goal is sparse computation that skips neurons unused at inference; it is not about training efficiency.
- With 1.3 billion models matched in weights, data and hyperparameters, it trained SwiGLU, ReLU, ReGLU and ReLU² up to 100 billion tokens. At the end, SwiGLU, ReGLU and ReLU² are nearly the same, and ReLU is much worse.
- ReLU² lags slightly behind the GLU forms along the way and catches up near the end.
- ReLU² loses the least performance when made sparse (under 0.1% at about 90% sparsity). It also says ReLU and ReLU² suit hardware better than SwiGLU and ReGLU.

It does not compare GELU. The abstract, the introduction and the section that trained the models were read; the loss comparison is only in figures, and its numbers were not read.
