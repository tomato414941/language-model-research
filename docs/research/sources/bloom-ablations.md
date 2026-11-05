# BLOOM's architecture comparisons

Le Scao et al., What Language Model to Train if You Have One Million GPU Hours? (arXiv 2210.15424). Comparisons made to decide BLOOM's architecture.

For activation functions, the mean of the EAI results of models trained on 112 billion tokens of The Pile is 42.79 for GELU and 42.95 for SwiGLU. SwiGLU is slightly better.

BLOOM itself used GELU. What looked like SwiGLU's slowness was later found to come from the inner width of 5,456 not suiting the GPU, so the paper recommends SwiGLU for the next models.

It does not compare ReLU². Only the section on activation functions was read.
