# Lost in Backpropagation

Godey, Artzi, Lost in Backpropagation: The LM Head is a Gradient Bottleneck (arXiv 2603.10145, 2026-03).

It says 95–99% of the gradient's magnitude is lost where the gradient is squeezed from the vocabulary dimension down to the hidden-state dimension.

Experiments at 2 billion: 6 layers, width 4,096, vocabulary 49,152, about 11 billion tokens. The output embedding table was factored as W=AB, and the width D was varied from 32 to 4,096.

- D=4096 reaches D=32's final loss in 700 million tokens (16× faster).
- The mean score at 11 billion tokens is 42.03 at D=32, 49.07 at D=256 and 50.77 at D=4096. Above D=1024 the gains slow down.
- The larger the vocabulary relative to the width, the harder it gets (vocabulary 1,024 to 131,072 at D=576). As a remedy, it only implicitly suggests keeping the vocabulary small, and there are no experiments with small models.

The main text of v1's HTML was read through a summary.
