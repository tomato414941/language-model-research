# STEM

Sadhukhan et al., STEM: Scaling Transformers with Embedding Modules (arXiv 2601.10639, ICLR 2026).

It replaces the up-projection matrix of a gated MLP with a per-layer table looked up by token. `y = W_d · (SiLU(W_g · x) ⊙ U[t])`. U is vocabulary × MLP width. Making the gate side the table instead is worse.

| Model | ARC-E | ARC-C | PIQA | HellaSwag | Mean of 8 tasks |
|---|---|---|---|---|---|
| 350 million baseline | 57.66 | 30.55 | 69.42 | 49.68 | 49.72 |
| 350 million STEM (a third of the layers) | 63.01 | 32.68 | 70.18 | 52.38 | 50.90 |
| 1 billion baseline | 66.98 | 41.88 | 73.44 | 59.65 | 55.82 |
| 1 billion STEM (a third of the layers) | 65.95 | 42.03 | 75.00 | 60.37 | 56.63 |

- Pretraining is 100 billion tokens for 350 million and 1 trillion tokens for 1 billion. On top of this there are 100 billion tokens of mid-training and 20 billion tokens of context extension.
- FLOPs per token go 0.74→0.70G and 3.00→2.83G. The gain in FLOPs per point is 1.08× with a third of the layers, 1.20× with half and 1.33× with all. There are no measured speeds.
- Total parameters are 1.14 billion for the 350 million STEM and 6.75 billion for the 1 billion STEM. The tables can exceed GPU memory, and moving them off the GPU during training is not implemented.

The HTML main text was read through a summary. The vocabulary size was not in what was read.
