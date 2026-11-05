# Primer

So et al., Primer: Searching for Efficient Transformers for Language Modeling (arXiv 2109.08668). The origin of ReLU².

- With perplexity at a fixed training time as the objective, it searched for the model's architecture by evolution. The most effective change was turning the MLP's ReLU into ReLU², and the next was a depthwise convolution after the Q, K and V projections (MDHA). The model with only these two is called Primer-EZ.
- ReLU² differs greatly in shape from GELU and Swish when x is large. Since it equals ReGLU with its two matrices the same, the paper argues that it has GLU's advantage without adding weights.

Perplexity on T5, C4, 110 million parameters, 525K steps (right of Figure 5 and Table 5).

| Activation function | Perplexity | Loss difference (our own calculation) |
|---|---|---|
| ReLU | 20.61 | |
| GELU | 20.34 | −0.44% from ReLU |
| Swish | about 20.2 (figure) | |
| ReGLU | about 20.2 (figure) | |
| SwiGLU | about 20.0 (figure) | |
| ReLU² | about 19.9 (figure) | about −0.7% from GELU |

The table also has a comparison that adds ReLU² to Synthesizer: at 145 million, from 20.35 to 19.55 (−1.3% in loss).

- In the insertions and removals on LM1B at about 35 million (Figure 26), ReLU²'s effect is large in T5 and very small in T2T, but in the good direction in both. The vertical axis is normalized, and the numbers cannot be read.
- In the same figure, the change that widens the MLP 12× (d_model to 384, d_ff to 4,608) is good in both. The text leaves it out of Primer and Primer-EZ, saying that it helps a lot at about 35 million but helps less in larger models.
- The optimizer is Adafactor, with the hyperparameters kept as for the Transformer. The context is 64 on LM1B and 512 on C4 and PG19.

The values from Figure 5 were read from the heights of the bars, and the GELU and ReLU values match the table. The relevant parts of the main text, Figures 4, 5, 6 and 26, and parts of Tables 4 and 5 were read.
