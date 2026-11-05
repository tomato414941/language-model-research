# Gemma's logit soft-capping

The technical reports of Gemma 2 (arXiv 2408.00118) and Gemma 3 (arXiv 2503.19786).

- Gemma 2 bounds the logits of each attention layer and of the final layer with `soft_cap * tanh(logits / soft_cap)`. The values are 50 for attention and 30 for the final layer. It cites Bello et al. (2016) as the source of the method. There is no comparison of the effect.
- Gemma 3 says it "replaced the soft-capping of Gemma 2 with QK-norm". No reason or comparison is given.

In both, only the paragraphs where soft-capping and QK-norm appear were read.
