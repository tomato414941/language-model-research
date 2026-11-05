# Polar Express

Amsel et al., The Polar Express (arXiv 2505.16932). The method and the comparison at 774 million parameters and 1 billion tokens are in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/polar-express.md). Only what follows on from it is written here.

- The experiments use GPT-2's shape, FineWeb, 32,768 tokens per update (32 sequences, context 1,024), 4 H100s, bf16 and 5 iterations. The learning rate is constant for the first 40% and then falls linearly. The embeddings, the output and the positional encodings are updated with AdamW.
- Validation loss at 124 million parameters and 1 billion tokens (about 8 tokens per parameter), without weight decay: AdamW 4.197 (learning rate 0.0005), the original Muon 3.639 (0.01), You's method 3.629 (0.01), Polar Express 3.588 (0.005).
- Validation loss at 774 million parameters and 10 billion tokens, with weight decay 0.1: the original Muon 2.921, You's method 2.919, Polar Express 2.913 (all at learning rate 0.002).
- 2–3 iterations are worse, and neither going beyond 5–6 nor computing exactly with SVD helps. The number of iterations barely changes the run time, but using SVD doubles the time of the update.

The abstract, the introduction and the experiments section were read. The proofs and the appendix were not read.
