# Optimal Embedding Learning Rate and Vocabulary Size

Hayou and Liu, Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size (arXiv 2506.15025).

- μP treats the vocabulary size as fixed, but in practice the vocabulary is larger than the width. For the case where the vocabulary also grows, the paper derives that the best embedding learning rate is √width times that of the hidden layers.
- In a small experiment (2 layers, Wikitext2, vocabulary 32,768, width 256 to 4,096, Adam, 10,000 updates), the best embedding learning rate falls as the width grows. The spread is large. For the hidden layers and the output, the same learning rate and initialization work best.
- A 1-billion model (24 layers, width 2,048, RoPE, SwiGLU) was trained on 1.75 trillion tokens, comparing the same learning rate with √width times (about 45×). √width times is faster from start to finish, and better in both the final training loss and the Wikitext test. Going larger than that makes little difference.

Read: the abstract, the introduction, the experimental part of Section 3, and Section 4. The numbers in the figures were not read.
