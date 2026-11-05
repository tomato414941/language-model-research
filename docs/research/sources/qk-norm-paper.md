# The QK-norm paper

Henry et al., Query-Key Normalization for Transformers (arXiv 2010.04245).

- It replaces attention's dot product with cosine similarity multiplied by a learned scale. The idea is to keep the dot product from growing so large that the softmax saturates.
- The target is low-resource machine translation. Over five language pairs, the test BLEU rose by 0.928 on average. In a separate small comparison, it went from 19.4 to 24.33.
- There are no results on language-model pretraining.

The abstract, the introduction and the captions of the results tables were read.
