# Practical Efficiency of Muon

Essential AI, Practical Efficiency of Muon for Pretraining (arXiv 2505.02222, 2025-05).

- From 100 million to 4 billion parameters, Muon reaches the same loss with 10–15% fewer tokens than AdamW. With larger batches, this gap holds or widens.
- It keeps only the first moment, so it uses the least memory of the optimizers compared (less than AdamW). The added compute shrinks in inverse proportion to the batch size.
- The embeddings and normalization are updated with Adam.
- With Muon and µP together, the learning rate and weight decay transferred up to 3.7 billion parameters and context 8,192.

The abstract, part of the introduction, part of the experimental settings and the conclusion were read. The results section was not read.
