# The RMSNorm paper

Zhang and Sennrich, Root Mean Square Layer Normalization (arXiv 1910.07467, NeurIPS 2019).

- Removes the subtraction of the mean from LayerNorm and divides only by the root mean square. The paper's form has learned weights.
- It says that on several models the quality is about the same as LayerNorm's, and the time 7–64% shorter.
- A table for a Transformer (width 512, machine translation, V100, TensorFlow).

| Normalization | BLEU (Test14, Test17) | Time for 1,000 steps |
|---|---|---|
| LayerNorm | 26.6, 27.7 | 248 seconds |
| RMSNorm | 26.8, 27.7 | 231 seconds (−6.9%) |

There is no comparison for language-model pretraining. Read: the abstract, the introduction, and the table in the machine translation section.
