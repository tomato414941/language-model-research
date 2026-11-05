# DeepSeekMoE

Dai et al. (arXiv 2401.06066, 2024-01). An MoE that splits the experts finely and keeps shared experts apart.

A comparison at 9 layers, width 1280 and 100 billion tokens (about 500 tokens per parameter).

| | Dense | Hash Layer | Switch | GShard | DeepSeekMoE |
|---|---|---|---|---|---|
| Total parameters | 200 million | 2 billion | 2 billion | 2 billion | 2 billion |
| Active parameters | 200 million | 200 million | 200 million | 300 million | 300 million |
| Compute per 2,000 tokens | 2.9T | 2.9T | 2.9T | 4.3T | 4.3T |
| The Pile loss | 2.060 | 1.932 | 1.881 | 1.867 | 1.808 |
| HellaSwag | 38.8 | 46.2 | 49.1 | 50.5 | 54.8 |

- A dense model that runs all the experts (Dense×16, 24.6T) has nearly the same loss on The Pile (1.806 and 1.808).
- The model with 16 billion in total, on 2 trillion tokens, matches DeepSeek 7B's results with about 40% of the compute. It is weak on multiple-choice tasks, which the authors attribute to its few attention parameters.
- Preliminary results at 145 billion in total match DeepSeek 67B with 28.5% of the compute.

The abstract, the experimental setup, Table 1, the conclusion and the openings of the sections were read. The section analyzing the experts and the appendix were not read.
