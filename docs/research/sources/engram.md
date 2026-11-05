# Engram

Cheng et al. (DeepSeek, Peking University), Conditional Memory via Scalable Lookup (arXiv 2601.07372, v2 in 2026-07).

- It hashes the preceding 2–3 tokens to look up a learned table, and adds the result through a gate made from the internal state. In the 30-layer model it goes in at layers 2 and 15. At inference the table can be kept off the GPU.
- A comparison at 262 billion tokens, 3.8 billion active parameters and 30 layers.

| | Dense model | MoE | Engram 27B | Engram 40B |
|---|---|---|---|---|
| Total parameters | 4.1 billion | 26.7 billion | 26.7 billion | 39.5 billion |
| Validation loss | 1.768 | 1.634 | 1.622 | 1.610 |
| The Pile loss | 2.091 | 1.960 | 1.950 | 1.942 |
| MMLU | 48.6 | 57.4 | 60.4 | 60.6 |

- With the total parameters held equal and the MoE's share moved to the table, the optimum is 20–30%. At 6e20 FLOPs, from 1.7248 to 1.7109. Read from the figure, moving just 5% gives 70–90% of the effect.
- Adding the table to an MoE of 3 billion in total (568 million active) on 100 billion tokens takes the validation loss from 1.808 to 1.788 with 260,000 rows and 1.743 with 10 million rows (read from the figure).
- Varying where one 1.6B table goes in a 12-layer model, layer 2 is best at 1.7705 and layer 12 gives 1.7826 (read from the figure). Splitting it between layers 2 and 6 gives 1.768. From there, removing the gate gives about 1.773, removing the vocabulary compression about 1.7735, and removing the convolution about 1.7685.
- With a table of 100 billion parameters kept off the GPU, the slowdown is 1.9% and 2.8%.
- Cutting off the table at inference leaves 29% of the original score on TriviaQA, 36% on MATH and 93% on C3. The authors caution that the results are noisy because the conditions differ from training.

Inconsistencies: the abstract's MMLU +3.4 is +3.0 in the table (+3.4 is MMLU-Redux). The text says OverEncoding gives no meaningful improvement on top of an MoE, but in the figure it goes from 1.808 to about 1.782 with 1.25 million rows. Engram 40B is lower than 27B on HumanEval, MBPP and C3; the authors guess it is undertrained but did not check.

The main text, the appendix's tables and the main figures were read. The cited works were not read.
