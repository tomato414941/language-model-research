# RETRO

Borgeaud et al., Improving Language Models by Retrieving from Trillions of Tokens (arXiv 2112.04426, ICML 2022). DeepMind. Parameter counts exclude the embeddings.

## Method

- The input is split every 64 tokens, and similar text is searched for in a database (DB) with frozen BERT vectors. The 64 tokens found and the 64 tokens that follow them are used. The retrieval during training is computed beforehand and kept as data.
- A 2-layer encoder (width 896, about 19 million parameters) reads the text found, and the main network refers to it with cross-attention once every 3 layers. The parameters grow from 132 million to 172 million (+30%) and from 7 billion to 7.5 billion (+8%).
- Every model is trained on 419 billion tokens with AdamW. The DB at training is 600 billion tokens, with 2 neighbors retrieved. The DB at evaluation is 1.75 trillion tokens (only 4% for books).
- The training DB is built from the same subsets as the training data (§2.1). Since retrieving a sequence's own continuation breaks causality, neighbors from the same document as the training sequence are excluded (§2.3). No measurement without this exclusion appears in what was read.
- The index is 215 GB for the 4-billion-token Wikipedia and 93 TB for all 1.75 trillion tokens. Retrieval takes 10 milliseconds.

## Results

BPB values. From left to right, 172 million, 425 million, 1.5 billion and 7.5 billion parameters; the baselines are 132 million, 368 million, 1.3 billion and 7 billion respectively.

| Evaluation data | Baseline | With retrieval |
|---|---|---|
| C4 | 0.98, 0.92, 0.84, 0.78 | 0.82, 0.77, 0.71, 0.66 |
| Wikipedia September 2021 (23 articles written after the DB was built) | 0.85, 0.78, 0.71, 0.65 | 0.79, 0.73, 0.66, 0.61 |
| Curation Corpus (summaries, not in the DB) | 0.69, 0.63, 0.56, 0.52 | 0.66, 0.61, 0.55, 0.50 |

- DB size (smallest model, C4, baseline 0.98): 0.97 at 2 billion and 4 billion, 0.96 at 9 billion–36 billion, 0.95 at 90 billion, 0.94 at 180 billion, 0.92 at 360 billion, 0.88 at 900 billion and 0.82 at 1.75 trillion.
- Retrieving more neighbors at evaluation helps up to 10 for the smallest model and up to 40 for the largest, and hurts beyond that.
- With retrieval turned off, it is nearly the same as the baseline.
- The Pile: the abstract says it "obtains comparable performance to GPT-3 and Jurassic-1 on the Pile, despite using 25× fewer parameters". Counting the table's 16 entries, the 7.5-billion-parameter RETRO beats Jurassic-1 (178 billion) on 10 and Gopher (280 billion) on 9.
- Question answering (Natural Questions): 30.4% without retrieval, RETRO 45.5%, FiD 51.4%.
- Retrofitting: the weights are frozen, and only the added parts are trained, on 3% of the pretraining tokens. On C4 it approaches the value of training from scratch. On Curation Corpus the smallest model's perplexity stays at about 20.7, the same as the baseline, while training from scratch gives about 18 (read from Figure 5).
- Small data: trained on Wikitext103 alone, RETRO from scratch overfit and was worse than the baseline. Retrofitted onto the baseline, retrieving from the 4-billion-token Wikipedia took the perplexity from 21.53 to 18.46.
- Contributions of the parts (smallest model, 157 billion tokens, C4): standard 0.822, no retrieval 0.987, only the passage found 0.950, only the continuation 0.895. With 1 neighbor retrieved at training 0.858, with 4 0.847.

## Separating out verbatim copying

For each evaluation chunk, the longest match with the 10 neighbors found in the DB is measured, and BPB is measured only on the chunks with short matches (§2.6). The paper concludes that the part of the improvement due to leakage is modest.

Reading the smallest model's values from Figure 6, the change from the baseline to retrieval is as follows.

| Evaluation data | All chunks | Match of 8 tokens or fewer |
|---|---|---|
| C4 | 0.980→0.817 | 1.015→0.977 |
| Wikipedia September 2021 | 0.852→0.794 | 0.862→0.838 |
| Curation Corpus | 0.646→0.619 | 0.655→0.629 |
| Wikitext103 | 0.915→0.335 | 0.923→0.587 |

## What is not stated, and what does not match

- The training time compared with the model without retrieval, and the cost of building the DB, are not stated. The time figure gives values relative to RETRO, and there is no comparison at the same cost either.
- The 7.5-billion-parameter line in the filtering figure does not match the table. Wikipedia September 2021 with retrieval is about 0.58 in the figure and 0.61 in the table. The baseline on The Pile's arxiv is about 0.42 in the figure and 0.742 in the table.
- The text says it does not beat the baseline on dm_mathematics and ubuntu_irc, but in the table it is slightly better (1.177→1.164, 1.200→1.178).

The main text, the appendix and the figures of the HTML version were read. The values in the figures were read off the axis ticks.
