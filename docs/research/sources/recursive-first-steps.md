# First Steps Toward Automated AI Research

Recursive's release (`recursive-org/first-steps-toward-automated-ai-research` on GitHub, 2026-06-11). Training code found by a system that proposes and tests ideas automatically, and the results measured. It is the company's own report; no external replication has been seen.

## NanoChat (autoresearch's task)

It trains for 5 minutes on one B200 and measures validation BPB. The data and tokenizer are the same as autoresearch's (ClimbMix, vocabulary 8,192, context 2,048; each row starts with BOS and is packed with documents). Measured with 10 seeds. The mean and standard deviation were computed from the public CSV.

| Solution | Mean | Standard deviation |
|---|---|---|
| Vanilla Transformer | 1.0587 | 0.0101 |
| Karpathy's starting point | 0.9751 | 0.0112 |
| Karpathy's best | 0.9646 | 0.0154 |
| The community's best | 0.9372 | 0.0079 |
| Solution found from the vanilla Transformer | 0.9344 | 0.0122 |
| Solution found from Karpathy's starting point | 0.9109 | 0.0067 |

What the best solution (`optimized_from_karpathy.py`) contains.

- **Shape:** 8 layers, width 768, six heads of 128 dimensions. 147,456 tokens per update (72 sequences × 2,048). Value embeddings on every other layer.
- **n-gram tables:** It hashes 2 tokens and 3 tokens to look up tables, and adds the result to attention's value through a gate. Two tables of 64× the vocabulary (524,288 rows) and half the width are concatenated. 2 tokens in 4 layers and 3 tokens in 3 layers, about 2.8 billion parameters in the tables alone (our own calculation). The hash changes from layer to layer. The tables use RMSProp without a first moment, which halves the memory, and their learning rate is not lowered.
- **Attention windows:** three layers see a quarter of the context and one layer all of it (`TTTL`).
- **Normalization:** RMSNorm on both the input and the output of the MLP.
- **Learning rate:** No warmup; after the first 5%, it falls linearly to 5% by the end. Adam's decay starts when 65% remains. Matrices 0.04, the lookup tables 0.6, the output table 0.004.
- **Others:** Muon's momentum lowered from 0.95 to 0.79. Adam's β1 lowered from 0.8 to 0.55. There are 3 periods in which the weight decay is temporarily raised 3–5×.

The solution found from the vanilla Transformer (`optimized_from_vanilla.py`) has 7 layers, starts with 131,072 tokens per update (64 sequences × 2,048), and at 50% of training raises it to 262,144 and the learning rate by 1.2×.

There is no table of how much each change helped.

## speedrun

From record #83 (80.61 seconds on the same machine) to 77.34 seconds (13 runs, validation loss 3.2789). Extending FP8 to attention's inputs and outputs, adding annealed noise to NorMuon, using cautious Adam for the tables, fusing the computation of ReLU², and more. From a different, weaker starting point it reached about 185 seconds, which includes attention within documents combined with windows, a 2-token table, and Muon on every update.

## What was read

The repository's README, the NanoChat and speedrun READMEs, `val_bpb.csv`, `lib.py`, the settings and model parts of `optimized_from_karpathy.py`, and the batch and shape settings of `optimized_from_vanilla.py` were read. The original post could not be retrieved (404) and was not read.
