# modded-nanogpt's retrieval hints (PRs #367 and #380)

PRs to `KellerJordan/modded-nanogpt`, both unmerged as of 2026-11-03. They count exact matches and n-grams of the training data on the CPU and use them as hints. The records compete on the time to reach a target validation loss (about 3.28), and accuracy is not measured.

## PR #367

- On eight GPUs, it took record #92's 40.475 ± 0.046 seconds and validation loss of 3.28188 on the same machine to 6.702 ± 0.018 seconds and 3.27406.
- It builds exact-match indexes of 6, 10 and 20 tokens on the CPU and passes the token that followed each match to the main model.
- **Separate indexes for training and validation:**
  - Training rows are built only from "documents of steps before that step" (`StreamIndex`).
  - Validation rows are built from the first 370 shards (37 billion tokens) (`SlotIndex`).
  - The number of candidates per position is 0.2 on average and 0.7 at most in training, and 4.9 at most in validation (from a code comment).
- The match length, the count and the share of the top tokens are passed together to the main model.
- The maintainer's comment: the idea came up the year before too, but was not implemented because it is less appealing in terms of generalization and scale. The comment says this PR shows the gap between IID perplexity and generalization. The rules for using the CPU are undecided.

## PR #380

- On top of #367, it stacks a smaller model, added short-context indexes, CPLM and more. Against #367's 21.749 seconds and 3.27460 on the same machine: 9.653 ± 0.041 seconds and 3.27588.
- **Counting on shards that training does not read:** exact counts of orders 2 to 8 are taken on "training shards the loader does not read" and mixed into the model's probabilities after training. The mixing weights are fitted per order and count bucket on positions other than the training positions ("weights never fit on training positions").
- Loss and time as each piece is added (the author's table): making the model smaller and shorter gives 3.4493 and 10.89 seconds. More hints give 3.4021. Chained counts give 3.3491. An in-document cache gives 3.3316. A hidden-state gate gives 3.3167. CPLM gives 3.2778.
- The author also says the use of the CPU is a gray area under the rules.

The two PRs' descriptions and comments, the beginnings of `retrieval.py`, `stream.rs` and `exact_counts.py`, and the lines of `config.py` related to the indexes were read. The full implementation was neither read nor run.
