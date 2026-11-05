# Tokenizer-Agnostic Engram Module

Lim and Chieu (arXiv 2607.29065, 2026-07). A study that makes the Engram table independent of the tokenizer, which also serves as an independent replication. One training run each.

- SmolLM2 1.7B on 32 billion tokens. The baseline has a benchmark mean of 0.556 and a wikitext BPB of 0.861.
- Adding an 800 million Engram (with the original hash) gives a mean of 0.605 and BPB 0.919. Evaluating with the table cut off gives a mean of 0.587 and BPB 1.034.
- The polynomial-hash form also gives a mean of 0.612 and BPB 0.925.
- Qwen3.5 700 million on 100 billion tokens. The baseline has a mean of 0.560 and BPB 0.821. Using a frozen table trained with another model gives a mean of 0.584 and BPB 0.811.
- Compared with tables of up to 3 tokens (N=3) or up to 7 tokens, a table of single tokens alone does not help.

Read: the body. The cited works were not read.
