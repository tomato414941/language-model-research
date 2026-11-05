# autoresearch

Karpathy's `karpathy/autoresearch` (2026-03). An AI agent repeatedly edits a simplified version of nanochat's training code on one GPU, trains it, and keeps the change if the result is better.

- Fixed (`prepare.py`, not to be changed): 5 minutes of training (excluding startup and compile), ClimbMix (`karpathy/climbmix-400b-shuffle`, with the last file for validation), a BPE with a vocabulary of 8,192, a context of 2,048, and the function that measures validation BPB.
- Free (`train.py`): the model's shape and size, the optimization, the batch and so on. Adding packages is not allowed, and memory is a loose constraint.
- The criterion is "simpler is better". Less code at the same performance counts as an improvement.
- The default settings are 8 layers, width 512, 4 heads (128 dimensions per head), 524,288 tokens per update, Muon and AdamW, and FlashAttention-3.
- Optimizer settings (`train.py`): Muon (Polar Express, NorMuon, cautious weight decay) for matrices and AdamW for everything else. The learning rates are 0.04 for matrices, 0.6 for the embedding, 0.004 for the output and 0.5 for scalars, and the AdamW values are multiplied by (width/768)^−0.5. β (0.8, 0.95), ε 1e-10. Momentum rises from 0.85 to 0.95 over the first 300 updates. Weight decay falls linearly from 0.2 to 0. The learning rate has no warmup, stays constant over the first half and falls linearly to 0 over the second half. No gradient clipping. The embedding and the output are not shared, and their initial scales are set separately.
- Model (`train.py`): RoPE, RMSNorm on Q and K, RMSNorm without learned weights, ReLU², zero initialization of the attention and MLP outputs, separate input and output embeddings (the input with standard deviation 1.0 followed by RMSNorm, the output 0.001), weights on x0 and resid, a logit softcap of 15, the embedding in bf16, and the window halved in 3 of every 4 layers.
- How it measures (`prepare.py`): BPB over about 21 million tokens of the last file with a context of 2,048; each row starts with BOS and is packed with documents. Training reads the data the same way, and `prepare.py` defaults to 10 files.
- The example run in `program.md`: val_bpb 0.9979, 300 seconds of training, 326 seconds in all, 45GB of memory, mfu 39.80%, 499.6 million tokens, 953 updates, 50.3 million parameters. The README says it was checked on an H100.
- The version Karpathy revised (commit 438a26e; the version Recursive measured as Karpathy's best) uses 2^18 (262,144 tokens) per update, 9 layers, `ASPECT_RATIO` 57, no learning-rate warmup, and a decay phase of 75%.

Read: the README, `program.md`, the constants and the model definition in `prepare.py` and `train.py`, and the constants in `train.py` at 438a26e. The results of the experiments (reportedly about 20 improvements in about 700 trials) were seen only second-hand.
