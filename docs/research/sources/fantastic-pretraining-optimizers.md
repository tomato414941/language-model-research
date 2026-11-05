# Fantastic Pretraining Optimizers

Wen et al., Fantastic Pretraining Optimizers and Where to Find Them (arXiv 2509.02046, v3 in 2026-10). Part of Marin.

- It compares 11 methods (AdamW, NAdamW, Mars, Cautious, Lion, Adam-mini, Muon, Scion, Kron, SOAP, Sophia), each tuned on its own. They are compared by the validation loss on C4/EN at the last checkpoint.
- Llama-shaped, 32 layers, 130 million, 300 million, 520 million and 1.2 billion parameters, context 4,096. The amount of training is 1–8×, where 1× is 20 tokens per parameter. Each update takes 400,000 tokens or more. 128 TPU-v5lite. The comparison is by tokens, and it says the added compute can be kept under 10%.
- Against a tuned AdamW, the speedup is at most 1.4×. Matrix-based methods give about 1.3× at 500 million parameters or fewer, and per-weight methods under 1.2× on average.
- The larger the model, the smaller the difference. At 1.2 billion, NAdamW, Muon and SOAP give about 1.1×, and the benchmark mean does not improve either (at 8× training, AdamW 67.15, Muon 66.98, NAdamW 66.70). From scaling laws, it predicts that at 7 billion with 1× training, Muon's loss is slightly higher.
- At 1×, Muon is the best; at 8× and above, SOAP and Kron catch up with it or exceed it.
- It says GPT-3's recipe (peak learning rate 6e-4) is a weak baseline, since tuning only the learning rate makes it about 2× faster at 100 million.
- An example of tuning AdamW (the table's heading says 130 million at 1×, the text says 300 million): learning rate 0.008, weight decay 0.1, minimum learning rate 0, warmup 2,000, gradient clipping 2, batch 128 (524,288 tokens at context 4,096), β1 0.9, β2 0.98, ε 1e-10. The loss falls from 3.298 to 3.263.
- The best weight decay differs by method (AdamW about 0.1, Lion about 0.6, Kron about 0.5). Reusing settings across methods makes the comparison unfair. Comparing by intermediate losses changes the ranking. Sophia is not much faster than AdamW at 500 million or below.
- It attributes the points where its results differ from Semenov et al. (benchmarking-optimizers.md) to differences in batch size and learning-rate tuning.

How many times faster than AdamW each method is, read from the 130 million figure.

| Method | 1× training | 8× training |
|---|---|---|
| Muon | about 1.4 | about 1.38 |
| SOAP | about 1.25 | about 1.40 |
| Kron | about 1.2 | about 1.38 |
| Mars | about 1 | about 1.22 |
| NAdamW | about 1 | about 1.18 |
| Cautious, Lion | about 1 | about 1.15 |
| Adam-mini | about 1 | about 0.96 |

The abstract, the introduction, the related work, part of the experimental method, the results and the conclusion were read. The appendix's individual tables were not read. Of the figures, only the one above was read, off its axis ticks.
