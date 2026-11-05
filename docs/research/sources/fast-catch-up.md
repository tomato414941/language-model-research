# Fast Catch-up, Late Switching

Wang et al., Fast Catch-up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws (arXiv 2602.14208, ICLR 2026).

With the total amount of data fixed, it studies in theory and in experiments how to grow the batch.

- On hard tasks, it is best to run most of training with a small batch and switch late. On easy tasks, it is best to keep growing the batch, and with few stages a large batch from the start is best. Whether a task is hard or easy is set by an exponent of the theoretical model (linear or kernel regression), and it is not stated which one language-model pretraining is.
- After the switch, the loss quickly catches up with the large-batch curve. Decaying the learning rate with cosine does not change this shape or the advantage of switching late (Appendix B.6).
- For a 50 million parameter LLaMA, the best switch is at about 70% of the tokens. The optimal points over budgets of 1.3 billion to 5 billion tokens fit a power law well (R²=0.990).
- The large experiments are MoE models of 1 billion and 1.1 billion (0.4 trillion and 1 trillion tokens), with batches such as 640→1280 and 1024→2048.

Small-scale settings (Appendix B): a LLaMA-style model in NanoGPT's code, C4, context 256. AdamW (β 0.9, 0.95, weight decay 0.1, gradient clipping 1.0), 1,000 warmup updates, and a learning rate of 1e-3 at 50 million and 200 million and 5e-4 at 500 million. Tokens are about 20× the parameters. The 50 million model goes from 128 to 256 sequences per update (33,000 to 66,000 tokens, our own calculation). Whether the learning rate was changed when the batch changed is not stated.

The loss numbers are only in the figures, not in the text or tables. The HTML main text and Appendix B were read through a summary.
