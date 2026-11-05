# Muon is Scalable

Liu et al. (Moonshot AI and others), Muon is Scalable for LLM Training (arXiv 2502.16982, 2025-02).

- What it adds to the original Muon (muon.md): weight decay in the same way as AdamW (because the weights grow too large when training is extended), a correction of the update size, which changes with the matrix's shape, and a factor of 0.2 that matches AdamW's update size (RMS 0.2–0.4). It says this lets AdamW's learning rate and weight decay be used unchanged.
- Newton-Schulz with 10 iterations is no better than with 5. Tuning the momentum gave no stable difference, so it stays at 0.95.
- RMSNorm, the output and the embeddings are updated with AdamW. Handling everything with Muon is left as future work.
- Scaling laws: Llama-shaped, 399 million to 1.5 billion parameters (excluding embeddings), context 8K, the optimal number of tokens (8.92 billion to 38.9 billion), AdamW grid-searched. The loss is 2.506·C^−0.052 for Muon and 2.608·C^−0.054 for AdamW. Muon needs about 52% of the compute for the same performance.
- Moonlight: an MoE with 3 billion active and 16 billion total parameters, trained on 5.7 trillion tokens.
- The optimizer takes 1–3% of the time of the forward and backward passes.
- The conclusion says Muon can replace AdamW as the standard optimizer.

The method (2.1, 2.2), the scaling laws (3.2), the first half of pretraining (3.3), and the discussion and conclusion were read. The details of distributed training (2.3), the update-size experiments (3.1) and fine-tuning (3.5) were not read.
