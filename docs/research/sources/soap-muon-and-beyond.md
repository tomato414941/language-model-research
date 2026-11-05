# SOAP, Muon, and Beyond

Khona et al. (NVIDIA), SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales (arXiv 2607.20548, 2026-07).

- Trains, among others, a MoE with 30 billion parameters in total (3 billion active) on 3 trillion tokens, and a hybrid with Mamba with 72 billion in total (8 billion active) on 1 trillion tokens.
- Transfers the learning rate by matching the size of the updates, and compares AdamW, Muon and SOAP. It says AdamW gets worse beyond a certain batch, while Muon and SOAP hold up to 100 million tokens per update.
- SOAP's loss spikes at large batches. This was resolved with QR orthogonalization at every step and KL-Shampoo's estimate of the covariance.
- Compared on Qwen3's shape with 30 billion in total, KL-SOAP slightly beats Muon. It recommends KL-SOAP when memory allows. This comparison does not separate Q, K and V.
- Nesterov-style momentum did not improve convergence. It notes that ε was not tuned.

Read: the abstract, the introduction, the beginning of the comparison method, Section 5.5, the beginning of the implementation, and the conclusion. The results and tables of Section 5.3 were not read.
