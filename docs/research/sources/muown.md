# Muown

Lion et al., Muown: Row-Norm Control for Muon Optimization (arXiv 2605.10797, 2026-05).

It splits each row of a weight into a magnitude (the row norm) and a direction, and updates the direction with Muon and the magnitude with Adam. It is weight normalization done inside the optimizer, and the forward pass does not change. The authors argue that the weights' spectral norm grows under Muon because the row magnitudes grow, and they made the magnitudes variables that can be moved directly.

## Settings

- nanoGPT with RoPE, RMSNorm and SwiGLU added (shared embedding and output). FineWeb-Edu, 524,288 tokens per update, WSD (2% warmup, decaying over the last 20%).
- 124 million (12 layers, width 768) trains on 5 billion tokens at context 1,024 (about 40 tokens per parameter). 500 million trains on 10 billion, 1 billion on 20 billion and 2.7 billion on 53 billion tokens, at context 2,048.
- Muon and Muown match their learning rates in Kimi's way (multiplying the update by 0.2·√max(m, n)). Muown uses no weight decay by default. Muon uses 0.1, the best value found in a search at 124 million.
- The extra memory is four vectors as long as the number of rows (about 2MB in the reported runs). The update time grows by 1.5% when each GPU processes every matrix, and is the same as Muon's when the matrices are split across GPUs.
- Weights whose rows are 0 cannot be split into magnitude and direction. The authors write that output projections that start at 0 and the like must either start at small values or be left out.

## Results

Perplexity. The differences are converted to loss (our own calculation).

| Size | Seeds | Muown | Best Muon | Difference |
|---|---|---|---|---|
| 124 million | 2 | 20.04 | 20.43 (weight decay 0.1) | −0.6% |
| 500 million | 3 | 14.13 | 14.37 (weight decay 0.1) | −0.6% |
| 1 billion | 1 | 11.90 | 12.20 (weight decay 0) | −1.0% |
| 2.7 billion | 1 | 9.57 | 9.82 (weight decay 0) | −1.1% |

- The 124 million values were read off Figure 3. The others come from Tables 1 and 2.
- It rarely diverges even at higher learning rates, and the range of good learning rates is wide. Muon diverges at higher learning rates (110.19 at 3e-3 for 1 billion; the 2e-3 run at 2.7 billion was stopped).
- In modded-nanogpt's code too, the loss is about 0.4% and 0.6% lower than Muon's at widths 768 and 1,280 (values read off Figure 8).
- In a Qwen2 0.5B-style model too, best against best gives 14.57 and 14.82 (Table 4).

## Comparison with NorMuon (124 million, Appendix A.3)

- Compared at 11 learning rates. The best values read off Figure 9 are 20.03 for Muown (learning rate 0.01) and 20.11 for NorMuon (0.02), a difference of about 0.13% in loss. Neither uses weight decay, and the number of seeds is not stated.
- Plain Muon under the same settings (no weight decay) gives 20.52, so NorMuon also captures most of the gap to Muon (our own calculation).
- The authors say that near the optimum it is as good as NorMuon or slightly better, with a wider range of learning rates that stay stable. It is not compared with NorMuon with weight decay.

The HTML main text and Appendices A and C were read (the tables directly from the HTML). The values in the figures were read from the figures' SVGs. The proofs (Appendix B) were not read.
