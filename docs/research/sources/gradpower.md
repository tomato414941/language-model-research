# GradPower

Wang, Wang, Zhang et al., GradPower: Powering Gradients for Faster Language Model Pre-Training (arXiv 2505.24275, v4 in 2026-06, ICML 2026).

A one-line change that turns the gradient into `sign(g)·|g|^p` element-wise before passing it to the optimizer. It is applied after gradient clipping and before the moments. The optimizer's other settings are unchanged.

## Conditions

- LLaMA-type (66 million to 2 billion) and Qwen2MoE (500 million to 2 billion). C4 uses context 256 and 512 sequences per update (131,072 tokens). OpenWebText uses context 1,024 and 480 sequences (491,520 tokens). The amount read on C4 is about 20 tokens per parameter.
- AdamW uses β 0.9, 0.95, weight decay 0.1, clipping 1.0, 1,000 warmup steps, and ends at a tenth of the peak. The learning rate found for AdamW is used unchanged for GradPower.
- p was searched once, on LLaMA 200 million (C4), and set to 1.2; the same value is used everywhere else.
- Six repeats with different seeds appear only in some figures (Appendix B.4). The rest are single runs.

## Results (AdamW)

| Conditions | AdamW | p=1.2 |
|---|---|---|
| 66 million, C4, cosine, 10,000 steps (mean of 6 seeds) | 3.366 | 3.351 |
| 200 million, C4, cosine, 30,000 steps (mean of 6 seeds) | 3.006 | 2.988 |
| 400 million, C4, cosine, 60,000 steps | 2.794 | 2.775 |
| 1 billion, C4, cosine, 150,000 steps | 2.570 | 2.548 |
| 2 billion, C4, WSD, 300,000 steps | 2.421 | 2.403 |
| 250 million, OpenWebText, cosine, 50,000 steps | 2.834 | 2.822 |

- Final validation losses, read by us from the figures' SVGs. The text gives −0.022 at 2 billion and −0.028 for Qwen2MoE 2 billion.
- Figure 1's annotations are 0.157×FLOPs (cosine) and 0.162×FLOPs (WSD), which we read as meaning about 16% less compute for the same loss.
- At 2 billion, the mean over 6 tasks goes 47.72→48.26 (Table 1). Only ARC-C falls, 26.45→26.28.
- The comparison of p (200 million, C4): 3.120 at p=0.6, 3.050 at 0.8, 3.002 at 1.0, 2.985 at 1.2 and 3.000 at 1.4 (values read from Figure 2). At this update size, p below 1 is much worse.
- Searching 12 combinations of β1 and β2 for both AdamW and GradPower, the two bests are 2.8328 and 2.8287, a difference of about 0.15% (Appendix B.5, 250 million, OpenWebText). The difference under the same conditions in the table above is about 0.4%. However, the table's 2.822 for GradPower (default β) is lower than B.5's best, and it is not stated whether B.5 and the table's row share the same conditions.

## Combined with Muon

- Muon is in Kimi's form (weight decay, and updates scaled up to match the matrix's shape), with the same learning rate as AdamW. p=1.2, and the other settings are the same as Muon's (Appendix B.2). Whether it is applied before the momentum is not stated.
- Values read from Figure 5: 2.965→2.952 at 200 million (C4, cosine, 30,000 steps) and 2.770→2.753 at 400 million (60,000 steps). The figure shows no noise band, and the number of seeds is not stated. Each update is 131,072 tokens.

## The size of one update

- On LLaMA 200 million, C4, WSD, it varied the update from 512 to 8,192 sequences (131,072 to about 2.1 million tokens) and compared p from 0.8 to 1.4 (Appendix B.3, Figure 6). At every size, 1e-3 was the best AdamW learning rate. The losses are compared at the point where the best setting reached about 3.5.
- Values read from the figure: the difference of p=1.2 against p=1.0 is −0.024 at 512 sequences, −0.010 at 2,048 (524,288 tokens), −0.001 at 4,096 (the best is 1.1, at −0.003) and +0.003 at 8,192. At 8,192 the values go up and down, with 0.9 and 1.1 both about 0.01 lower than 1.0.
- The paper says the optimal p falls as the update grows (as the gradient noise shrinks), and can go below 1 for large updates. On images (ResNet-34, CIFAR-10), p=0.8 is best at an update of 128, 0.85 at 64 and 1.0 at 32, and p=1.2 has lower accuracy than 1.0 at all of them (Table 2).

The HTML main text and Appendix B were read (sections 4.4 and 4.6 and Appendices B.2–B.5 directly from the HTML, the rest through summaries). The figures' numbers were read from their SVGs. The proofs (Appendices C, D) were not read.
