# COUNT

Magnusson, Smith, Koh, The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation (arXiv 2610.20456, 2026-10). The implementation, tables and weights are released at `IanMagnusson/COUNT`. The full results are in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/count-ngram.md).

## How it works

- A frozen table holds, for each history, the top K next tokens and their scores. A history of order n is n−1 tokens, and order 1 has no history. Histories do not include document boundaries.
- The score is stupid backoff: a seen continuation scores `C(h,w)/C(h)`; otherwise it backs off to a shorter history and is multiplied by 0.4. The scores are not normalized.
- The input is `f_t = (1/J) Σ_j Σ_k S(w_k | h_{t,j}) B[w_k]`. B is the model's own output embedding table, and gradients also flow into B. `LayerNorm(e_t + f_t)·√d` goes into the first layer.
- The table stores K=32; training uses K=8 and J=6 (orders 1 to 6).

## Table size

Built from Dolma 3 (about 6 trillion tokens). Sizes (GB) at K=32 for different minimum counts of the history and of the n-gram.

| Minimum history count | Minimum n-gram count | Size |
|---|---|---|
| 12,319 | 27 | 17.114 (used at 60 million to 190 million) |
| 3,255 | 7 | 68.458 |
| 1,536 | 7 | 142.037 |
| 460 | 7 | 459.222 |
| 265 | 7 | 797.487 (used at 370 million to 1 billion) |

- Most of it is orders 4 to 6. With K at 8 it shrinks to about 30%, and with K at 4 to about 20% (read off Figure 7).
- The domain table, from 10.66 billion tokens with a minimum count of 4 and K=32, is 83.4GB. It is swapped in at inference; no training was done with this table.
- It counts the whole corpus that training uses (§3.1). The amount read is about 600 million tokens at 60 million and D/N=10, and 40 billion tokens at 1 billion and D/N=40, so the text behind the table is about 10,000× to 150× the amount read (our own calculation). The rows are histories that appeared at least several hundred times (12,319 times for the 17GB table), so the text being read makes up a small share of the counts.

## Settings close to our scale

The 60 million model has 8 layers, width 384, 8 heads and an MLP of 1,536. AdamW (learning rate `0.00235(N/108M)^(-1/3)`, β2 0.99, weight decay 0.1), 262,144 tokens per update, context 8,192, vocabulary 100,278. N is the parameter count excluding the input embeddings and including the output embeddings, and D/N is the ratio of the tokens read to N.

Read off Figure 2 (mean accuracy over 41 multiple-choice tasks, and correct-answer BPB over 74 tasks).

| Setting | Compute | Accuracy: plain → COUNT | Correct-answer BPB: plain → COUNT |
|---|---|---|---|
| 60 million, D/N=10 | about 2×10¹⁷ | 29.7 → 31.3 | 2.05 → 1.91 |
| 60 million, D/N=20 | about 4×10¹⁷ | 30.4 → 31.3 | 1.91 → 1.83 |
| 60 million, D/N=40 | about 8×10¹⁷ | 31.1 → 31.4 | 1.80 → 1.74 |
| 100 million, D/N=10 | about 6×10¹⁷ | 30.6 → 31.4 | |
| 190 million, D/N=10 | about 2×10¹⁸ | 31.3 → 32.3 | 1.66 → 1.63 |
| 190 million, D/N=40 | about 8×10¹⁸ | 32.9 → 32.2 | 1.51 → 1.47 |

The gap is larger at lower compute. Accuracy is noisy from point to point, and some points went down.

## Other results

- Mixing into the output (late fusion) has a smaller effect than COUNT, and putting it into pretraining is no different from using it only at inference (Figures 9 and 11). Late fusion uses only the longest matching history, because the model cannot later correct the coarse predictions of the low orders. In the inference-only comparison, the additive form is fixed at λ=0.1 and the multiplicative form at α=0.1 (B.1). Read off Figure 9 at 60 million and D/N=10, the validation loss is 4.06 plain, 4.04 multiplicative, 3.96 additive and 3.82 COUNT, and the task loss is 2.05, 2.00, 1.99 and 1.91. The additive form gets about 40% of COUNT's improvement. The gap shrinks at larger scale.
- Orders 1 to 6 give a lower loss than orders 1 to 3 at every budget at 190 million.
- As the basis, the output embeddings work better than the input embeddings (1 billion). With the input embeddings the loss also falls, but the task loss and accuracy are worse than plain.
- LayerNorm and the √d scaling lower the loss of COUNT at 190 million. In the plain model they help a little at small scale and hurt at larger scale.
- At 1 billion, the model stays better than the plain one even with the table shrunk by more than an order of magnitude. Changing K and J far from their training values loses most of the effect.
- Speed at 1 billion on eight H200s is 93.1–97.8% of plain. GPU memory barely grows. The techniques are lookup in a separate thread, prefetching, and summing without storing the candidates.
- A trainable table at 190 million runs at 4.4% of the speed when the whole table is updated, and at 96.3% when only the touched rows are.
- The authors also say that for rare high-order contexts the table can leak fragments of the source text.

The main text and appendix were read, and Figures 1, 2 and 7 were read off the plots. The numbers were read by eye, so they carry some error. In the released code, the README, the embedding part and the table format were read.
