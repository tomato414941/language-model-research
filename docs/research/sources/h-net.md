# H-Net

Hwang, Wang and Gu, Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (arXiv 2507.07955). Its abstract and main numbers are in `language-model`'s [h-net.md](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/h-net.md); this note records what was read beyond that.

## How it works

- The chunk boundary probability is p = (1 − cos) / 2, from the cosine between the vectors of two adjacent bytes, each projected by its own matrix (initialized to the identity). The start of a row is always a chunk boundary. A chunk starts wherever p exceeds 0.5.
- The main network's output is smoothed with the chunk boundary probabilities before it goes back to bytes (z_t = p_t ẑ_t + (1 − p_t) z_{t−1}). Without the smoothing, the compression ratio fluctuated widely and the scores also fell sharply.
- The confidence of each chunk boundary decision (the probability of the chosen side) is multiplied in as 1 in the forward pass, with the gradient passed straight through in the backward pass (STE).
- The ratio loss is L = N/(N−1)·((N−1)FG + (1−F)(1−G)). F is the fraction of positions chosen as chunk boundaries and G is the mean probability. Its weight α = 0.03 was fixed in every experiment.
- As training techniques, it adds a normalization after each network, residual projections that start at 0, and higher learning rates further toward the outer layers.

## Experimental settings

- The one-stage H-Net targets N = 6 for its chunk boundaries, and the actual chunks averaged 4.7–4.8 bytes (4.6 with GPT-2's tokenizer).
- The byte models use sequences of 8,192 bytes, and the BPE Transformer sequences of 1,792 tokens. The batch is 256 sequences. AdamW and WSD (10% warmup, 20% inverse-square-root decay). The data is 100 billion tokens' worth of FineWeb-Edu.
- The one-stage L is "M4 + T22 + M4" (the outer layers are four Mamba-2 layers on each side at width 1,024, and the main network is 22 Transformer layers at width 1,536).

## Comparing the outer layers

In the one-stage H-Net, the encoder and decoder were set to M4-M4, M2T1-T1M2, T1M2-M2T1 and T2-T2 (M is Mamba-2 and T is Transformer, with T using a window of 1,024) and compared over 36 billion tokens. T2-T2 used more compute yet was "by far the worst": it chunked the bytes poorly, and the main network's compute grew as well. The larger the share of Mamba, the better, and M4-M4 was the best. Even with BPE tokens as the input, Mamba outer layers were better. The authors think this is because the SSM's property of compressing into a fixed-size state suits the job of chunking.

## Speed and scaling

- It says the current implementation trains up to about 2× slower than standard models.
- The two-stage H-Net overtook the BPE Transformer's perplexity after reading about 30 billion bytes in training, and the gap widened after that.

The abstract, Section 2's mechanism and training techniques, Section 3's table of settings, the text of Section 3.3's comparisons, and the section on inference and efficiency were read. The numbers in the figures were not read.
