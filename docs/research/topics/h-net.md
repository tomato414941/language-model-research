# H-Net (reads bytes and learns its chunks)

Reads UTF-8 bytes without a tokenizer. Where to form chunks is learned together with the model, and the main network runs on the chunks. The numbers from the paper and from Cartesia's post are in `language-model`'s [h-net.md](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/h-net.md), and the paper's mechanism and settings are in [the source note](../sources/h-net.md). What follows is the view as of 2026-11.

## Why try it

There are no public numbers for Japanese. In Japanese one character is 3 bytes, and a BPE with a small vocabulary cannot merge them well. In the paper, the gap for Chinese was larger than for English.

However, research's training data is English FineWeb (from 2026-11-04) and, from 2026-11-05, FineMath and Common Pile's code ([training data](training-data.md)), so comparisons in research can measure only English. The effect in Japanese has to be measured separately, on data that includes Japanese.

## The outer layers

The paper made the encoder and decoder Mamba-2, and the all-Transformer form was the worst. We will also try Mamba-3 and Gated DeltaNet. Both keep the recurrence that selectively forgets; Gated DeltaNet is said to make memory precise by rewriting the state (the delta rule), and Mamba-3 to fix counting by putting rotation into the state. We guess this helps the outer layers' job of grouping bytes into words and spelling words out one byte at a time. We have not found comparisons in H-Net's outer layers. The main text of the two papers has not been read yet.

## Expectations when compared at the same cost

We think it loses at research's scale (a main network of 54.4 million parameters, 30 minutes of training).

In the 2026-11-04 measurement, the H-Net implementation here (Mamba-2) was 4.7× slower per update than the baseline, and made only 178 updates at the same cost. Its validation BPB was 2.431, twice that of the second bundle on the same data (1.2146). It lost by the reading decided in advance ([comparison at the same cost](../experiments/h-net-comparison.md)). Without a fast implementation, a comparison at this scale is decided almost entirely by the difference in speed.

On 2026-11-05 the implementation was fixed, and an update on the H100 went from 3.917 seconds to 2.583–2.651 seconds (same machine, a chunk boundary every 6 bytes). Mamba-2 was run on the fused path (the convolution in Triton), waiting for the GPU was cut to once per 32 rows, and the main network was compiled into a single graph. It was still about 4× the second bundle (0.616 seconds), and much of the time went to the outer Mamba-2 ([breakdown of the speed](../experiments/h-net-comparison.md)). So the outer width was cut from 512 to 256, and the ratio loss's weight was made to rise while the chunk boundaries are finer than the 6-byte target. The slow backward pass of the convolution was fixed too. Trained for 30 minutes under the frame on the same day and measured once, an update took 1.55 seconds, and the chunk boundaries stayed about 5.5 bytes apart. Even so, it lost to the current best (research-v4) on all three OLMES BPBs and on validation BPB. QA was 27% higher, validation BPB 20% higher. Math and Code were 2.5–3×, but research-v4 also read FineMath and code while H-Net read only FineWeb, so we think much of that gap comes from the training data ([the one run under the frame](../experiments/h-net-comparison.md)). Per amount read, H-Net is about 1.5× slower than research-v4. Outer layers of Mamba-3 and Gated DeltaNet were not measured.

- The paper's wins are in comparisons that match compute, not time. Even with the authors' implementation, training is up to 2× slower.
- In the paper's XL, it overtakes BPE only after reading about 100 billion bytes with one stage and about 30 billion bytes with two stages. Until then every H-Net is worse than BPE. Training here reads only 2–3 billion bytes.
- The 2026-11-05 run, against the second bundle, had about the same amount read (2.2 billion and 2.5 billion bytes) and compute per byte (approximate), and its validation BPB was 12% higher. This agrees with the paper's view that it loses before it overtakes.
- Because the main network is small, the outer layers take a large share of the compute. With a width of 256, the ratio of the outer width to the main network's width (640) is smaller than in the paper's one-stage L (1,024 and 1,536).

Losing here does not mean it fails at production scale. v6's training reads about 30 billion bytes, around where the overtaking happens in the paper.

## Current status: on hold

Not adopted, but not dropped either. We think it lost because the H-Net here has not caught up yet. It has yet to catch up in three things:

- Improvements since the paper. Cartesia's post ([From Tokens to Bytes](https://cartesia.ai/blog/from-tokens-to-bytes-part-1), 2026-10-28) compares an improved H-Net (unpublished) that replaces only the tokenizer part of a BPE Transformer, and says its loss was lower at every size. The smallest tier in its table is 8 layers, width 640 and a main network of 56.4 million parameters, about the same size as the main network here. The comparison was by compute and amount of data; time is not given. The improvements are not described either, but the post says the paper's H-Net places a chunk boundary only after seeing the first byte of the next chunk, and that this weakened next-byte prediction and the stacking of stages. With the paper's H-Net as is, given where the overtaking happens above, its chances of winning in this frame are slim.
- Speed. Per amount read, it is about 1.5× slower than research-v4 (2026-11-05). The frame matches time, so the slower it is, the less it reads.
- The comparison. The post uses a strong auxiliary loss to space the chunk boundaries like BPE tokens (4.57 bytes), and matches the outer layers to the embeddings and the output layer in compute and parameters. The 2026-11-05 run had a chunk boundary every 5.5 bytes, 1.4× coarser than FineWeb's tokens (3.81 bytes). Its training data also differed from the main line (no FineMath or code), and the main network had none of the main line's changes.

Resume when the improved version's code or a description of its improvements is published, or when work here closes one of the gaps without it: outer layers fast enough that H-Net reads about as much per second as BPE, or a change in how chunks are formed, such as ATDC's or ByteFlow Net's below, that removes the 12% deficit at the same amount read and compute. The main line's progress alone is unlikely to be the trigger: changes that H-Net's main network can also take (the optimizer, attention) shrink its lag behind the current best but improve BPE as much, and changes that work on tokens (tables of token pairs and triples, value embeddings) help only BPE. On resuming, measure with the training data and the main network matched to the main line, and with chunk boundaries spaced both like BPE tokens and more coarsely (see Compute Optimal Tokenization below). Before that run, h-net/evaluate_hnet.py and h-net/run.sh need porting to frame.py's evaluation of 2026-11-05 (0258b47), which scores the correct answers with `frame.task_scores` and `frame.suite_scores` instead of running olmes.

## Follow-ups and related work (searched 2026-11-05)

- The paper's latest arXiv version is still v2 (2025-07-15), and it was accepted at ICLR 2026; whether the accepted version changed was not checked. No public code or weights for Cartesia's improved H-Net were found.
- [ATDC](https://arxiv.org/abs/2605.30080) (2026-05) raises the compression ratio from low to high over training as a curriculum, and reports steadier training and better downstream scores than with a fixed ratio, on FineWeb-Edu 100B. It targets what the 2026-11-05 run showed: chunk boundaries swinging over the first 100 or so updates, and the ratio loss's weight held at its cap.
- [ByteFlow Net](https://arxiv.org/abs/2603.03583) (2026-03) picks chunk boundaries by the coding rate of latent representations with a top-K selection, so its computation graph keeps a static shape, and reports beating BPE Transformers and earlier byte-level models. A static shape may make it easier to compile, and so faster.
- [H-Net++](https://arxiv.org/abs/2508.05628) (2025-08) extends H-Net for Persian. [Bolmo](https://allenai.org/blog/bolmo) (Ai2, 2025-12) turns existing OLMo models into byte-level models; it starts from another model's weights, which the frame does not allow.
- A [720-million-parameter H-Net](https://huggingface.co/jfrick/hnet-720m) trained by an individual, at the same compute per byte as a token model, was 2.7–4.5% worse in held-out BPB after 23.8 GB and took 2.2× the time per byte, its byte-level layers running at about 10% MFU. Both models were still at a constant learning rate.
- A byte-level H-Net of 17.5 million parameters submitted to [Parameter Golf](https://sameersegal.github.io/learn-parameter-golf/pr/1104) beat the best 10-minute run of its subword version only when trained for 4 hours.
- [The Efficiency Gap in Byte Modeling](https://arxiv.org/abs/2605.12928) (2026-05) found that byte-level Transformers without chunks need about 7.9× BPE's compute to reach BPB 1.0 and about 2.3× to reach BPB 0.8, with parity projected near 10^22 FLOPs.
- [Compute Optimal Tokenization](https://arxiv.org/abs/2605.01188) (2026-05; 988 BLT models of 50 million to 7 billion parameters) found that the compute-optimal compression rate differs from BPE's 4.57 bytes per token and falls as compute grows. At the small compute here, chunks coarser than BPE tokens may be best, which runs against the suspicion that the 2026-11-05 run's 5.5 bytes were too coarse.
