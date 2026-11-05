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

When the improved version's code or a description of its improvements is published, measure again with the chunk boundary spacing, the training data and the main network matched to the main line.
