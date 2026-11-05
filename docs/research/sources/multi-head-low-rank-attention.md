# Multi-Head Low-Rank Attention

Multi-Head Low-Rank Attention (arXiv 2603.02188). A paper on a method that shrinks attention's KV cache. In a comparison that comes before its main topic (4.2.1, Table 38), it compares initial values for the output weights of the attention and the FFN.

A model of about 2.9 billion with the Llama 3 architecture built on top of nanoGPT (24 layers and width 3,072 with standard attention), FineWeb-Edu, 98.3 billion tokens (about 34 tokens per parameter), AdamW, learning rate 1.6e-4. All other weights are `N(0, 0.02)`.

Mean perplexity over seven datasets. The loss differences are our own calculation.

| Attention | Output at 0.02 | Output at 0 | Loss difference |
|---|---|---|---|
| Standard attention (MHA) | 14.094 | 13.860 | −0.63% |
| MQA | 14.283 | 14.306 | +0.06% |
| GQA | 14.164 | 14.139 | −0.07% |
| MLA | 13.927 | 13.727 | −0.55% |
| MFA | 14.374 | 14.315 | −0.15% |
| TPA | 14.478 | 13.985 | −1.30% |
| GLA-2 | 14.009 | 13.779 | −0.63% |
| GLA-4 | 14.140 | 13.815 | −0.88% |
| GTA | 14.285 | 13.972 | −0.83% |

- For standard attention, 0 is better on all seven datasets, and on FineWeb-Edu it goes from 9.571 to 9.434 (−0.64%).
- The authors judge 0 to be better and use 0 in the later experiments.
- The 0.02 it is compared with has no scaling by the number of layers as in GPT-2. Scaled for 24 layers it would be 0.0029, so the comparison point is about 7× larger. There is no comparison between the scaled value and 0.
- The number of seeds is not stated.

The experimental settings, the text of 4.2, and Tables 18 and 38 in the appendix were read. The main method was not read.
