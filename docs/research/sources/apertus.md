# Apertus

Apertus v1 Technical Report (arXiv 2509.14233). The models are released at 8B and 70B, and the 70B trained on 15 trillion tokens. Two of xIELU's authors are also among its authors.

The architecture and training method are GQA, RoPE, RMSNorm, QK-norm, xIELU, Pre-Norm, AdEMAMix, a WSD learning rate and Goldfish loss. It says QK-norm was added to stabilize training by keeping the attention logits from growing too large.

**Comparison item by item (Table 3):** 1.5 billion parameters, 100 billion tokens (about 67 tokens per parameter), the Llama architecture as the baseline; training loss.

| Change | Loss |
|---|---|
| Baseline | 2.037 |
| Cut attention between documents | 2.037 |
| Cosine to WSD, learning rate halved | 2.033 |
| AdamW to AdEMAMix | 2.002 |
| SwiGLU to xIELU (inner width from 8,192 to 12,288) | 1.997 |

- SwiGLU to xIELU: −2.0% (our own calculation). One run, and not the validation loss.
- At 3 billion with everything in, the loss goes from 1.906 to 1.843. The table has no row for QK-norm alone. It says stability and gradient magnitudes also improve, and the same loss is reached with 30–40% fewer tokens.

The sections on the model's architecture and training method and on the item-by-item comparison were read. The data, the evaluation and the other sections were not read.
