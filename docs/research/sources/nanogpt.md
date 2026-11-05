# nanoGPT

Karpathy's `karpathy/nanoGPT`. MIT license. `model.py` is 330 lines and `train.py` 336 lines (commit `3adf61e`).

- The default architecture is the same as GPT-2 small (12 layers, width 768, 12 heads, context 1,024, vocabulary 50,304, 124 million parameters). Learning rate 6e-4, AdamW, weight decay 0.1.
- Attention is computed with PyTorch's `scaled_dot_product_attention`. The position embeddings are learned, the activation is GELU, and the normalization is LayerNorm.
- `--init_from=gpt2` can load OpenAI's GPT-2 weights.
- It has no tokenizer of its own; `data/openwebtext/prepare.py` uses `tiktoken`'s GPT-2 tokenizer. `train.py` only reads `train.bin` and `val.bin` (16-bit) and `meta.pkl`.

`model.py`, `train.py` and `data/openwebtext/prepare.py` were read.
