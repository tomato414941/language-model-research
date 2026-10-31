# nanoGPT

Karpathy の `karpathy/nanoGPT`。MIT ライセンス。`model.py` 330行、`train.py` 336行（コミット `3adf61e`）。

- 標準の形は GPT-2 small と同じ（12層、幅768、12ヘッド、文脈1,024、語彙50,304で1.24億パラメータ）。学習率6e-4、AdamW、weight decay 0.1。
- 注意の計算は PyTorch の `scaled_dot_product_attention`。位置の埋め込みは学習、活性化は GELU、LayerNorm。
- `--init_from=gpt2` で OpenAI の GPT-2 の重みを読み込める。
- トークナイザは持たず、`data/openwebtext/prepare.py` が `tiktoken` の GPT-2 のものを使う。`train.py` は `train.bin`、`val.bin`（16ビット）と `meta.pkl` を読むだけ。

読んだのは `model.py`、`train.py`、`data/openwebtext/prepare.py`。
