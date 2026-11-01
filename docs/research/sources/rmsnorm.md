# RMSNorm の論文

Zhang、Sennrich、Root Mean Square Layer Normalization（arXiv 1910.07467、NeurIPS 2019）。

- LayerNorm から平均を引く計算をなくし、2乗平均だけで割る。論文の形は学習する重みを持つ。
- いくつかのモデルで、品質は LayerNorm と同じくらいで、時間は7〜64%短いとしている。
- Transformer（幅512、機械翻訳、V100、TensorFlow）の表。

| 正規化 | BLEU（Test14、Test17） | 1,000ステップの時間 |
|---|---|---|
| LayerNorm | 26.6、27.7 | 248秒 |
| RMSNorm | 26.8、27.7 | 231秒（−6.9%） |

言語モデルの事前学習の比較はない。読んだのは要旨、導入、機械翻訳の節の表。
