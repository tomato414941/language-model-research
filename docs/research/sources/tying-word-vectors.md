# Tying Word Vectors and Word Classifiers

Inan ら、Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling（arXiv 1611.01462、ICLR 2017）。LSTM の言語モデル。入力の表を出力にも使う形を、損失の枠組みから導いた。

共有だけを足した比較（テストの perplexity）：

| モデル | PTB | Wikitext-2 |
|---|---|---|
| 200ユニット | 87.3 から 85.1 | 105.9 から 100.5 |
| 650ユニット | 77.7 から 73.9 | 95.3 から 87.7 |

どちらも小さいデータで、共有で変数が大きく減る条件。

読んだのは要旨、導入、表1。
