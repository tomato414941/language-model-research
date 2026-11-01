# RoFormer

Su ら、RoFormer: Enhanced Transformer with Rotary Position Embedding（arXiv 2104.09864）。

- query と key の次元の組を、位置に比例する角度で回す。スコアは相対的な距離だけで決まり、学習するパラメータはない。
- 機械翻訳（WMT）の BLEU は27.3から27.5。GLUE は3勝3敗で、QQP 71.2から86.4、MNLI 84.6から80.2。中国語の長文は BERT-512 67.77、RoFormer-512 68.29、RoFormer-1024 69.79。
- GPT の形で、学習する位置埋め込みと直接比べた結果はない。ばらつきと費用も示していない。著者は、なぜ効くかは説明できないとしている。

読んだのは本文と実験の表。
