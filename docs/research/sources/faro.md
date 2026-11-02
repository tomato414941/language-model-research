# FARO

Wu ら、FARO: Function-Space Adaptive Return-to-Risk Optimization（arXiv 2610.40858、2026-10）。

層ごとの更新の予算を学習中に配り直すオプティマイザ。11層、幅768、語彙50,257、FineWeb。

- 全部の学習で FARO-Adaptive は3.2780、手で組んだ中で一番良い NorMuonAndAdam は3.2794。
- 800ステップの比較で、等しいリスクに配る ERC を足すと3.6017から3.5881（−0.0136）。6つの種のすべてで良い。1ステップの差は約2ミリ秒で、メモリは同じ。

読んだのは要旨と、HTML の本文の前の約75%を要約を通したもの。
