# Polar Express

Amsel ら、The Polar Express（arXiv 2505.16932）。方法と7.74億パラメータ・10億トークンの比較は `language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/polar-express.md)にある。ここにはその続きだけを書く。

- 実験は GPT-2 の形、FineWeb、更新1回32,768トークン（32本、文脈1,024）、4 H100、bf16、反復5回。学習率は最初の40%を一定にし、あとは線形に下げる。埋め込み、出力、位置符号は AdamW で更新する。
- 1.24億パラメータ、10億トークン（1パラメータ約8トークン）、weight decay なしの検証損失：AdamW 4.197（学習率0.0005）、元の Muon 3.639（0.01）、You の方法 3.629（0.01）、Polar Express 3.588（0.005）。
- 7.74億パラメータ、100億トークン、weight decay 0.1 の検証損失：元の Muon 2.921、You の方法 2.919、Polar Express 2.913（どれも学習率0.002）。
- 反復は2〜3回だと悪く、5〜6回より増やしても、SVD で厳密に計算しても良くならない。反復の回数で実行時間はほとんど変わらないが、SVD を使うと更新の時間が2倍になる。

読んだのは要旨、導入、実験の節。証明と付録は読んでいない。
