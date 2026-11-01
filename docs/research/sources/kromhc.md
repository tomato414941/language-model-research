# KromHC

KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices（arXiv 2601.21579）。残差の流れを複数に分けて混ぜる mHC の、パラメータを減らした形。

nanochat の残差を置き換えて比べている（行列は Muon、残差の流れ、埋め込み、出力は AdamW）。元の nanochat の形は x0 と resid の重みを持ち、mHC の形ではこれを置き換える（付録 G）。FineWeb-Edu、1パラメータ約20トークン、流れは4本。

| 形 | 増えるパラメータ（6層） | 6層（約6,000万）の検証 BPB | 12層（約1.86億）の検証 BPB |
|---|---|---|---|
| 元の残差 | | 1.047 | 0.864 |
| mHC | 46万 | 1.042 | 0.861 |
| mHC-lite | 61万 | 1.045 | 0.864 |
| KromHC | 24万 | 1.047 | 0.862 |

CORE スコアは 6.477、7.971、8.208、9.018（6層）、14.774、16.023、13.217、16.872（12層）で、BPB より差が大きい。計算量と時間の比較は読んでいない。

読んだのは要旨、4節の結果、付録 G と H の始め。
