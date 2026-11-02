# T-Muon

Li ら、T-MUON: Tensorizing Muon via t-SVD for Large Language Model Pre-training（arXiv 2610.24938、2026-10）。

重みをテンソルと見て直交化する。単位変換でヘッドごとに分ける形（Head-II）が一番安定して良かったとしている。

| 条件 | AdamW | Muon | Head-II |
|---|---|---|---|
| GPT 5,000万、FineWeb 10億トークン | 47.353 | 42.235 | 40.568 |
| LLaMA 6,000万、FineWeb 10億トークン | 32.166 | 26.558 | 25.948 |
| LLaMA 6,000万、FineWeb-Edu 12億トークン（種1つ） | 23.89 | 20.58 | 20.23 |

数字は検証の perplexity。Muon との差は loss で0.6〜1.1%（自分たちの計算）。Muon の最後の損失に届くステップは1.14〜1.19倍速い。最良の分け方は条件で変わるとしている。

読んだ範囲に、更新の時間と NorMuon との比較はない。読んだのは要旨と、HTML の本文の前の約9割を要約を通したもの。
