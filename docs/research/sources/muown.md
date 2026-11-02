# Muown

Lion ら、Muown: Row-Norm Control for Muon Optimization（arXiv 2605.10797、2026-05）。

行の大きさを別の変数にして Adam で更新し、向きを Muon で更新する。順伝播は変わらない。増えるメモリは層ごとに m 次元のベクトル4つ（報告の実行で約2MB）で、更新の時間は最大1.5%増える。既定で weight decay を使わない。

FineWeb-Edu の perplexity。

| 大きさ | Muown（weight decay 0） | Muon（weight decay 0） |
|---|---|---|
| 10億 | 11.90（η=3e-3） | 12.20（η=2e-3）。η=3e-3 は 110.19 |
| 27億 | 9.57（η=2e-3） | 9.82（η=1e-3）。η=2e-3 は発散で中止 |

対数で約1%の差（自分たちの計算）。学習率を上げても発散しにくい。5億の表は列の対応がはっきり読めなかった。1.24億は図だけで、数字はない。

比べた相手は Muon、SOAP、AdamW、Lion で、読んだ範囲に NorMuon はない。

読んだのは要旨と、HTML の本文の前の約7割を要約を通したもの。
