# Dispersion Loss

Liu ら、Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models（arXiv 2602.00217、ICML 2026）。

小さいモデルでは、トークンの向きが狭い範囲に集まる（GPT-2、Qwen3-0.6B）。大きいモデルでは起きにくい。大きいモデルからの蒸留では直らないとしている。

損失 `log Σ_{i≠j} exp(−arccos(cos(z_i, z_j)) / (π τ))` を重み0.1、τ=1で足す。

| モデル | 通常 | 分散の損失あり |
|---|---|---|
| GPT-2 | 34.95 ± 0.11 | 35.61 ± 0.12 |
| GPT-2-m | 38.55 ± 0.18 | 39.35 ± 0.15 |
| Qwen3-0.6B | 45.93 ± 0.37 | 47.45 ± 0.16 |
| Qwen3-1.7B | 54.57 ± 0.17 | 55.68 ± 0.21 |

上は学習済みのモデルの続きを2億トークン読んだ結果（wikitext-103、3つの種）で、10課題の平均。ゼロからの学習は Qwen3-0.6B を1,560億トークン読み、34.84→36.01（PIQA +4.0、TruthfulQA +7.4）。時間は1.03〜1.05倍。perplexity は報告していない。

読んだのは HTML の本文の前の約8割を要約を通したもの。
