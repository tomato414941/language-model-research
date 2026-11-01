# 学習の安定の比較

Methods of improving LLM training stability（arXiv 2410.16682）。

- 8.3億パラメータで学習率を上げて発散させ、方法を比べた。
- QK-norm と logit の softcap が基準の中で一番安定し、学習率60e-3で発散する。σReparam と LayerScale は40e-3で発散する。
- QK-norm と softcap を組み合わせる形と、QKV のあとの正規化は、学習率をさらに1.5倍にできる。

比べているのは発散の境目で、同じコストでの損失の差ではない。読んだのは要旨と結果の本文。
