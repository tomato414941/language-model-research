# Stochastic activations

Lomeli ら、Stochastic activations（arXiv 2509.22358）。

- MLP の活性化を、学習中に SiLU と ReLU から確率で選ぶ。最後に ReLU で調整すると、ReLU だけで学習するより良く、推論で疎にできる。
- 目的は推論を速くすること（MLP の処理で約1.65倍）。品質は SiLU を超えるものではない。

読んだのは要旨と結果の本文の一部。
