# Ceramic の LayerNorm の記事

Costello ら、[Revisiting LayerNorm: aka Norms are Important](https://www.ceramic.ai/blog/revisiting-layernorm)（2025-05-08）。

- 約20億パラメータで、LayerNorm、1パスの LayerNorm、RMSNorm を比べ、評価の損失はどれも3.0〜3.15。
- 素直な PyTorch では両者の速さが最大40%違うが、Triton のカーネルにするとほぼ同じになり、差はなくなったとしている。

重みの有無と、置く場所は書かれていない。データとトークン数も書かれていない。詳しい数字は別の表にあり、読んでいない。読んだのは要約ツールを通した本文。
