# OLMoE

Muennighoff ら、OLMoE: Open Mixture-of-Experts Language Models（arXiv 2409.02060）。動くパラメータ10億、全体70億の MoE。

- OLMo（arXiv 2402.00838）は、一番安全で速いという理由で、学習する重みのない LayerNorm を使った。
- OLMoE では、これを学習する重みのある RMSNorm に替えると良くなった（図14）。重みのない形は勾配の急な跳ねが多く（図16）、勾配のクリップで真の勾配でなくなるのが理由と見ている。RMSNorm で処理量は15%落ちるが、最終モデルに採用した。
- RMSNorm の重みに weight decay をかけるかどうかの差は小さい。
- QK-norm は安定と性能を少し良くした（図18）。RMSNorm の形でも、学習の損失が少し下がり、勾配の大きな跳ねを防いだ。処理量は約10%落ちるが採用した。

読んだのは4.2.3と QK-norm の節の本文。図の数字は読んでいない。最適化は AdamW。
