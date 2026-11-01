# RMSNorm の重みの比較の記事

Lucas Lingle、[Parametric or Nonparametric RMSNorm?](https://lucaslingle.substack.com/p/parametric-or-nonparametric-rmsnorm)（2025-04-30）。

- LLaMA の形、28層、幅3,584。AdamW（β 0.9、0.95、ε 1e-8、weight decay 0.1）。更新1回1,048,576トークン、文脈256、10万回の更新。学習率はそれぞれ探した。
- 一番低い損失は、重みあり 2.240、重みなし 2.253。
- LayerNorm とは比べていない。種の数は書かれていない。データも書かれていない。
- 著者は自分の以前の論文（mu-transfer.md）とは逆の結果と認めていて、重みありが合意の選択だとしている。

読んだのは要約ツールを通した本文。
