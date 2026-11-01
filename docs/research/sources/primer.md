# Primer

So ら、Primer: Searching for Efficient Transformers for Language Modeling（arXiv 2109.08668）。ReLU² の出どころ。

- 決まった学習時間での perplexity を目的にして、モデルの形を進化で探した。一番効いた変更が、MLP の ReLU を ReLU² にすることで、次が Q、K、V の射影のあとの depthwise 畳み込み（MDHA）。この2つだけのものを Primer-EZ と呼ぶ。
- ReLU² は x が大きいと GELU や Swish と形が大きく違う。ReGLU の2つの行列が同じ場合と等しいので、GLU の利点を、重みを増やさずに持つと論じている。

T5、C4、1.1億パラメータ、525K 回の perplexity（図5の右と表5）。

| 活性化関数 | perplexity | 損失の差（自分たちの計算） |
|---|---|---|
| ReLU | 20.61 | |
| GELU | 20.34 | ReLU から −0.44% |
| Swish | 約20.2（図） | |
| ReGLU | 約20.2（図） | |
| SwiGLU | 約20.0（図） | |
| ReLU² | 約19.9（図） | GELU から約 −0.7% |

表には Synthesizer に ReLU² を足した比較もあり、1.45億で 20.35 から 19.55（損失で −1.3%）。

- 約3,500万の LM1B での挿入と除去（図26）では、ReLU² は T5 では大きく、T2T ではごく小さいが、どちらも良い向き。縦軸は正規化されていて、数字は読めない。
- 同じ図で、MLP を12倍に広げる変更（d_model を384に、d_ff を4,608に）は両方で良い。本文は、約3,500万では大きく効くが大きいモデルでは効きにくいとして、Primer と Primer-EZ からは外している。
- 学習法は Adafactor で、ハイパーパラメータは Transformer に合わせたまま。文脈は LM1B で64、C4 と PG19 で512。

図5の値は棒の高さから読んだもので、GELU と ReLU の値は表と合う。読んだのは本文の該当部分、図4、5、6、26、表4、5の一部。
