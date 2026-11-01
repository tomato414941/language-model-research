# Optimal Embedding Learning Rate and Vocabulary Size

Hayou、Liu、Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size（arXiv 2506.15025）。

- μP は語彙の大きさを固定して考えるが、実際は語彙が幅より大きい。語彙も大きくなる場合、埋め込みの学習率は隠れ層の √幅 倍が良いと導いている。
- 小さい実験（2層、Wikitext2、語彙32,768、幅256〜4,096、Adam、1万回）で、一番良い埋め込みの学習率は幅とともに下がる。ばらつきは大きい。隠れ層と出力は同じ学習率と初期値が良い。
- 10億のモデル（24層、幅2,048、RoPE、SwiGLU）を1.75兆トークンで学習し、同じ学習率と √幅 倍（約45倍）を比べた。√幅 倍は最初から最後まで速く、最後の学習の損失も Wikitext のテストも良い。それより大きくしても差はわずか。

読んだのは要旨、導入、3節の実験の部分、4節。図の数字は読んでいない。
