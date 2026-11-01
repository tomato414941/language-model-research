# ReLU² Wins

Zhang ら、ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs（arXiv 2402.03804）。

- 目的は推論で使わないニューロンを飛ばす疎な計算で、学習の効率の話ではない。
- 重み、データ、ハイパーパラメータを揃えた13億のモデルで SwiGLU、ReLU、ReGLU、ReLU² を1,000億トークンまで学習した。最後は SwiGLU、ReGLU、ReLU² がほぼ同じで、ReLU はずっと悪い。
- ReLU² は途中では GLU の形に少し遅れ、終わりに近づいて追いつく。
- 疎にしたときの性能の落ちは ReLU² が一番小さい（疎の割合が約90%で0.1%未満）。ハードウェアとの相性も ReLU と ReLU² が SwiGLU と ReGLU より良いとしている。

GELU は比べていない。読んだのは要旨、導入、モデルを学習した節で、損失の比較は図にしかなく、数字は読んでいない。
