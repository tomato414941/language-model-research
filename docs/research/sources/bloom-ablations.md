# BLOOM の形の比較

Le Scao ら、What Language Model to Train if You Have One Million GPU Hours?（arXiv 2210.15424）。BLOOM の形を決めるための比較。

活性化関数については、The Pile を1,120億トークン学習したモデルの EAI の結果の平均で、GELU 42.79、SwiGLU 42.95。SwiGLU が少し良い。

BLOOM 自身は GELU にした。SwiGLU の遅さに見えたものは、中の幅5,456が GPU に合わないためと後から分かったので、次のモデルには SwiGLU を勧めている。

ReLU² は比べていない。読んだのは活性化関数の節だけ。
