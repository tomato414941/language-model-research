# BTLM-3B-8K

Dey ら、BTLM-3B-8K: 7B Parameter Performance in a 3B Parameter Model（arXiv 2309.11568）。

活性化関数については、GELU を SwiGLU にして損失が1.37%下がった。計算を揃えるために、中の幅を4倍から8/3倍にしている。モデルは Cerebras-GPT の μP の設定で、1パラメータ20トークン。

ReLU² は比べていない。読んだのは活性化関数と位置の節だけ。
