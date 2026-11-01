# modded-nanogpt の lambda

snimu、modded-nanogpt の学習された lambda を見た記事（snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html）。medium の部門の1回の学習の値で、あるなしの比較はない。

- `x = x_lambda * x + x0_lambda * x0` の x0_lambda は、3層を除くどの層でも大きく、多くの層で x_lambda より大きい。8、14、15層では学習の途中で0を横切るが、検証の損失は跳ねない。
- 記事のはじめの版は2つの値のラベルが逆で、翌日に訂正している。
- x0 の働き（比較の材料になる、など）は著者の仮説。

読んだのは取得の道具で抜き出した要約で、記事の図は見ていない。
