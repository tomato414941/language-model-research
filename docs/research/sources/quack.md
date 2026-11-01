# QuacK

Controlling changes to attention logits（arXiv 2511.21377）。

- Q と K の重みが大きくなるのが不安定の原因とし、QK-norm の代わりに、重みの大きさに応じて Q と K の学習率を下げる。
- 普通の注意（MHA）では、QK-norm が全体で一番良く、QuacK はわずかに下。学習率が小さい（3e-4）と、どの方法も差がほとんどない。何もしないと学習率3e-3で不安定だった。
- QK-norm の2回の RMSNorm をなくすと、学習が約10%速くなった。
- Multi Latent Attention には QK-norm を使いにくいので、QuacK が一番良い。

実験は AdamW 系の学習率で、Muon のように更新の大きさを揃える手法での比較ではない（自分たちの読み）。読んだのは要旨、導入、結果の本文で、図の数字は読んでいない。モデルの大きさは確かめていない。
