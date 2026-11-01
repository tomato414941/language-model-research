# AdamW

Loshchilov と Hutter、Decoupled Weight Decay Regularization（arXiv 1711.05101、ICLR 2019）。

- Adam の weight decay を、勾配の更新から切り離す修正。SGD では L2 正則化と weight decay は同じだが、Adam では違う。L2 だと、過去の勾配が大きい重みほど弱くしか効かない。
- CIFAR-10（ResNet、1,160万と2,560万パラメータ）と ImageNet32x32 で、学習率0.001のとき、テスト誤差が Adam より相対で15%下がった。
- 学習率と weight decay の最良の値が、互いに独立に近くなる。
- Adam にも学習率のスケジュールは効き、cosine annealing が固定や段階的な減衰より良かった。最良の weight decay は、更新回数が多いほど小さい。
- 実験は画像分類で、言語モデルで他の手法と比べたものはない。利用例として Transformer（Radford ら）を挙げている。著者は、他の課題でも確かめる必要があると書いている。

読んだのは本文。付録は読んでいない。
