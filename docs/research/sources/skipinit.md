# SkipInit

De、Smith、Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks（arXiv 2002.10444）。正規化のない残差の CNN の話で、言語モデルではない。

- 残差の出力に、学習する値 α を1つ掛ける。始めを 1/√深さ 以下にすれば深いネットワークを学習できる。深さに関係なく使えるので、実用には0を勧めている。
- Wide-ResNet、CIFAR-10 の精度は、1/√深さ と0でほぼ同じ（16層 93.0 と 93.3、100層 94.2 と 94.2、1,000層 94.2 と 94.3）。α=1 は16層より深いと学習できない。

読んだのは2節と表1。
