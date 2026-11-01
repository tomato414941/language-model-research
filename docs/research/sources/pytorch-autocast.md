# PyTorch の自動の16ビット計算

PyTorch のコードに書いてあること。

- GPU の autocast で32ビットにする一覧（`AT_FORALL_FP32`）に、`pow`、`layer_norm`、`rms_norm` がある。16ビットにする一覧には `scaled_dot_product_attention` などがある。
- `square` は `at::pow(self, 2)` を呼ぶ。
- `rms_norm` は中で32ビットに上げて計算し、入力と同じ型で返す。

CPU の autocast は一覧が別で、torch 2.14.1 で試すと、16ビットを入れた `rms_norm` と `square` は16ビットのまま。32ビットの0次元の値と16ビットの配列の積は16ビットになる。GPU の autocast の動きは試していない。

読んだのは torch 2.14.1 の配布物の `autocast_mode.h` の一覧と、v2.9.0 の `UnaryOps.cpp` の `square`、`layer_norm.cpp` の `rms_norm`。
