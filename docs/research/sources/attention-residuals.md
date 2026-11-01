# Attention Residuals の規模

Kimi Team（arXiv 2603.15031）。手法は `language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/attention-residuals.md)にあり、ここにはそこにないことだけを書く。

- 規模の表の一番小さい行は1.94億パラメータ、387億トークン（1パラメータ約200トークン）で、検証の損失は基準 1.931、一番良い形で 1.899。
- 一方、Marin の小さい規模の結果では1.06〜1.12倍分（breakthrough-survey.md の読書メモから写した数字）。

読んだのは規模の表の行と、メモリの節の一部。列の見出しは PDF のテキストからの読み取りで、図は見ていない。
