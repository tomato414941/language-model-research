# Patch-Level Training の規模の節

Shao ら（arXiv 2407.12665）。手法と主な表は `language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/patch-level-training.md)にあり、ここにはそこにないことだけを書く。

- 出力の層は1つのままで、次のパッチの K トークンを同時に予測する。
- パッチの段階はデータを早く消費するので、データを多く必要とする。
- 優位はモデルが大きいほど小さくなり、データを増やすほど早く伸びる（表3、表4）。
- 一番小さい比較は3.7億に3,600億トークンで、1パラメータ約970トークン。

読んだのは手法と3.3、3.4の節の本文。表3、表4の数字は読んでいない。
