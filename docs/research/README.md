# 調査の記録

新しく調べる前にここを見る。

- `topics/`：テーマごとの、今分かっていることのまとめ。1テーマ1ファイル。技術、データセット、評価の課題、道具、GPU など、使うかどうかを判断する対象を書く。新しい結果が出たら書き換える。
- `sources/`：他の人が公開した1つの出典に書いてあること。1出典1ファイル。論文、記事、公開の実験記録を置く。原則として書き換えない。読書ノートなど又聞きで読んだものは、原文を読んでいないことを書く。
- `experiments/`：自分たちの1回の調査や実験でしたことと、分かったこと。1回1ファイル。他の人のモデルでも、自分たちが測ったものはここに置く。

リンクは `topics/` から他の2つへの一方向にする。データセットのカードの内容はそのデータの `topics/` のメモに書き、コードは引用するメモに出典を書く。論文の本文やデータそのものは置かない。チャットの発言は入れず、公開の論文や記録で裏を取ってから、その出典のメモに書く。

`language-model` の記録にある内容は写さず、そちらを指す。

## テーマ

- [h100.md](topics/h100.md)
- [scoring.md](topics/scoring.md)
- [climbmix.md](topics/climbmix.md)
- [retrieval.md](topics/retrieval.md)
- [optimizer.md](topics/optimizer.md)

## 出典

- [autoresearch.md](sources/autoresearch.md)
- [nanogpt.md](sources/nanogpt.md)
- [modded-nanogpt-first-commit.md](sources/modded-nanogpt-first-commit.md)
- [retro.md](sources/retro.md)
- [flash-attention.md](sources/flash-attention.md)
- [transformer-engine.md](sources/transformer-engine.md)
- [adamw.md](sources/adamw.md)
- [muon.md](sources/muon.md)
- [muon-is-scalable.md](sources/muon-is-scalable.md)
- [practical-efficiency-of-muon.md](sources/practical-efficiency-of-muon.md)
- [fantastic-pretraining-optimizers.md](sources/fantastic-pretraining-optimizers.md)
- [optimizer-scaling-laws.md](sources/optimizer-scaling-laws.md)
- [soap-muon-and-beyond.md](sources/soap-muon-and-beyond.md)
- [benchmarking-optimizers.md](sources/benchmarking-optimizers.md)
- [algoperf.md](sources/algoperf.md)

## 実験

- [baseline.md](experiments/baseline.md)
- [gpu-prices.md](experiments/gpu-prices.md)

## `language-model` にある関係の記録

- まとめ：[compute-optimal-size](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/compute-optimal-size.md)、[batch-size](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/batch-size.md)、[normuon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/normuon.md)、[muon-plus](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muon-plus.md)、[muoneq](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muoneq.md)、[soap-muon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/soap-muon.md)、[head-dim-128](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/head-dim-128.md)、[qk-norm](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/qk-norm.md)、[swiglu](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/swiglu.md)、[zero-output-init](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/zero-output-init.md)、[vocabulary-size](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/vocabulary-size.md)、[interruptible-gpu](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/interruptible-gpu.md)
- 出典：[chinchilla](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/chinchilla.md)、[power-lines](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/power-lines.md)、[climb](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/climb.md)、[nanochat-log](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/nanochat-log.md)、[modded-nanogpt-anvil2](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/modded-nanogpt-anvil2.md)、[marin-nano-arch-ablations](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md)、[modded-nanogpt-optimization-track](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/modded-nanogpt-optimization-track.md)、[imu-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md)
