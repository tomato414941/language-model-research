# ClimbMix

2026-10時点の基準の学習データ。`autoresearch` や nanochat と同じ `karpathy/climbmix-400b-shuffle`（`915333b`）で、結果を比べやすい。データの選び方は候補の1つで、変えるなら基準の重みを新しい物差しで採点し直す。

- 英語だけ。1ファイル約92MB、6,543ファイル。
- 配布元の表示は MIT だが、元の Nemotron-ClimbMix は CC BY-NC 4.0。

データの由来、効果、混ぜ方が一部のベンチマークの成績で選ばれていることは、`language-model` の [climbmix.md](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/climbmix.md) を見る。
