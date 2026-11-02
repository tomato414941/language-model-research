# ClimbMix

2026-10時点の基準の学習データ。`autoresearch` や nanochat と同じ `karpathy/climbmix-400b-shuffle`（`915333b`）で、結果を比べやすい。データの選び方は候補の1つで、変えるなら基準の重みを新しい物差しで採点し直す。

- 英語だけ。1ファイル約92MB、6,543ファイル。
- 配布元の表示は MIT だが、元の Nemotron-ClimbMix は CC BY-NC 4.0。
- そのまま使い、評価の問題を含む文書を除く処理はしていない。入っている数は数えていない。v5 との違いは[全問の測定](../experiments/harness-milestone.md)に書いた。

データの由来、効果、混ぜ方が一部のベンチマークの成績で選ばれていることは、`language-model` の [climbmix.md](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/climbmix.md) を見る。
