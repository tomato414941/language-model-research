# 最初の束

一度の学習で測る、最初の変更のまとまり。ここの見方は2026-11時点の仮のもので、まだ GPU で測っていない。新しい結果が出たら書き換える。

`configs/bundle.py` が今の束で、次の変更が入っている。それぞれ設定で外せる。

- オプティマイザ（optimizer.md）：Muon と AdamW の併用、学習率、Polar Express、NorMuon、cautious weight decay、学習率の下げ方。
- Value embeddings。
- ヘッド128次元（5ヘッド）。
- 10層、幅640。

FlashAttention-3 と速度の調整は、H100 で確かめる必要があるのでまだ入っていない。

## Value embeddings

トークンごとの表を、最後の層から1層おきに置き、注意の value に足す（[`language-model` のメモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/value-embeddings.md)）。形は nanochat の `gpt.py` に合わせた。ゲートは入力の先頭12次元から 3·sigmoid でヘッドごとに作り、重みは小さな正の値から始める。autoresearch は 2·sigmoid、32次元、重みは0から。

根拠は、[ResFormer](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/value-residual.md)（同じ損失に要るトークンが20.3%、パラメータが16.1%少ない。AdamW、8,200万〜4.7億）、[Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md)（ゲートまでで −1.1%）、nanochat の LOG（ステップ、計算量、時間のすべてで勝った）。ResFormer は最初の層の value を再利用する形で、表を持つ形そのものの論文は見つけていない。

表の学習率0.1は未調整で、nanochat の幅768の値（0.15）と autoresearch の幅640の換算（約0.66）より低めにした。基準のモデルは value の大きさが nanochat より小さいと見たため。表は計算をほとんど増やさないが、10層・幅640で2,620万パラメータ増え、メモリを使う。

## ヘッド128次元

根拠は Marin の表の1行（−1.0%）と、nanochat の LOG の「Head dim definitely prefers to be 128 instead of 64」だけ。128と64を直接比べた論文は、1回の検索では見つからなかった。LEMA（arXiv 2609.25802）は64より小さい幅と比べているが読んでいない。手間が小さいので入れている。

## 10層、幅640

計算量（6ND、約2.7e17）を基準と同じにしたときの比較。[Chinchilla](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/chinchilla.md) の式による loss の変化で、右の列は [Volkova ら](../sources/optimizer-scaling-laws.md)の係数（トークンへの効き1.41〜2.08倍）を掛けたもの。別々の研究を組み合わせた粗い計算で、モデルの大きさも式の範囲より小さい。

| 形 | パラメータ | トークン | 1パラメータあたり | 式 | Muon で補正 |
|---|---|---|---|---|---|
| 12層・幅768（基準） | 9,100万 | 5億 | 5.5 | 基準 | 基準 |
| 12層・幅640 | 6,400万 | 7.1億 | 11 | −1.0% | −0.4〜−0.7% |
| 10層・幅640 | 5,400万 | 8.4億 | 15 | −1.2% | −0.3〜−0.8% |
| 8層・幅640 | 4,500万 | 10.2億 | 23 | −1.3% | −0.1〜−0.7% |
| 8層・幅512 | 2,900万 | 15.6億 | 53 | −0.8% | +0.1〜+1.1% |

nanochat の LOG（2026-01-27）は、Muon と表を使う形で、1e18〜1e19 の最適な比を埋め込みを除いて約10.5とした。今の計算量まで伸ばすと約5,500万パラメータで、この形に近い。幅÷層は基準と同じ64で、nanochat の LOG は64のほうが128より良いとしている。

差は小さく、細かい順位は当てにならない（[Approach 2 の問題](../sources/problems-with-chinchilla-approach-2.md)、[個人の確認](../sources/chinchilla-check.md)）。今の規模で実際に比べた結果はなく、nanochat の最小は 1e18、[Delphi](../sources/delphi.md) は 3e18 から。

更新の回数は計算量からの見積もり（1,602回）で、学習の前に実際の速度から決め直す。

## 見方が変わりそうな条件

- 束が基準より悪いか、見込みより大きく外れた場合。原因は1つずつ外して確かめる。
- 表の学習率など未調整の値で、不安定になった場合。
- 同じ計算量で形を変えた実測が、上の表と違った場合。
- 推論の費用も数える課題に変える場合。小さいモデルが有利になる。
