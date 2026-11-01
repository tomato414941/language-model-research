# Logit の softcap

予測の値（ロジット）を `c * tanh(logits / c)` で滑らかに抑える。重みは増えない。`language-model` にも短い[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/logit-softcap.md)がある。

## 分かっていること

**品質：** なしと比べた比較は1つだけで、差は小さい。差は自分たちの計算。

| 出典 | 比べたもの | 結果 |
|---|---|---|
| [modded-nanogpt の記録9](../sources/modded-nanogpt-records.md) | なし → 30（QK-norm あり、3,200回） | 3.2844 から 3.2791（−0.16%） |
| 記録18 | 30 → 15 | 同じ損失に要る更新が1,490回から1,390回（換算で約 −0.4%） |
| [nanochat](../sources/nanochat-log.md)（24層） | 5〜30 | 5はひどく、ほかはほぼ同じ。一番良い20で約 −0.14% |

記録18の著者は、小さい規模で効く構造の押しつけと見ている。

**安定：** [発散の比較](../sources/training-stability-methods.md)（8.3億）では QK-norm と並んで一番安定し、2つを組み合わせると学習率をさらに1.5倍にできる。2番目の束には QK-norm が入っている。

**費用：** [2番目の束](../experiments/second-bundle.md)の H100 で、1回の更新が2.7%遅い（0.599秒と0.615秒。換算で +0.15%）。記録9（語彙50,304）は2.0%で、語彙が小さければもっと小さいという読みは外れた。

**出だし：** 2番目の束の形で、本来の約500分の1のバッチで試すと、x0、resid の重みで見えた4回目の更新の損失の跳ね（9.36）が、softcap ありでは出なかった（8.44。[確認の記録](../experiments/second-bundle-code-check.md)）。品質の比較ではない。

**採用：** [Gemma 2](../sources/gemma-soft-capping.md) は最後の層に30を使い、Gemma 3 は QK-norm に置き換えた。autoresearch と modded-nanogpt の最適化の部門は15を使い、最適化の部門は普通でない部品を外したあとでも残している。

**ほかの形：**

- modded-nanogpt の最適化の部門の `15 * x / √(x² + 15²)`。tanh より軽いが、比較は見つけていない。
- `23 * sigmoid((x + 5) / 7.5)` は、nanochat では少し悪く、FP8 でしか効かないかもしれない。
- [z-loss](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/z-loss.md)：損失の罰で大きさを抑える。IMU-1 などが使っているが、効果の比較は読んでいない。

## 今の見方

2026-11時点の見方では、2番目の束に入れる。autoresearch と同じ `15 * tanh(logits / 15)` にする。同じコストでの効果は0〜0.4%と見ている。理由は次の通り。

- 比較はどれも良い向きで、悪くなった報告は見つけていない。
- 1行の変更で、測った費用は下の境目より小さい。
- 束は測るのが1回なので、Muon で動いた例に揃える。

ただし根拠は modded-nanogpt の流れにほぼ限られ、Gemma 3 は QK-norm があれば要らないと判断した形になっている。

`tanh` は bf16 だと精度が足りないので、nanochat と autoresearch のように fp32 に戻してからかける。

## 見方が変わりそうな条件

- 形を変えて遅くなる量が変わった場合。3%より遅いなら、外すことを考える。3%は換算で約0.17%で、見込みの中くらいの値に近い。測った2.7%はこの近くにある。
- softcap のあるなしを今の規模で比べた結果が出た場合。Gemma 3 の判断が小さい規模でも当てはまるなら、要らない。
- 語彙を大きくする場合。費用が増える。
- 効果を項目ごとに測った場合。
