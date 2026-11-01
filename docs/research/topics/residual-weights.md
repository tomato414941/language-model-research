# x0、resid の重み

層ごとに学習する2つの数で、各ブロックの前に `x = resid[i] * x + x0[i] * x0` とする。`x0` は埋め込みの直後の値で、深い層にもトークンそのものの情報を届ける。増える重みは層の数の2倍だけ。`language-model` に短いメモ（[x0](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/x0-lambdas.md)、[resid](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/resid-lambdas.md)）がある。

## 分かっていること

差は自分たちの計算。

| 出典 | 条件 | 足したもの | 結果 |
|---|---|---|---|
| [nanochat](../sources/nanochat-log.md)（01-11） | Muon、8〜20層、BPB | x0 と resid | 8層 −0.95%、12層 −0.79%、16層 −0.63%、20層 −0.46% |
| [modded-nanogpt の記録9](../sources/modded-nanogpt-records.md) | Muon、12層、幅768、3,200回の更新 | 近道（value residual のあと） | −0.28%。1回の更新は146から152ミリ秒（4.1%遅い） |
| [Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md) | 1.5億〜2.8億、1つずつ足す | resid、そのあと x0 | −0.25%、−0.08% |

- **nanochat の数字は高めに出ている：** 学習率を振って一番良い値を選んだ結果で、Value embeddings を入れる前の比較。Value embeddings もトークンの情報を深い層へ届けるので、重なるかもしれない（自分たちの読み）。記録9は value residual のあとで、差が小さい。
- **費用は割れている：** nanochat は計算はほとんど増えないとし、記録9は4.1%遅い（換算で +0.23%）。式の上では1層ごとに掛け算2つと足し算1つで、今の GPU では測っていない。
- **学習率：** nanochat は、足す x0 は0.5でよく、掛ける resid は100分の1でないとうまくいかないとしている。x0 の beta1 は0.96が一番良く、0.98で大きく悪くなる（d20）。[autoresearch](../sources/autoresearch.md) は x0 を0.1、resid を1.0から始め、学習率は0.5と0.005。今の nanochat は x0 を0.20から0.05へ、resid を1.15から1.05へ、層ごとに下げて始める。
- **一番良い学習率は深さで違う：** nanochat の x0 の値は8層で0.2、12層で0.6、16層で0.2、20層で0.1。今の0.5は調整していない。
- **出だし：** 2番目の束の形で、本来の約500分の1のバッチで試すと、重みありは4回目の更新で損失が跳ね（9.36と8.04）、30回目ごろまでにほぼ追いつく（[確認の記録](../experiments/second-bundle-code-check.md)）。品質の比較ではない。
- **学習された値：** [1回の学習を見た記事](../sources/modded-nanogpt-lambdas.md)では、x0 の重みは多くの層で大きい。あるなしの比較ではない。

**採用：** 使っているのは nanochat、autoresearch、modded-nanogpt（同じ流れのコード）と Marin の比較で、大きなモデルでの採用は見つけていない。modded-nanogpt の最適化の部門は「普通でない部品」として外している。

**近い考えの別の形：**

- [TIDE](../sources/tide.md)（Apple）：トークンごとの表を複数持ち、各層で混ぜて足す。7.5億〜30億で効果があり、大半は珍しいトークン。今の語彙なら表は小さいが、Muon や今の規模の結果はない。
- [KromHC](../sources/kromhc.md)、mHC：残差の流れを複数に分ける。nanochat の x0 と resid の形と比べて、6層で −0.48%、12層で −0.35%（mHC）。流れの分だけ計算が増える。mHC は DeepSeek-V4 が採用している。
- [Attention Residuals](../sources/attention-residuals.md)：前の層の値を注意で選ぶ。

## 今の見方

2026-11時点の見方では、2番目の束に入れる。設定は autoresearch と同じにする（x0 は0.1から、学習率0.5、beta1 0.96。resid は1.0から、学習率0.005）。同じコストでの効果は0〜0.5%と見ている。理由は次の通り。

- 比較はどれも良い向きで、悪くなった報告は見つけていない。
- 1行の変更で、重みはほとんど増えない。
- 束は測るのが1回なので、Muon で動いた例に揃える。autoresearch の学習率の組も、この形で動いている。

`x0` は埋め込みの直後の値なので、埋め込みを分離して RMSNorm をかける形が前提（embeddings.md）。

TIDE、mHC、Attention Residuals は候補に残す。

## 見方が変わりそうな条件

- 遅くなる量が分かった場合。5%より遅いなら、外すことを考える。5%は換算で約0.27%で、品質の利点の中くらいの値に近い。
- Value embeddings との重なりが測れた場合。
- 本来のバッチでも学習の初めに損失が大きく跳ねる場合。x0 の学習率を下げる。
- TIDE を Muon と今の規模で比べた結果が出た場合。
- 効果を項目ごとに測った場合。
