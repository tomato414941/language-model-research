# 活性化関数

MLP の中で、広げた値を曲げる計算。基準も最初の束も GELU。候補は次の通り。

- **ReLU²：** 正の値を2乗し、負の値を0にする。重みは増えない。
- **SwiGLU：** 行列を1つ足して、ゲートを掛ける。LLaMA などの標準。
- **xIELU、xIPReLU：** ReLU² の負の側にも勾配を持たせ、層ごとに学習する値を2つ持つ。

## 分かっていること

損失の差は、同じ表の中で比べた自分たちの計算。[Primer](../sources/primer.md) は perplexity の対数で比べた。

**GELU から ReLU²：**

| 出典 | 規模 | 損失の差 |
|---|---|---|
| Primer | 1.1億、C4 | 約 −0.7%（図から読んだ値） |
| Primer | 約3,500万、LM1B | 良い向き。T5 では大きく、T2T ではごく小さい（数字なし） |
| [xIELU の論文](../sources/xielu.md) | 11億、40億トークン | −0.84%（相手は GELU に近い SiLU） |
| [Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md) | 1.5億〜2.8億 | 0（QK-norm のあとに足した） |
| [記録5のコード](../sources/modded-nanogpt-records.md) | 約1.2億（12層、幅768） | コメントに「約1〜2%」。測り方は書いていない |

**ReLU² と SwiGLU は割れている：**

| 出典 | 規模 | 結果 |
|---|---|---|
| [nanochat の LOG](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/swiglu.md) | d12、d24 | 回数、時間、計算量のすべてで ReLU² が勝ち |
| Primer | 1.1億 | ReLU² が約0.2%勝ち（図） |
| xIELU の論文 | 11億、1,250億トークン | ReLU² が0.68%勝ち |
| [ReLU² Wins](../sources/relu2-wins.md) | 13億、1,000億トークン | ほぼ同じ。途中は SwiGLU が少し先を行く |
| xIELU の論文 | 11億、40億トークン | SwiGLU が1.04%勝ち |

SwiGLU は GELU より良いという報告が多い（[GLU Variants](../sources/glu-variants.md) で −2.6%、[BTLM](../sources/btlm.md) で −1.37%、[BLOOM の比較](../sources/bloom-ablations.md)で少し良い）。

**xIELU は ReLU² より良いという報告がある：** xIELU の論文で −0.60%（1,250億トークン）と −1.16%（40億トークン）。xIPReLU は −0.43% と −1.16%。[Apertus](../sources/apertus.md) は70Bに採用し、15億で SwiGLU から −2.0%。結果はどれも11億以上で AdamW 系、xIELU の著者が関わるもの。今の規模と Muon の結果は見つけていない。負の側に勾配を持たせる近い案の LeakyReLU(0.5)² は、[nanochat](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/leaky-relu-squared.md) で1回あたりは少し良いが遅く、時間では負けた。

**速さ：** xIELU の論文の11億で、GELU 544ミリ秒に対して ReLU² 534ミリ秒（1.8%速い）。xIELU は560ミリ秒で ReLU² より4.9%遅い（最適化なし）。xIPReLU は542ミリ秒で1.5%遅い。今の換算で、それぞれ約0.1%の得、約0.27%の損、約0.08%の損。今の H100 では測っていない。

PyTorch の自動の16ビット計算（autocast）では、GPU の上で2乗だけが32ビットになるはず（`square` は `pow` を呼び、`pow` は `autocast_mode.h` の32ビットの一覧にある。`F.rms_norm` も同じ一覧にある）。GPU では確かめていない。nanochat、autoresearch、modded-nanogpt のコードも同じ書き方で、`torch.compile` の下で差がどれくらいかは分からない。

**初期値：** ReLU² の出力は入力の大きさの2乗で変わるので、MLP の入口の重みの大きさに左右される。入力の2乗平均が1のときの入口の出力の標準偏差は、nanochat 0.4、modded-nanogpt 約0.58、autoresearch 1.0 で、最初の束の初期値は幅640で約0.51（どれも自分たちの計算）。

**他の用途：** ReLU² は推論で疎にしやすい（ReLU² Wins）。[Stochastic activations](../sources/stochastic-activations.md) も推論を速くする話で、今の BPB の候補ではない。

## 今の見方

2026-11時点の見方では、2番目の束は ReLU² にする。同じお金での効果は0〜1%と見ている。理由は次の通り。

- GELU との比較は、0のものはあるが悪くなった報告がない。少し速い。
- 同じくらいの規模で Muon と動かした例（nanochat、autoresearch。同じ流れのコードで、独立した確認ではない）があるのは ReLU² だけ。束は測るのが1回で、悪さをしても気づきにくい。
- 1行の変更で、重みが増えない。

入口の重みの初期値は、他のプロジェクトの範囲の中なので変えない。

xIELU は候補に残す。効果の見込みは ReLU² より0.3〜1%良いくらいだが、今の規模の結果がない。

## 見方が変わりそうな条件

- 今の規模で xIELU か xIPReLU と ReLU² を比べた結果が出た場合。
- 学習が不安定になった場合。ReLU² は勾配が値に比例して大きくなる。入口の重みの初期値も見直す。
- 効果を項目ごとに測った場合。
- 推論の費用も数える課題にした場合。ReLU² がさらに有利になるかもしれない。
