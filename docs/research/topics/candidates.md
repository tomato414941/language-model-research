# 候補

[2番目の束](../experiments/second-bundle.md)を、同じコストでさらに良くする候補のうち、入れるか決めていないものの一覧。上にあるものほど優先する。2026-11-02 時点の仮の順番で（[最新研究の調査](../experiments/latest-research-survey.md)、[広い範囲の調査](../experiments/broad-research-survey.md)）、ほとんどは今の規模で測っていない。順番は、選択式の点の報告があること、読む量が今に近いこと、手間が小さいことを重く見た。新しい候補は足し、入れると決めたものは消す。入れたものは `configs/` にあり、理由はそれぞれの `topics/` にある。

- 候補は種類で外さない。モデルや学習法のほか、速度、料金、データ、課題設定も入れる。効きにくそうなものも、理由を添えて後回しにするだけで消さない。
- 順番は、効果の見込み、根拠の強さ、手間で決める。大きくても根拠が弱いもの、実装が重いものは下にある。
- 数字は同じコストで BPB が下がる割合の粗い見込みで、束を測る前に基準に対して見積もったもの。束の上では小さくなるものがある。重なるものもあるので足し算はできない。「不明」は数字の根拠がないもの。
- 大半が過去のメモや要旨からの判断で、元の論文を読み切っていない。決める前に読む。

## 測って小さかったもの

| 候補 | 測った結果 |
|---|---|
| 注意の計算の方式（cuDNN、FlashAttention-2、3、4） | 1回の更新の差は1.6%以内。形が新しいときの準備で約0.2% |

## 次に調べる

| 候補 | 効果の見込み | 根拠 | 手間 |
|---|---|---|---|
| n-gram の計数を入力に足す（COUNT。engram.md） | 全体で検証の損失が約2〜3.6%、課題の損失が約0.3〜2.5%。今の計算量に近い点では正解の BPB −6.6%、正解率 +1.6（図の読み取り） | 1研究のみ（2026-10）、基準は AdamW。表は上位8候補で約5.4GB（[数えた](../experiments/count-table-check.md)） | 中 |
| 埋め込みの向きを広げる損失を足す（[Dispersion Loss](../sources/dispersion-loss.md)） | 点の平均 +0.7〜1.5 | 1研究。小さいモデルで起きる問題で、途中学習は読む量が少ない。時間は3〜5%増 | 小 |
| 最後の重みを混ぜる（途中の重みを残すので、1回の学習で何通りも測れる） | 0.1〜0.6%。IMU-1 では点の平均 +.014 | 記録 #92、Terminal Shrinkage。nanochat では効かなかった。[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/checkpoint-averaging.md) | 小 |
| 注意の層を減らす（Composer は注意と MLP が1対2、記録 #92 は10層のうち7つ） | 点の平均 +1.1〜3.1%（3.5億〜30億）。同じ時間で読む量も増える | 探索2つと speedrun が同じ向き。[調査](../experiments/broad-research-survey.md) | 小 |
| MLP の一部をトークンで引く表にする（[STEM](../sources/stem.md)、[L³](../sources/l3.md)。engram.md） | 点の平均 +1〜2 | 2研究、3.5億と8億の結果で、STEM の読む量は今の20倍以上。ゲート付きの MLP やメモリが要る | 中 |
| 浅く始めて後で深くする（[Deep Progressive Training](../sources/deep-progressive-training.md)） | 計算3〜5倍分なら約6〜9%だが、当てはまるかは不明 | 1研究。1.24億は1パラメータ約2,500トークンで今の約200倍。点の報告なし。1回で試せる | 小 |
| 学習率を下げる期間と下限（今は後半で0まで。Recursive は95%で5%まで、記録 #92 は80%で20%まで） | 不明 | 割れている。[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/lr-schedule.md) | 小 |
| 1回の更新のトークンを小さく始める（今は524,288で一定。記録 #92 は約5.7万から59万、Recursive は147,456で一定） | 不明 | [Fast Catch-up](../sources/fast-catch-up.md)。nanochat は効果が小さいとして入れず、[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/batch-size-schedule.md) | 小 |
| 文書の頭にメタ情報を付けて学習し、最後に外す（MeCo、KoCo） | 要るデータが33%減（16億） | ClimbMix は `text` だけなので、話題などを作る必要がある。[調査](../experiments/broad-research-survey.md) | 中 |
| 学習する n-gram の表（engram.md） | 約1.1%（記録 #92 の表を大きくした分）。COUNT と重なる | 記録 #92、Recursive。GPU のメモリが要る | 中 |
| 文書の前のトークンをコピーするポインタを出力に混ぜる（CPLM） | 約0.7%（時間 −11%） | speedrun の PR #379 のみ、未採用。損失だけの報告 | 中 |
| 勾配を1.2乗する（[GradPower](../sources/gradpower.md)） | 約1%（20億の AdamW） | 1研究、Muon の数字は未確認 | 小 |
| ヘッドごとに直交化する（[T-Muon](../sources/t-muon.md)） | 0.6〜1.1% | 1研究、NorMuon との比較なし | 小 |
| [Muown](../sources/muown.md) | 約1%（10億以上） | 1研究、NorMuon との比較なし、1.24億は図のみ | 小 |
| 文書の途中から始まる行を少し混ぜる | 不明 | 2番目の束で弱さが分かった。document-rows.md | 小 |
| 束の学習率と weight decay の調整（良い値は形で変わるので、束の形を決めたあと） | 不明 | 実測できる | 小 |
| 埋め込みを共有したまま入力の学習率を上げる、学習の途中で分離する | 不明 | 記録の一部のみ | 中 |
| TIDE（トークンごとの表を複数持ち、各層で混ぜて足す） | 不明 | 7.5億以上、Adam のみ。residual-weights.md | 中 |
| 似た文書を同じ行に詰める（BM25） | 不明、大きい可能性 | 1研究のみ、13億で −3.2〜−4.1%。document-rows.md | 中 |
| 文書の中だけで注意を計算する | 0〜4% | 割れている（13億で −3.9%、nanochat でほぼ0）。document-rows.md | 中 |
| xIELU、xIPReLU（ReLU² より良いという数字。activation.md） | 0.3〜1% | 中（11億以上のみ、70B の Apertus が採用）。speedrun では約10%遅く負けた | 小 |
| MLP を広げる倍率（今は4倍。Primer は約3,500万で12倍が大きく効いた） | 不明、大きい可能性 | 1研究のみ、文脈64 | 小 |
| GPU の効率の改善（nanochat の約46%まで。autocast をやめる形など） | 約1.5% | 中 | 中 |
| 安い H100（モデルは変わらない） | 約1.2% | 計算上 | 小 |
| 易しい文書から読む順番 | 不明（正解率で1〜3%） | 1研究のみ、BPB では未確認 | 中 |
| Attention Residuals | 0.3〜1.7% | 中 | 中 |
| パッチ単位の学習 | 最大約4% | 1研究のみ、1パラメータ970トークン以上の結果 | 中 |
| Partial Key Offset | 約1% | 弱（MoE 上の結果、Partial RoPE が前提） | 小 |
| 線形注意との混成（Gated DeltaNet、Kimi Delta Attention、[Mamba-3](../sources/mamba-3.md)） | 1〜2%。Mamba-3 は1.8億で点 +0.7、8.8億で +2.5 | 中 | 大 |
| 重要でないトークンを逆伝播から外す（CENTRIFUGE） | 時間 −34.7%（11億以上） | 1研究、専用の実装が要る | 大 |
| MLP を疎か低ランクにする（Mixed Sparsity Training、CR-Net、Bi-Phase） | FLOPs 4分の1という報告。実時間は未確認 | 弱〜中 | 大 |
| 正規化を定数で近似する（anGPT） | 収束が最大40%速い（QK-norm の GPT と比べて）。時間3%増 | 1研究 | 中 |
| 大きいモデルの枝刈りや蒸留から始める（別の出発点） | 不明。読む量が少ないほど効くという報告 | Pruning vs. Scratch、Where to Begin。同じ計算で差がないという報告もある。[調査](../experiments/broad-research-survey.md) | 中 |
| mHC | 1.3%まで | 弱〜中。speedrun では悪化 | 中 |
| データの選び方（DCLM-Edu、PreSelect など。ClimbMix は nanochat が試した中で一番良く、今は400B のうち0.2%しか読まない。v6 の英語に替えると6課題の平均が .025 下がったが、原因は分けていない。v6-english-data.md） | 不明、大きい可能性 | 割れている | 中 |
| データの繰り返し | 不明 | 中 | 小 |
| 語彙の大きさ | 不明、大きい可能性 | 大規模のみ | 中 |
| SOAP-Muon | 約0.9%（ステップのみ） | 中 | 中 |
| 準備の時間の短縮（小さい規模でのみ効く） | 約1.6% | 実測から計算 | 中 |
| Partial RoPE | 0〜0.1% | 弱 | 小 |
| EMA-Nesterov | 約0.35% | 弱 | 小 |
| MUDD | 0.2〜4.9% | 弱 | 中 |
| MoE | 0〜数% | 今の規模では弱い | 大 |
| 文脈の長さ（autoresearch は2,048。2,048は1回の更新が7.5%遅い） | 不明 | 弱 | 小 |
| 文脈を短いものから伸ばす | 不明 | 弱 | 中 |
| 注意の窓を区切る（SSSL） | 不明（速度） | 弱 | 小 |
| 中断ありの貸出（モデルは変わらない） | 最大約6.9% | 別のプロジェクトの試しでは止まった | 中 |
| 他の GPU（4090、5090、B200 など。[調査](../experiments/broad-research-survey.md)） | 4090 で計算あたり約1.4倍（粗い計算）なら約1.8%。B200 は料金次第で約3倍 | サブプロジェクト。実測なし | 中 |

## 効果が小さそうな候補

| 候補 | 効果の見込み | 根拠 |
|---|---|---|
| 注意のゲート | 0〜0.3% | 割れている |
| Value residual（Value embeddings と重なる） | 約0.4% | 弱 |
| Layernorm scaling | 0〜0.4% | 割れている |
| z-loss | 不明 | 弱 |
| Muon+、MuonEq | ごく小さい | 弱 |
| Anvil の更新則（記録 #92 で約0.15%）、FARO、Newton-Muon、radial brake、u/w floor、PowerCool | 小さいか不明 | 弱 |
| Scion、Kron | 学習量が多いと有利 | 中 |
| MuonH | 不明（Adam 系との比較のみ） | 中 |
| NAdamW、Mars、AdEMAMix、Schedule-Free AdamW | 今の条件では小さそう | 割れている |
| 別のデータ（Ultra-FineWeb、Essential-Web、FinePhrase など） | 不明 | 未検討 |
| 数字を1桁ずつに分ける、区切りの正規表現 | 不明 | 弱 |
| Narrow QK（速度と引き換えに損失が悪化） | 不明 | 弱 |
| Liger、compile の時間の短縮 | 小さい | 弱 |

## 今の規模では効きにくそうな候補

| 候補 | 後回しの理由 |
|---|---|
| Engram の完全な形 | 小さい規模では2トークンの表と差がない（nanochat の表は小さい。engram.md） |
| Screening | 主な利点は長い文脈 |
| H-Net | 構造が大きく変わる。比較が少ない |
| CompleteP | 効くのは深いモデル |
| Per-layer embeddings | 中身を未確認 |
| Sampled softmax | 語彙が小さいので得が少ない。記録 #92 の時間 −12%は語彙50,304の場合で、損失の計算をまとめた分も含む |
| 長い文書の分割 | ほぼ差がない |
| MTP、SwiGLU、LeakyReLU²、XSA、直交初期化 | nanochat では効かないか、時間で負けた |
| FP8 | 12層ではかえって遅い。記録 #92 は専用のカーネルで時間 −7% |
| GQA | 主に推論のための手法 |
| RETRO 型の検索 | 2〜6%悪化の見積もり（DB を巨大にしない限り） |
| 合成データの事前学習 | 同じ計算で、自然な文章だけの学習に勝たなかった（[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/pre-pretraining.md)） |

## BPB では測れない候補

課題設定を変えるときに、改めて順番を決める。

- 指示学習（SFT）、好みの学習（APO）、指示用のデータ
- 正解で強化学習する（RLVR）
- ツール、ハーネス、検索で事実を引く力
- 重みにプログラムを組み込む（Qwenputer）
- 推論の費用も数える形（Train-to-Test、小さいモデルを長く学習する）
- 日本語、複数の言語、日本語の評価
- 第2の規模で効果を確かめる

個別の見方は同じ `topics/` にあり、ほかの多くは `language-model` の [`topics/`](https://github.com/tomato414941/language-model/tree/main/docs/research/topics) にある。
