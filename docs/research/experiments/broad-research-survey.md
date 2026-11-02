# 広い範囲の調査

2026-11-02 に、同じ日の latest-research-survey.md が事前学習の工夫に偏っていたので、分野を広げて探し直した。候補の順番を決める材料にした。

## 探し方と限界

- **学会ごとの分野別の一覧（papernotes.org）：** ICLR 2026 の Pretraining 79本、ICML 2026 の Pretraining 27本、NeurIPS 2025 の Pretraining 51本、ACL 2026 の Pretraining 12本、ICML 2025 の Pretraining。ICLR 2026 と ICML 2026 の LLM Efficiency、Optimization、ICML 2026 の Model Compression は、学習に関わるものだけを要約で拾った。
- **分野ごとの検索：** 蒸留、逆伝播を減らす方法、合成データ、Transformer 以外の構造、圧縮、BabyLM、再帰、JEPA、料金、ハードウェア、学習中の重み付け、脳に着想を得た方法、推論時の学習、中国語と日本語の文献。

限界。

- 下の表の「一覧」は、一覧の機械の要約を読んだだけで、原文で確かめていない。「検索」は検索の要約だけを読んだ。
- NeurIPS 2026 はこのサイトにほとんど載っていない。ICML と NeurIPS の公式の一覧は取得できなかった。EMNLP と COLM、ICLR の Optimization の後半は見ていない。
- 中国語と日本語の検索では、ほとんど見つからなかった。MiniMind（約6,400万を約2時間）の費用の根拠は見つけていない。NLP2026 の一覧は見ていない。
- 今の束は1パラメータ12トークン。下の報告の多くは、読む量がそれより桁違いに多い。

## 計算を表に置き換える

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| STEM（ICLR 2026、stem.md） | MLP の入口をトークンごとの表にする。350M で点の平均 49.72→50.90 | 本文 |
| L³（ICML 2026、l3.md） | 層の中にトークンごとの表を置く。同じ計算の 809M で 48.28→50.25 | 本文 |
| Hierarchical Memories（arXiv 2510.02375） | 1.6億のモデルに、46億の記憶から1,800万を引いて足すと、2倍以上の普通のモデルに並ぶ | 検索 |
| LMLM（arXiv 2505.15962） | 事実の答えを損失から外して検索させる。3.82億が事実の課題で70億に近づく | 検索 |
| Scone、Memory Mosaics（NeurIPS 2025） | 頻出の n-gram の表、連想記憶 | 一覧 |

COUNT、記録 #92、Recursive は latest-research-survey.md にある。

## 形を変えて計算を減らす

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| Deep Progressive Training（ICML 2026、deep-progressive-training.md） | 0〜1層で80%まで学習してから深くする。計算が3〜5倍分 | 本文の一部 |
| Composer（arXiv 2510.00379、ICLR 2026） | 探索で注意と MLP が1対2の形が見つかる。3.5億〜30億で点が平均 1.1〜3.1%上がる | 検索 |
| AIRA-Compose（arXiv 2605.15871） | 11のエージェントが24時間で形を探す。10億で点が +2.4%、Mamba との混成で +3.8% | 要旨 |
| Mixed Sparsity Training（arXiv 2408.11746） | 疎な学習で GPT-2 の FLOPs が4分の1。実時間は要旨にない | 要旨 |
| Sparser, Faster, Lighter（ICML 2026） | L1 で MLP の99%以上が0になり、下流の影響はわずか。学習の時間は要旨にない | 要旨 |
| BLaST（arXiv 2507.03117）、CR-Net、Bi-Phase | 疎や低ランクの MLP。BLaST は事前学習で1.11倍 | 検索、一覧 |
| CENTRIFUGE（arXiv 2502.00340、ICLR 2026） | 重要でないトークンを逆伝播から外す。50%外すと逆伝播 −49.9%、全体 −34.7%。11億〜400億 | 検索 |

## 別の系統

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| Mamba-3（ICLR 2026、mamba-3.md） | 同じパラメータで 1.8億 42.8→43.5、8.8億 52.8→55.3 | 本文の一部 |
| 拡散の言語モデル（arXiv 2610.17590） | 同じ損失に要る計算の差が16倍から、ブロック4で3.0倍に縮まる。まだ普通のモデルより多い | 要旨 |
| ParaRNN、Energy-Based Transformers、Dual-objective（ICLR 2026） | RNN、エネルギー、2つの目的の混合 | 一覧 |
| 逆伝播を使わない学習、スパイキング、予測符号化 | 言語モデルの規模の結果は見つからない。層の4分の3を0次の推定にする Hybrid Pretraining は二次情報のみ | 検索 |
| Tiny Recursive Model（arXiv 2510.04871） | パズルの結果で、言語のモデルの結果は見つからない | 検索 |

## 損失と目標

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| Dispersion Loss（ICML 2026、dispersion-loss.md） | 埋め込みの向きを広げる損失を足す。GPT-2 の途中学習で 34.95→35.61 | 本文の一部 |
| GradPower（ICML 2026、gradpower.md） | 勾配を1.2乗する1行。20億で点 47.72→48.26、loss −0.022 | 要約 |
| Lost in Backpropagation（lost-in-backpropagation.md） | 出力の表が勾配の95〜99%を失うとする | 本文の一部 |
| Next-ToBE、Future Summaries、LLM-JEPA、CoCoMix（ICLR 2026） | 未来の語の袋を当てる、未来の要約を当てる、潜在の予測、SAE の概念を当てる | 一覧 |

## データ

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| MeCo（arXiv 2501.01956、ICML 2025） | 文書の頭に URL などを付けて学習し、最後の10%で外す。16億で要るデータが33%減。モデルが作った話題でも効く | 検索 |
| KoCo（ACL 2026）、Beyond URLs（ICLR 2026） | 意味の座標、細かいメタ情報を付ける | 一覧 |
| ADAPT（arXiv 2605.05227、ICLR 2026） | 文書の重みを学習中に変える。同じ計算で事前の選別より良いとする。数字は要旨にない | 要旨 |
| AC-ODM（ICML 2026）、Demystifying Data Organization（ACL 2026）、How Text Quality Interventions Reshape Neural Scaling Laws（ICLR 2026） | 混ぜ方の自動調整、読む順番、質の介入の2,000モデルの比較 | 一覧 |
| Beyond Rephrasing（arXiv 2607.28109） | 本の形に整えた合成データで、中間の学習の点が約 +1.09 | 検索 |
| It's All Training（NeurIPS 2026） | 百科の種から全部を合成して1段で学習する | 一覧 |
| Is a Document Educational or Just Wikipedia-Style?（ACL 2026） | 質の判定器は Wikipedia 風の文体に騙される | 一覧 |

ClimbMix の列は `text` だけで、URL などのメタ情報はない（2026-11-02 に確かめた）。

## 既存の重みや大きいモデルから始める

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| Small LLMs: Pruning vs. Training from Scratch（arXiv 2606.14150） | Llama-3.1-8B を枝刈りして始めると、同じトークンでランダムな初期値に勝つ。差は読む量と枝刈りの率が増えると縮み、率0.8でほぼ消える | 要旨 |
| Where to Begin（arXiv 2510.07227） | 部分ネットワークと蒸留で、100億トークンで FLOPs が5.16分の1、1,000億で1.26分の1。先生の計算を数えるかは要旨にない | 要旨 |
| Knowledge Distillation vs. Pretraining from Scratch（arXiv 2404.19319） | 同じ計算なら、普通の蒸留とゼロからの学習はほぼ同じ | 検索 |
| Don't Ignore the Tail、Energy Accounting of Distillation（ICML 2026） | H100 1枚で1週間の蒸留が可能。蒸留はエネルギーが約2.4倍 | 一覧 |

## 学習の進め方

| 研究 | 報告 | 読んだ深さ |
|---|---|---|
| anGPT（arXiv 2505.22014、NeurIPS 2025） | 正規化を定数で近似する。QK-norm の GPT より収束が最大40%速く、時間は3%増、weight decay と warmup が要らない | 検索 |
| Seesaw、WSM（ICLR 2026） | 学習率を下げる代わりにバッチを増やして時間 −36%。一定の学習率の重みの合流で下げる段階を置き換える | 一覧 |
| AdaGC（ICML 2026） | テンソルごとの勾配クリップで点が +1.27〜2.48%（大きなモデル） | 一覧 |
| SiameseNorm（arXiv 2602.08064、ICML 2026） | Pre と Post の2つの流れ。13億の算数で 28.1→39.6 | 検索 |
| Sharpness-Aware Pretraining（arXiv 2605.02105） | 2,000万〜1.5億で SAM、大きな学習率、短い下げ方が、事後学習での忘却を最大80%減らす | 要旨 |
| Conditioned Initialization（ICLR 2026） | 注意の行列の初期値の条件数を1にする。20〜30%速いという数字は二次情報で、言語の差は小さい | 検索 |
| RMNP（ICML 2026） | 直交化を行の正規化に替えて13〜44倍速い。今は Muon の費用が小さいので得は少ない | 一覧 |
| Dropout Universality（arXiv 2605.21648） | 前の層を強くする dropout。MLP と ViT の結果で、言語モデルの結果はない | 検索 |

## 料金とハードウェア

- Vast の料金（gpu-prices.md）で、RTX 4090 の中央値 $0.47 と H100 の $2.40 を比べると、計算あたりの料金は4090が約1.4倍良い（前の測定の速さからの粗い計算で、モデルの大きさが違う）。
- SemiAnalysis の 2026-07 の料金の調査では B200 が $1.73、MI355X が $1.50（1チップ1時間。推論の比較のページ）。B200 のピークは H100 の約2.3倍なので、この値で借りられれば計算あたり約3倍良い。Vast の B200 の値は確かめていない。
- MI355X の学習の費用を計算あたりで比べた資料は見つからない。
- TPU Research Cloud は申請制で TPU が無料。公式のリポジトリは 2026-04 にアーカイブされ、今も受け付けているかは確かめていない。コードの移植が要る。

## 隣の分野

- **BabyLM（EMNLP 2025 の報告の要旨）：** 新しい目的関数と構造が上位に来た。学習の FLOPs と成績は完全には相関しない。2024年の報告の要約では、読む順番の工夫の多くは効かなかった。
- **Parameter Golf（検索の要約。記事は取得できなかった）：** 16MB、8枚の H100 で10分。上位に Muon の weight decay、埋め込みのスペクトル初期化、文書ごとの推論時の学習がある。
- **Hutter Prize：** 2026年6〜7月に大きく進み、enwik9 が 100,424,672 バイト。古典的な文脈混合と文章向けの前処理で、学習と評価が同じデータなので BPB は比べられない。
- **推論時の学習（arXiv 2604.06169、ICLR 2026 ほか）：** 長い文脈の課題の結果。
- **自動の探索：** Recursive、AIRA-Compose。人が読む案より、組み合わせと調整で伸びている。

## 測り方の研究

- Relative Scaling Laws（arXiv 2510.24626、検索）：255のモデルで、損失と正解率はシグモイドの形で強く結びつく。
- Scale Efficiently（`language-model` のメモ）：損失は大きさでほぼ決まるが、点は形に左右される。
- Can Small Training Runs Reliably Guide Data Curation?（ICLR 2026、一覧）：小さなモデルの順位は学習率で入れ替わる。
- AI Progress Should Be Measured by Capability-Per-Resource（NeurIPS 2025、一覧）：同じ資源あたりの能力で進歩を測るべきとする主張。

## 分かったこと

- **点が上がるという報告は多い：** STEM、L³、Mamba-3、Composer、AIRA-Compose、Dispersion Loss、COUNT、MeCo、LARE など。latest-research-survey.md を書いた時点の「COUNT だけ」は、探した範囲が狭かったための誤り。
- **計算をメモリに置き換える流れが一番広い：** n-gram の表、トークンごとの表、記憶の束、データベースまで形はさまざま。
- **注意を減らす向きが別々に出ている：** Composer の探索、AIRA-Compose、speedrun の記録 #92。
- **読む量の違いに注意：** 今は1パラメータ12トークン。STEM の350M は約285、Deep Progressive Training の1.24億は約2,500。近いのは COUNT（10〜40）、T-Muon（約20）、speedrun、nanochat で、Dispersion Loss の途中学習も読む量が少ない。
- **料金も大きな手段：** 計算2倍で約3.9%という換算なら、GPU の選び方だけでモデルの改善の候補の多くより大きく動く可能性がある。

## 確かめていないこと

- どの手法も今の規模では測っていない。
- 「一覧」と「検索」のものの原文。
- 組み合わせの効果。
