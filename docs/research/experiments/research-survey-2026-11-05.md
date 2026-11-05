# 2026-11-05 の調査

4番目の束の8つ目を決める前に、新しい論文と記事を37本確かめた。ほとんどは検索の要約で読み、IMU-1 と Parameter Golf の分析だけは本文を読んだ。

## 束に関わったもの

| 出典 | 分かったこと |
|---|---|
| IMU-1（arXiv 2602.02522） | 7,000万の代わりのモデルで、QK-norm −0.63%、正規化した value residual −0.38%、LayerNorm scaling −0.36%、ヘッドごとのゲート −0.05%。NorMuon と cautious weight decay で −5.2%。WSD は下げる期間 20% が cosine と同じで、10% と 30% は悪い。学習のデータにはコードが約19%、数学が約13%ある（集めた量からの見積もり） |
| Parameter Golf: What Really Works?（arXiv 2607.01517） | FineWeb の BPB を小さいモデルで競った記録の分析。上位の中でも効いたのは、評価のときの n-gram の混ぜ合わせ、Gated DeltaNet、value residual、窓の注意。全体では文書ごとの注意（VarLen）、トークンの組の表、MuonEq-R、重みの平均なども良い向き。1つの手法の効きはたいてい1%未満で、積み重ねで効く。相関の分析で、比べ合わせた実験ではない |
| SmolLM2（arXiv 2502.02737） | ウェブ、コード、数学を混ぜる。後の段階で 75/20/5。FineMath と Stack-Edu を作った |
| FineMath の比較（Hugging Face） | 数学のデータの質で、数学の課題の点が変わる |

## そのほか確かめたもの

- **最適化：** MUON+（2602.21545。NorMuon、AdaMuon より良いとする）、HTMuon（2603.10067。6,000万と1.35億で NorMuon より良く、組み合わせるとさらに良い）、MuonEq（2603.28254）、Newton-Muon（2604.01472）、MALT（2608.05088）、SOAP, Muon, and Beyond（2607.20548。8B で SOAP がわずかに上）、最適化の調査（2608.28557）、Benchmarking Optimizers（2509.01440）、AdaMuon（2507.11005）、MARS（2411.10438）。
- **学習の進め方：** Small Batch Size Training（2507.07101）、Multi-Power Law（2503.12811）、Marin 8B の振り返り（WSD-S）、Marin の学習率の予定の issue #764。
- **データと目的：** Data-Constrained Pretraining（2606.06888。データが足りない側の話）、Bridging Compute- and Data-Optimal Pretraining（2607.25271。検証の BPB が下流の点をよく予測する）、Self-Play Pretraining（2609.30063）、ESLM（2505.19893）、SwallowMath/SwallowCode（2505.02881。ほかのモデルで書き直すので使えない）、Scaling Inherently Interpretable LMs（2608.07594）。
- **表と記憶：** Engram（2601.07372）、Memory Grafting（2605.20948）、Lngram v2（2609.03426）、Qwen の新しい形の設計（2608.30320。n-gram の表を含む）。
- **注意：** 注意の進化の調査（2609.39661）、Deconstructing Attention（2510.11602）、Language Models Can Control Their Own Attention（2609.02737。推論の時の話）、Tucker Attention（2603.30033）、小さいモデルの形の比較（2512.20877）。
- **トークナイザ：** 46の設定の比較（COLM 2026、2608.18062）、言語ごとの影響（2610.12144）、バイトでの模擬（2604.27263）、バイトのモデルへの蒸留（2609.12303）、ByteFlow（2603.03583）、トークナイザの拡張（2607.15232）、Boundless BPE（2504.00178）。
- **Parameter Golf：** 公式のリポジトリと順位表の記事。

## 決めたこと

- 8つ目は、FineMath（4plus）とコード（Common Pile）を、それぞれ学習の行の10%混ぜることにした。OLMES の数学とコードのスイートに直接効く。混ぜ方は SmolLM2 と IMU-1 を目安にした。
- value residual は IMU-1 と Parameter Golf の両方で良い向きで、次の束の候補に残す。今の束は Value embeddings を持ち、重なる可能性がある。
- IMU-1 では下げる期間が 30% で 20% より悪く、4番目の束の 75% とは逆の向き。束では試すが、結果が悪ければまずここを見直す。
