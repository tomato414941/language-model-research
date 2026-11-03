# Common Pile v0.1

Kandpal ら、The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text（arXiv 2506.05209、2025）。30の出典から集めた、パブリックドメインと公開ライセンスの文章。7B の Comma v0.1-1T と 2T を学習している。

## 学習用の混ぜ方

- 学習データ（`common-pile/comma_v0.1_training_dataset`）の本段階は、元の4,636億トークンを重み付けして実効1兆344億トークン（README）。
- 実効で多い順に、peS2o（論文）2,600億、StackExchange 1,432億、Stack v2 の教育的なコード 1,355億、CC ライセンスの Web（CCCC）914億、USPTO（特許）394億、PubMed 366億。
- 重みの決め方（§4.2）：出典ごとに280億トークンで小さいモデルを学習し、その成績で重みを手で上げ下げした。1兆トークンで最大6回の繰り返しに収める。MixMin は手で決めた混ぜ方に勝たなかった。成績を測った課題は §4.3 と同じものと読めるが、明記はされていない。
- フィルタ（§4.1）：言語の判定、CCCC に DCLM の品質分類器（低いしきい値）、OCR の誤り、有害さ、PII、出典ごとの定型文。20-gram の90%以上が共通の文書を重複として除く。

## 結果

- 小さい規模の比較（§4.3）：OLC、Common Corpus、KL3M より全課題で良い。FineWeb がほとんどの課題で一番良く、Comma は MMLU と ARC で一番良い。HellaSwag、PIQA、CommonsenseQA では Pile、OSCAR、FineWeb よりかなり悪い。
- Comma v0.1-1T は、同じ予算のモデルに半分以上の課題で勝つ。ARC-C と MMLU に強く、HellaSwag と PIQA で遅れる（§4.4）。
- 評価の問題を除く処理は書かれていない。WinoGrande は Data Provenance Initiative のデータに入っているので評価から外した。

読んだのは HTML 版の §4.1〜4.4 と、学習データの README の該当部分（ツールで抜き出した引用から）。数値の表と付録は読んでいない。
