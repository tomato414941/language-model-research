# FineWeb

Penedo ら、The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale（arXiv 2406.17557、2024）。データは `HuggingFaceFW/fineweb`。

## 作り方

- Common Crawl の96スナップショットの WARC から、trafilatura で文字を取り出す（§3.2）。
- fastText の英語の判定（0.65以上）、URL のブロックリスト、MassiveText の品質と繰り返しのフィルタ（§3.3）。続いて C4 のフィルタ（文末の句読点の規則を除く）と独自のヒューリスティック（§3.5、§3.6）。
- MinHash の重複除去は、スナップショットごとに別々に行う（§3.4）。メールアドレスと IP アドレスは匿名化（§3.7）。
- 処理を選ぶ実験は、CommonSense QA、HellaSwag、OpenBook QA、PIQA、SIQA、WinoGrande、ARC、MMLU の8課題の成績で比べた（§3.1）。
- 評価の問題を除く処理は書かれていない。

## 公開

- 約15兆トークン（GPT-2 のトークナイザで数えた）。ライセンスは ODC-By 1.0 で、Common Crawl の利用規約にも従う。
- 一様に抜いた標本（10BT、100BT、350BT）がある。10BT は15ファイル、約30.6GB（リビジョン `9bb295d`）。
- FineWeb-Edu は、FineWeb から教育的な度合いの分類器（Llama-3-70B の注釈で学習）で選んだ1.3兆トークンの部分集合（§4）。

読んだのは HTML 版の §1、§3.1〜3.7、§4 の該当する文（ツールで抜き出した引用から）と、Hugging Face のファイルの一覧。
