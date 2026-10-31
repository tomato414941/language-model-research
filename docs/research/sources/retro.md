# RETRO

Borgeaud ら、Improving Language Models by Retrieving from Trillions of Tokens（arXiv 2112.04426、ICML 2022）。

- 2兆トークンのデータベースから検索しながら学習して答える。パラメータが25分の1で、The Pile で GPT-3 や Jurassic-1 と同等の性能と報告している。
- 効果は1.5億〜70億パラメータで一定。データベースや検索する数を増やすと評価時に改善できる。fine-tuning 後の質問応答にも効いた。
- 評価データの一部が学習データに含まれる場合の評価方法を別に用意している。

読んだのは要旨と導入の要約まで。重複を除いたときの数字は読んでいない。
