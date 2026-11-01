# 入力と出力の埋め込み

トークンを読む表（`wte`）と、次のトークンを予測する表（`lm_head`）。基準と最初の束は1つの表を共有し、学習率は出力に合わせた0.004。

- **共有：** GPT-2、Gemma、Qwen3 の4B以下、SmolLM3。
- **分離：** Llama 3、OLMo 2、3、DeepSeek-V3、Qwen3 の8B以上。[nanochat](../sources/nanochat-log.md)、[autoresearch](../sources/autoresearch.md)、modded-nanogpt の最適化の部門。あとの3つは同じ流れのコードで、入力は標準偏差1.0前後で始めて直後に RMSNorm をかけ、学習率を大きくする（0.3〜0.7）。出力は0か0.001で始めて、学習率は小さい（0.002〜0.008）。

`language-model` にも短いメモ（[分離](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/untied-embeddings.md)、[学習率](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/embedding-learning-rate.md)）がある。

## 分かっていること

**一番近い比較：** [modded-nanogpt の記録7と8](../sources/modded-nanogpt-records.md)。1回524,288トークンで今と同じ、Muon、学習率も共有の0.0036で今と近い。同じ更新回数の検証損失の差は、125回で −2.45%、750回で −0.84%、1,500回で −0.56%、3,000回で −0.42%。分離、入力の RMSNorm、学習率、`lm_head` の0が1つにまとまっている。3.28 に届く更新は10%減で、換算で約 −0.6%。

**反対の結果：**

- あとの記録51は共有に戻して、更新が2.6%減った（換算で +0.14%相当）。FP8 の係数の調整も一緒で、共有の学習率は0.008。Value embeddings がある点は今の束に近く、Value embeddings があると入力の表の分離の利点は小さいのかもしれない（自分たちの読み）。あとの記録も、共有と途中からの分離の間を行き来している。
- [SmolLM3](../sources/smol-training-playbook.md) の12億と14.6億は、WinoGrande 以外で同じくらい。学習率は揃えた比較で、入力の学習率を上げてはいない。
- [Rethinking Embedding Coupling](../sources/rethinking-embedding-coupling.md) は事前学習の後の評価で 62.3 から 62.7。
- LSTM の[古い](../sources/using-the-output-embedding.md)[比較](../sources/tying-word-vectors.md)は共有が良いが、小さいデータで過学習が減る話で、今の条件とは違う。

SLlama（1,000万トークンの小さい比較で分離が良い）は要約しか読んでいない。

**入力の学習率：** [Importance of Embedding Layer Learning Rate](../sources/hyperparameter-transfer-embedding.md) は、AdamW で入力の学習率が一番大事な違いで、`lm_head` の初期値はほとんど効かないとしている。[Hayou、Liu](../sources/embedding-lr-vocabulary.md) は10億のモデルで √幅 倍が良いとしている。共有だと入力と出力で同じ学習率になり、上げられない。理論の上では `lm_head` の出力に係数を掛ければ共有でも上げられるが、実際に比べた結果は読んでいない。

**しくみ：** [Weight Tying Biases](../sources/weight-tying-biases.md) は、共有した表は出力の勾配で決まり、最初の数層の働きが悪くなるとしている。入力の勾配を増やしても下流の評価は良くならなかった。

**費用：** 大きいモデルが分離する理由は、表の割合が小さくなること。今は語彙8,192で2つ目の表は524万（全体の約6%）。増えるのはメモリとその更新で、注意や MLP の計算は変わらない。同じパラメータ数で比べる SmolLM3 の不利（層が減る）は、同じお金で比べる今の条件には当たらない。

**学習率の値：** [autoresearch](../sources/autoresearch.md) は入力0.6、出力0.004に (幅/768)^−0.5 を掛ける。幅640で0.66と0.0044。同じ ClimbMix、語彙8,192、Muon で動いている。

## 今の見方

2026-11時点の見方では、2番目の束は分離する。入力は標準偏差1.0で始めて直後に RMSNorm、学習率0.66。出力は標準偏差0.001で始めて、学習率0.0044。同じお金での効果は0〜0.6%と見ている。理由は次の通り。

- 束は測るのが1回なので、Muon で動いた例に揃える。autoresearch の学習率の組は分離した形のもの。
- 増えるのはメモリだけで、今の語彙なら小さい。
- 今の共有の表は入力の学習率が autoresearch の約160分の1で、低すぎる可能性がある。

共有のまま学習率を上げる形と、学習の途中で分離する形は候補に残す。どちらも今の形と Muon で単独に比べた結果がない。

## 見方が変わりそうな条件

- 共有と分離を今の形で比べた場合。Value embeddings があると差が小さいかもしれない。
- 語彙を大きくする場合。2つ目の表が大きくなる。
- 学習が不安定になった場合。入力の学習率は大きいので最初に疑う。
- 効果を項目ごとに測った場合。
