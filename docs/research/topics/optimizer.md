# オプティマイザ

何を使って重みを更新するか。「今の見方」は2026-11時点で読んだものからの仮のもので、自分たちではまだ測っていない。新しい結果が出たら書き換える。

## 読んで分かったこと

- **AdamW：** [論文](../sources/adamw.md)の実験は画像分類で、言語モデルで他の手法と比べたものではない。
- **Muon と AdamW の関係：** 全部の代わりではなく併用する。[Muon](../sources/muon.md) は隠れ層の行列に使い、埋め込み、出力、正規化、スカラーは AdamW で更新する。[Kimi](../sources/muon-is-scalable.md) と DeepSeek-V4（arXiv 2606.19348）も同じ分け方で、基準の12層・幅768なら、パラメータの約92%が Muon 側になる。
- **差の大きさ：** 報告は割れている。

| 出典 | 条件 | Muon の差（AdamW 比） |
|---|---|---|
| [元の記事](../sources/muon.md) | 幅768、更新1回524,288トークン、FineWeb の速度記録 | 1.35倍 |
| [Fantastic](../sources/fantastic-pretraining-optimizers.md) | 1.3億、学習量1倍 | 約1.4倍 |
| 同上 | 12億 | 約1.1倍 |
| [Volkova ら](../sources/optimizer-scaling-laws.md) | ClimbMix、5,000万〜15億、学習量1.5〜10倍 | トークンへの効きが2.08倍 |
| 同上 | FineWeb、同じ範囲 | トークンへの効きが1.41倍 |
| [Kimi](../sources/muon-is-scalable.md) | 3.99億〜15億、最適なトークン量 | 計算量52%（約2倍） |
| [Essential AI](../sources/practical-efficiency-of-muon.md) | 1億〜40億 | トークンが10〜15%少ない |

- **他の手法：** 行列を使う SOAP や Kron は、学習量を増やすと Muon に並ぶか上回る（[Fantastic](../sources/fantastic-pretraining-optimizers.md)、[Volkova ら](../sources/optimizer-scaling-laws.md)、[NVIDIA](../sources/soap-muon-and-beyond.md)）。[Semenov ら](../sources/benchmarking-optimizers.md)は、長い学習で AdEMAMix と MARS を一番良いとしていて、Fantastic と食い違う。原因は決着していない。[AlgoPerf](../sources/algoperf.md) の調整ありの部門は、行列を使う Distributed Shampoo が優勝した。[modded-nanogpt の最適化の部門](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/modded-nanogpt-optimization-track.md)は、ステップ数で SOAP-Muon が Muon より少ない。ただし時間は問わない比較。
- **学習率：** Fantastic は、ピーク学習率6e-4の AdamW を基準として弱いとする。nanoGPT も同じ6e-4。同じ更新サイズの1.3億の調整例では0.008だった。モデルの形が違うので、そのまま当てはまるとは限らない。
- **費用：** 幅768、更新1回524,288トークンなら、Muon の計算の増分は上限0.7%（元の記事の計算）。メモリは AdamW より少ない。

## 今の見方

今の基準（9,100万パラメータ、5億トークン、更新1回524,288トークン、学習量は約0.27倍）で最初に試す候補は、Muon と AdamW の併用にしている。設定は autoresearch の値から始める。この実装には NorMuon と Polar Express が入っている。

そう見ている理由は次の通り。

- 比べられた中で一番少ない学習量1倍で、Muon が先頭だった。ただし0.27倍は比較の範囲の外で、外挿になる。
- 計算とメモリの費用が小さい。
- autoresearch が同じ953回の更新で動かしている。他の変更も入っているので、Muon だけの効果は分からない。

効果の粗い見積もりは、学習率の調整で最大2倍、Muon でさらに1.4倍、合わせて最大2.8倍分。計算量2倍で loss が約3.9%下がるという換算（[Open Athena の記事](https://www.openathena.ai/blog/pretraining-speedup/)にある Delphi の式 `loss = 1.65 + 117.8·C^−0.0966`）で、約6%になる。別々の条件の数字を掛けたもので、順番を決めるための目安にすぎない。

## 見方が変わりそうな条件

- 自分たちの測定で、見積もりと大きく違った場合。特に、調整した AdamW に Muon の束が勝てない場合。
- 学習量を増やす場合。SOAP や Kron のほうが良くなる可能性がある。
- 更新1回のトークン数を大きく減らす場合。Semenov らの AdEMAMix や MARS を見直す。
- SOAP-Muon などを、時間まで含めて今の規模で比べた結果が出た場合。
- モデルを大きくする場合。Fantastic では差が薄れ、NVIDIA ではバッチを大きくすると差が出ている。

## 関係するもの

- `language-model` のメモ：[NorMuon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/normuon.md)、[Muon+](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muon-plus.md)、[MuonEq](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muoneq.md)、[SOAP-Muon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/soap-muon.md)、[IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md)。
- 読んでいないもの：Polar Express（Amsel ら）。autoresearch の実装が使っている。
