# オプティマイザ

何を使って重みを更新するか。「今の見方」は2026-11時点で読んだものからの仮のもの。自分たちの測定は束全体の1回だけ（[最初の束](../experiments/first-bundle.md)）で、Muon だけの効果は測っていない。新しい結果が出たら書き換える。

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
| [Polar Express の論文](../sources/polar-express.md) | 1.24億、10億トークン、更新1回32,768トークン、weight decay なし | loss が AdamW 4.197、Muon 3.639（−13%） |

- **Muon の改良：** [Polar Express](../sources/polar-express.md) は費用を変えずに、10億トークンで loss が1.4〜1.7%、100億トークンで0.3%下がった。NorMuon（[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/normuon.md)）は、同じ損失に要る更新が Muon より1.24億で6%、3.5億で15%少ない。11億では AdamW より21.74%少なく、Muon は10.43%。メモリは Adam の約半分で、更新の時間は AdamW より2.9%長い（54億の場合、arXiv 2510.05491）。
- **Cautious weight decay：** 1パラメータ20トークンの3.38億で AdamW が3.0136から3.0059（−0.26%）、100トークンの10億で AdamW が2.65から2.56、Muon が2.51から2.42（約−3.5%。arXiv 2510.12402、`language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/cautious-weight-decay.md)）。[IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md) の7,000万では約−1%で、基準の値に不一致がある。学習量が少ないと効果は小さい側と見ている。
- **Q、K、V を分けるか：** [元の記事](../sources/muon.md)は分けるほうが良いとし、NorMuon の著者は分けても改善がなかったとしていて、報告が割れている。ヘッドごとに分けると、[T-Muon](../sources/t-muon.md) は普通の Muon で6,000万〜3.5億の loss が0.6〜0.7%低いとする。NorMuon の [speedrun](../sources/modded-nanogpt-paired-head-muon.md) では、K と V を2ヘッドずつにした得は約0.04%で、[Group Muon](../sources/group-muon.md) ではヘッドごとに全部分けると最後は少し悪い。
- **2026年の改良：** [Muown](../sources/muown.md) は10億以上で約1%、[FARO](../sources/faro.md) は800ステップで約0.4%。どれも読んだ範囲では今使っている NorMuon と cautious weight decay の組と直接比べていない、または差が小さい。[記録 #92](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/anvil.md) では、更新則を戻しても4.9ミリナットで、差の大半は最後の重みの作り方から来ている。[効かなかった報告](../sources/modded-nanogpt-wr-blog.md)では、AdEMAMix など遅い状態を持つものは1,500ステップ未満で負けている。今の1,281回の更新も短い側。
- **勾配の累乗：** [GradPower](../sources/gradpower.md) は勾配を要素ごとに p 乗してからオプティマイザに渡す。論文の p=1.2 は1回131,072トークンの結果で、今と同じ524,288トークンと Muon の [Parameter Golf](../sources/parameter-golf-gradpower.md) では悪化し、0.9 のほうが少し良かった。

- **他の手法：** 行列を使う SOAP や Kron は、学習量を増やすと Muon に並ぶか上回る（[Fantastic](../sources/fantastic-pretraining-optimizers.md)、[Volkova ら](../sources/optimizer-scaling-laws.md)、[NVIDIA](../sources/soap-muon-and-beyond.md)）。[Semenov ら](../sources/benchmarking-optimizers.md)は、長い学習で AdEMAMix と MARS を一番良いとしていて、Fantastic と食い違う。原因は決着していない。[AlgoPerf](../sources/algoperf.md) の調整ありの部門は、行列を使う Distributed Shampoo が優勝した。[modded-nanogpt の最適化の部門](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/modded-nanogpt-optimization-track.md)は、ステップ数で SOAP-Muon が Muon より少ない。ただし時間は問わない比較。
- **学習率：** Fantastic は、ピーク学習率6e-4の AdamW を基準として弱いとする。nanoGPT も同じ6e-4。同じ更新サイズの1.3億の調整例では0.008だった。モデルの形が違うので、そのまま当てはまるとは限らない。
- **費用：** 幅768、更新1回524,288トークンなら、Muon の計算の増分は上限0.7%（元の記事の計算）。メモリは AdamW より少ない。
- **Weight decay の大きさ：** [nanochat](../sources/nanochat-log.md) の一番良い値は幅512で約0.40、幅768で約0.22で、幅の2乗に反比例するとしている。幅640なら約0.3で、今の0.2は小さい側。ただし nanochat は学習が今より長く、autoresearch は幅512で0.2を使っている。
- **細かい調整の限界：** nanochat は、ある規模で調整した値が別の規模で悪くなるとしている。測るのも1回 $0.55 かかるので、読むだけでは決められない。

## 今の見方

基準（9,100万パラメータ、5億トークン、更新1回524,288トークン、学習量は約0.27倍）で最初に試すものとして、最初の束は Muon と AdamW の併用にした。

設定は autoresearch の値から始めた。ただし次の点で同じではない。

- 埋め込みと出力の学習率：[autoresearch](../sources/autoresearch.md) は別々の表に0.6と0.004を使う。最初の束は1つの表を共有するので、そのままは使えない。出力の0.004を使い、位置符号と正規化の重みにも同じ値を使った（分離の見方は embeddings.md）。[最初の Muon の記録](../sources/modded-nanogpt-muon-record.md)の0.0036に近い。
- 行列の学習率0.04：autoresearch と更新の回数とサイズは同じだが、幅が違い（512と768）、調整もしていない。nanochat の既定は0.02。最初の Muon の記録は、同じ形で学習が6.5倍長く、更新の大きさは今の値の約4分の1。
- Polar Express、NorMuon、cautious weight decay：autoresearch の値はこれらを含むので、最初はすべて使う。設定でそれぞれ外せるので、結果が悪いときは順に確かめられる。
- Q、K、V：モデルは1つの行列のまま変えず、オプティマイザの中で3つに分けて直交化する。基準の重みはそのまま読める。

そう見ている理由は次の通り。

- 比べられた中で一番少ない学習量1倍で、Muon が先頭だった。ただし0.27倍は比較の範囲の外で、外挿になる。
- 計算とメモリの費用が小さい。
- autoresearch が同じ953回の更新で動かしている。他の変更も入っているので、Muon だけの効果は分からない。

効果の粗い見積もりは、学習率の調整で最大2倍、Muon でさらに1.4倍、合わせて最大2.8倍分。計算量2倍で loss が約3.9%下がるという換算（[Open Athena の記事](https://www.openathena.ai/blog/pretraining-speedup/)にある Delphi の式 `loss = 1.65 + 117.8·C^−0.0966`）で、約6%になる。別々の条件の数字を掛けたもので、順番を決めるための目安にすぎない。Polar Express の論文の比較では13〜15%下がっていて、学習量が少ないと見積もりより大きい可能性がある。ただし更新のサイズが今の約16分の1で、weight decay もない。最初の束は全体で18.5%下がり、この粗い見積もりより論文の比較に近かった。ただし Muon 以外の変更と、基準の AdamW の弱い設定の分も含む。

## 見方が変わりそうな条件

- 自分たちの測定で、見積もりと大きく違った場合。特に、調整した AdamW に Muon の束が勝てない場合。
- 学習量を増やす場合。SOAP や Kron のほうが良くなる可能性がある。
- 更新1回のトークン数を大きく減らす場合。Semenov らの AdEMAMix や MARS を見直す。
- SOAP-Muon などを、時間まで含めて今の規模で比べた結果が出た場合。
- モデルを大きくする場合。Fantastic では差が薄れ、NVIDIA ではバッチを大きくすると差が出ている。

## 関係するもの

- `language-model` のメモ：[NorMuon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/normuon.md)、[Muon+](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muon-plus.md)、[MuonEq](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/muoneq.md)、[SOAP-Muon](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/soap-muon.md)、[IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md)。
