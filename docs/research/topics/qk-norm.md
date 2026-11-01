# QK-norm

注意の内積の前に、Q と K を正規化する。主な目的は学習の安定。[`language-model` のメモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/qk-norm.md)にも短いまとめがある。

## 分かっていること

- **品質：** [IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md)（7,000万、AdamW）で学習の損失が −0.63%。[Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md) では 1.3160 から 1.3156 でほぼ0。[OLMoE](../sources/olmoe.md) は少し良くなったとし、[QuacK](../sources/quack.md) は学習率が小さいと差がほとんどないとしている。
- **安定：** [発散の比較](../sources/training-stability-methods.md)、OLMoE、QuacK はどれも効くとしている。70B の [Apertus](../sources/apertus.md) も安定のために入れているが、単独の比較は表にない。ただし最初の束は、QK-norm なしで学習率0.04のまま最後まで安定して学習した。
- **費用：** OLMoE で処理量が約10%落ち、QuacK でも QK-norm をなくすと約10%速い。今の換算で約0.6%の損で、品質の利点と同じくらい。計算の式の上ではごくわずかなので、効率の良い実装なら小さい可能性がある。autoresearch は QK-norm を入れて GPU の使用率が約39.8%で、入れていない最初の束の約39%と変わらない。ただし式に正規化は含まれず、活性化関数や正規化の違いも混ざっている。
- **代わり：** QuacK は学習率で抑える方法で、普通の注意では QK-norm に少し負ける。Muon は更新の大きさを揃えるので、そのまま当てはめられない（自分たちの読み）。

## 今の見方

2026-11時点の見方では、同じコストでの効果は0付近（−0.6%〜+0.5%）。2番目の束には入れる。先に H100 で速さだけを測る案もあったが、測らずに入れて進めることにした。

## 見方が変わりそうな条件

- 遅くなる量が分かった場合。5%より遅いなら、外すことを考える。5%は、品質の利点を中くらいの0.3%と見たときの損益の分かれ目（換算で約0.3%）。
- 学習が不安定になった場合。入れる理由が強くなる。
- 効果を項目ごとに測った場合。
