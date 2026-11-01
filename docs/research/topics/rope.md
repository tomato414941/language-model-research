# RoPE

位置を、学習する埋め込みではなく、query と key の回転で伝える。基準の nanoGPT は学習する位置埋め込みを使っている。2026-11時点では入れるかどうかを決めていない。

- [RoFormer の論文](../sources/roformer.md)には、GPT の形で学習する位置埋め込みと比べた結果がない。今の形で効くかの直接の根拠は見つけていない。
- autoresearch、nanochat、多くのフロンティアのモデルは使っている。文脈を変えても位置の表が要らない。
- 一部の次元だけに回転をかける形は、`language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/partial-rope.md)にある。Marin では約0.1%、nanochat では改善なし。[Marin の記事](../sources/marin-moe.md)の Partial Key Offset は、効果の半分がこの形から来るとしている。Qwen 3.6 が回転させるのは次元の25%という記事があるが、この比較の元の研究は読んでいない。

## 見方が変わりそうな条件

- 同じ形で学習する位置埋め込みと比べた結果が見つかった場合。
- 文脈を長くする候補を進める場合。
