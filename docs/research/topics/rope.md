# RoPE

位置を、学習する埋め込みではなく、query と key の回転で伝える。基準の nanoGPT は学習する位置埋め込みを使っている。

- **直接の比較：** [EleutherAI の記事](../sources/rotary-embeddings-blog.md)では、GPT の形で学習する位置埋め込みと比べて、検証の損失が1.25億で −1.8%、14億で −3.0%。14億は1パラメータ約6トークンで、今の規模（約14トークン）に近い。2021年の記事で、RoPE を広めた人たちの比較で、種の数も分からない。最適化は Muon ではない。
- **論文：** [RoFormer の論文](../sources/roformer.md)には、GPT の形で比べた結果がない。
- **速度記録：** [modded-nanogpt の記録](../sources/modded-nanogpt-records.md)の2番は、学習率の調整と合わせて1.43倍分で、分けた数字はない。
- **費用：** 計算が1〜3%増える（記事の値）。今の規模で約0.1%の損に相当する。位置の表（最初の束で約66万パラメータ）はなくなる。
- **採用：** autoresearch、nanochat、多くのフロンティアのモデルが使っている。
- **一部の次元だけに回転をかける形：** `language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/partial-rope.md)にある。Marin では約0.1%、nanochat では少し悪い。[Marin の記事](../sources/marin-moe.md)の Partial Key Offset は、効果の半分がこの形から来るとしている。Qwen 3.6 が回転させるのは次元の25%という記事があるが、この比較の元の研究は読んでいない。
- **代わりの方式：** 2026-11 の検索では、言語モデルの学習で RoPE を上回るという比較は見つけていない。RoPE を疑う[論文](../sources/rope-not-relative.md)はあるが、代わりは提案していない。

## 今の見方

2026-11 時点の見方では、2番目の束に入れる。見込みは約2%。

## 見方が変わりそうな条件

- Muon と今の形で、学習する位置埋め込みと比べた結果が見つかった場合、自分たちで測った場合。
- 文脈を長くする場合。文脈を伸ばしたときの扱いが問題になる。
