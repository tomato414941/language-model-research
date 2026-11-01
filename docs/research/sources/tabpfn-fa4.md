# TabPFN の FlashAttention-4 への置き換え

`PriorLabs/TabPFN` の [Pull Request #1291](https://github.com/PriorLabs/TabPFN/pull/1291)。表データのモデルの FlashAttention-3 を FlashAttention-4 に替える提案で、2026-11-01 の時点では未マージ。

- H100 の測定：FlashAttention-4 は全体で4〜10%速い（10,000行以上。それ未満は同じ）と書いたあと、再測定では FlashAttention-3 が速い場合も出ている（10万行で57ミリ秒と68ミリ秒など）。
- FlashAttention-4 の `flash_attn_func` は `torch.compile` でトレースできず、呼び出しごとにグラフが切れる。FlashAttention-3 はできる。この分で約4%遅くなり、それでも SDPA の約1.17倍。
- FlashAttention-3 はソースからのビルドに30〜60分かかり、FlashAttention-4 は pip で入る。

言語モデルの学習の数字ではない。読んだのは要約ツールを通した説明と会話で、元の表は確かめていない。
