# FlexAttention と FlashAttention-4

PyTorch の記事 [FlexAttention + FlashAttention-4: Fast and Flexible](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)（更新 2026-03-04）。

- FlexAttention に FlashAttention-4 のバックエンド（`kernel_options={"BACKEND": "FLASH"}`）を足した。Triton の実装の1.2〜3.2倍速い。
- Triton の FlexAttention は、Hopper で FlashAttention-3 の約60%の処理量（当初は約80%）。FlashAttention-3 を Hopper の最高水準の実装と呼んでいる。
- GB200 の因果マスクでは、FlexAttention の FLASH バックエンドは、FlashAttention-4 に組み込みの因果の経路より順伝播が遅い。
- H200 の数字は FlexAttention でしか書けない形（ALiBi、文書マスク、スライドウィンドウ）で、Triton との比較。順伝播1.30〜1.89倍、逆伝播1.35〜2.01倍。FlashAttention-3 との直接の比較はない。
- Llama 3 70B、H100 64枚、長さ8192の学習で、Triton と同じ損失に収束した。
- 開発中で、破壊的な変更がありうるとしている。

図の中の数字は読んでいない。
