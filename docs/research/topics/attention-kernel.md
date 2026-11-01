# 注意の計算の方式

H100 で注意の計算を何で動かすか。2026-11時点で読んだものからの仮の見方で、測った結果ではない。

PyTorch 2.15 は、H100 では既定で cuDNN を先に試す（[読んだコード](../sources/pytorch-sdpa.md)）。基準の学習も同じ版なので cuDNN だった可能性が高いが、どれが動いたかの記録はない。nanochat の「FlashAttention-3 で学習が約9%速い。注意の計算だけなら FlashAttention-2 の2倍」（[LOG](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md)、2026-01-11）は FlashAttention-2 からの比較で、今の基準にはそのまま当てはまらない。H100 で cuDNN と FlashAttention-3、4 を比べた数字は見つけていない。

| 方式 | 入れ方 | 読んで分かったこと |
|---|---|---|
| cuDNN（`sdpa`） | PyTorch の既定 | 追加なし |
| FlashAttention-2（`fa2`） | cuDNN を切る | 追加なし。[FlashAttention-3 の論文](../sources/flashattention-3.md)の比較相手 |
| FlashAttention-3（`fa3`） | [`kernels-community/flash-attn3`](../sources/hf-kernels.md) を直接呼ぶ | 約550MB を取得する。PyTorch 2.15 に合うのは stable ABI のビルドだけ。論文では FlashAttention-2 の1.5〜2倍で、H100 向けの実装 |
| FlashAttention-4（`fa4`） | [`flash-attn-4`](../sources/flashattention-4.md) を PyTorch の呼び出しの裏に入れる | ベータ。主な対象は Blackwell。直接呼ぶと `torch.compile` のグラフが切れる（[TabPFN](../sources/tabpfn-fa4.md)）ので、PyTorch の入口を使う |

使わない経路は次の通り。

- **`varunneal/flash-attention-3`：** nanochat と autoresearch が使っていて、nanochat は H100 ではこちらが良いとしている（差の数字はない）。Torch 2.12 までで、`model` 型なので今の `kernels` では読めない。
- **PyTorch の FlashAttention-3 への差し替え：** `torch.ops.flash_attn_3` を探すので、ソースからのビルドが要る（30〜60分）。`kernels-community` のビルドは名前が違う。
- **[FlexAttention](../sources/flexattention-flashattention-4.md) の FLASH バックエンド：** 独自の注意のためのもので、因果の注意では FlashAttention-4 の組み込みより遅い。

## 今の見方

読むだけでは決まらないので、順位をつけずに短い計測で比べる（`speed.py`）。1方式の計測は1分ほどで、約$0.04。同じ入力で PyTorch の既定と答えが合わないものは使わない（`attention_check.py`）。

## 見方が変わりそうな条件

- 実測の結果。PyTorch や cuDNN の版が変わると順位も変わりうる。
- GPU を Blackwell に変える場合。FlashAttention-3 の kernel がない。
- 文脈を伸ばす、スライドウィンドウを試す場合。nanochat の LOG は、PyTorch の呼び出しでは窓の効果を得られないとしていて、FlashAttention-3 は `window_size` で扱える。
- `kernels` のビルドがなくなる場合、FlashAttention-4 の正式版が出る場合。
