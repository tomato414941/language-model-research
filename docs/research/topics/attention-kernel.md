# 注意の計算の方式

H100 で注意の計算を何で動かすか。2026-11時点の、[最初の束の実測](../experiments/first-bundle.md)と読んだものからの見方で、測ったのは1台の機械、1つのモデルの形（10層・幅640、文脈1,024）だけ。

## 測った結果

1回の更新の差は1.6%以内で、FlashAttention-3 が一番遅かった。ほかに差が出たのは、形が新しいときの最初の1回の準備で、FlashAttention-2 と 4 は既定の方式と FlashAttention-3 より約23秒短い。新しい機械で1回だけ学習するなら時間の約3%で、換算で BPB が約0.2%下がる程度。同じ機械の同じ形の2回目では、既定の方式でも短くなった。

| 方式 | 入れ方 | 1回の更新 | 更新の外 |
|---|---|---|---|
| PyTorch の既定（`sdpa`） | そのまま | 0.557秒 | 34秒 |
| FlashAttention-2（`fa2`） | cuDNN を切る | 0.558秒 | 11秒 |
| FlashAttention-3（`fa3`） | [`kernels-community/flash-attn3`](../sources/hf-kernels.md) を直接呼ぶ。約550MB を取得する | 0.566秒 | 33秒 |
| FlashAttention-4（`fa4`） | [`flash-attn-4`](../sources/flashattention-4.md) を PyTorch の呼び出しの裏に入れる。ベータ | 0.558秒 | 11秒 |

PyTorch 2.15 の既定は、[読んだコード](../sources/pytorch-sdpa.md)では H100 で cuDNN を先に試す。この環境の cuDNN は 9.26 で条件に合うが、実際に cuDNN が動いたことは直接は確かめていない。cuDNN を切ると準備の時間が消えるので、cuDNN が形ごとに作るものの時間と見ているが、これも確かめていない。

[2番目の束](../experiments/second-bundle.md)の形でも、既定の方式と FlashAttention-2 の1回の更新は同じ（0.639秒）で、更新の外は44秒と12秒だった。

## 読んだものとの違い

[FlashAttention-3 の論文](../sources/flashattention-3.md)の注意の計算だけで1.5〜2倍、nanochat の学習で約9%（[LOG](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md)、2026-01-11）という数字は、FlashAttention-2 からの比較。ここでは FlashAttention-2 と同じか少し遅かった。計算の式の上で注意は約19%なので、差が出るとしても全体では小さい。読み込んだビルド、CUDA 13.2、文脈の短さ、compile との相性のどれが効いているかは調べていない。

## 今の見方

追加のインストールが要らない FlashAttention-2 が一番手軽で、差は小さい。大きな改善の候補としては扱わない。形や文脈を変えるたびに、`speed.py` の短い計測で確かめれば足りる。

使わなかった経路は次の通り。

- **`varunneal/flash-attention-3`：** nanochat と autoresearch が使っていて、nanochat は H100 ではこちらが良いとしている（差の数字はない）。Torch 2.12 までで、`model` 型なので今の `kernels` では読めない。
- **PyTorch の FlashAttention-3 への差し替え：** `torch.ops.flash_attn_3` を探すので、ソースからのビルドが要る（30〜60分、[TabPFN](../sources/tabpfn-fa4.md)）。`kernels-community` のビルドは名前が違う。
- **[FlexAttention](../sources/flexattention-flashattention-4.md) の FLASH バックエンド：** 独自の注意のためのもので、因果の注意では FlashAttention-4 の組み込みより遅い。

## 見方が変わりそうな条件

- 文脈を大きく伸ばす場合。計算に占める注意の割合が増える。スライドウィンドウを試す場合は、nanochat の LOG によると PyTorch の呼び出しでは効果を得られず、FlashAttention-3 は `window_size` で扱える。
- ヘッドの数を増やす、層の幅を変える場合。
- PyTorch や cuDNN の版が変わる場合。
- GPU を Blackwell に変える場合。FlashAttention-3 の kernel がない。
