# 正規化

各層の入力の大きさを揃える計算。基準の nanoGPT は LayerNorm で、学習する重み（大きさ）を持つ。形と計算の候補は次の3つ。

- LayerNorm：平均を引いて、標準偏差で割る。
- RMSNorm：平均を引かず、2乗平均で割る。少し軽い。
- それぞれ学習する重みの有無。最初の束では重みは全部で13,440個で、計算にはほとんど関係しない。

## 分かっていること

- **LayerNorm と RMSNorm：** 品質はほぼ同じ（[RMSNorm の論文](../sources/rmsnorm.md)、[Ceramic](../sources/revisiting-layernorm.md)）。速さは、論文の Transformer で6.9%差があり、Ceramic では素直な PyTorch で最大40%、良いカーネルで0。正規化の計算だけの差か、全体の差かは資料ごとに違う。
- **重みの有無は割れている：**
  - 普通の設定では重みありが良い。[OLMoE](../sources/olmoe.md) は勾配の跳ねが減り、[Lingle の記事](../sources/parametric-rmsnorm-blog.md)は 2.240 と 2.253（0.6%差）。どちらも AdamW で、OLMoE は勾配をクリップしている。
  - `lm_head` の初期値を小さくする μP の設定では重みなしが良い（[μ-Transfer](../sources/mu-transfer.md)）。
- **小さい規模の例：** autoresearch、nanochat、modded-nanogpt（同じ流れのコード）はどれも重みなしの RMSNorm。LLaMA、Qwen、DeepSeek は重みあり（Lingle の記事による）。今の規模で、Muon と組み合わせて比べた結果は見つけていない。
- **FlashNorm（arXiv 2407.09577）：** 要旨によると推論を速くする話で、学習の候補ではない。

## 今の見方

2026-11時点の見方では、重みなしの RMSNorm にする。理由は次の通り。

- μP の設定で重みなしが良いという結果は、`lm_head` を小さく始める形のもの。2番目の束で埋め込みを分離するなら、この形に近くなる（embeddings.md）。
- 勾配の跳ねを問題にした報告は勾配のクリップの話で、Muon の束はクリップを使わない。
- autoresearch の学習率の組は、この形で動いている。

品質への効果は0付近（±0.6%くらい）で、速度が少し良くなる程度と見ている。大きな改善の候補ではなく、形を揃えるために入れる。

## 見方が変わりそうな条件

- 勾配が不安定になる場合。学習する重みを戻す。
- 今の規模で、重みの有無を比べた結果が出た場合。
