# Hugging Face の kernels

`huggingface/kernels` の `kernels/src/kernels/variants.py`、`docs/source/migration.md`、`security.md`、`kernel-requirements.md` と、Hub にある FlashAttention-3 のビルド（2026-11-01 に読んだ）。コンパイル済みの計算を Hub から読み込むライブラリ。

- ビルドの選び方：Torch の版が付いたビルドは、環境の Torch と major.minor が一致しないと使えない。stable ABI のビルド（`torch-stable-abi29` など）は、環境の Torch がその版以上なら使える。CUDA は major が同じで、minor が環境以下なら使える。
- 0.14 から `kernel` 型のリポジトリだけを読み込み、`model` 型は読み込まない。`model` 型は 2026-09-13 から取り下げるとしている。
- リモートのコードを読み込むので、既定では信頼された公開者のものだけを許す。`kernels-community` はこれに入る。ビルドには sigstore の署名と、元のコミットの記録がある。
- `torch.compile` でグラフが切れないように書くことを勧めている。
- `kernels-community/flash-attn3`（`kernel` 型、BSD-3-Clause）：
  - version 2 は stable ABI 2.9 の cu126、cu128、cu130、cu134 だけ。cu130 の `.so` は約550MB。version 1 は stable ABI 2.9 の cu126、cu128、cu130 と Torch 2.10〜2.12 専用で、cu130 の `.so` は約821MB。
  - 対象の GPU は 8.0 と 9.0a。bf16、fp16、fp8。head は64、96、128、192、256 に切り上げる。
  - `flash_attn_func` の定義は両方の版で同じ。形は (B, T, H, D) で、dropout の引数はない。`torch.compile` 向けの `fake` の関数がある。
- `varunneal/flash-attention-3`（`model` 型）：README によると Torch 2.8〜2.12、CUDA 12.6〜13.0、bf16 と fp8 だけ。nanochat、autoresearch、modded-nanogpt が使っている。

読んだのはコードと文書の該当部分、Hub の API のファイル一覧、ビルドの Python の部分。`.so` は動かしていない。
