# PyTorch の注意の計算の選び方

PyTorch 2.15.0 の `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`、`torch/nn/attention/`（`__init__.py`、`_registry.py`、`_fa3.py`、`_fa4.py`）、`torch/backends/cuda/__init__.py`、2.13〜2.15 のリリースノート、cu132 の wheel のメタデータ（2026-11-01 に読んだ）。

- `scaled_dot_product_attention` は、優先順に条件を確かめて、最初に通った実装を使う。実装は cuDNN、FlashAttention、省メモリ、数式。
- CUDA 13 以降のビルドでは、実行時の cuDNN が 9.15.0 より新しく、GPU の compute capability の major が 9 か 10 なら、cuDNN を先頭にする（`check_prefer_cudnn_attention`）。環境変数 `TORCH_CUDNN_SDPA_DEPRIORITIZED` で外せる。コメントには、CUDA 12.9 以前の wheel の cuDNN は古いとある。
- `torch-2.15.0+cu132` の wheel は `nvidia-cudnn-cu13==9.26.0.51` を要求する。
- 2.15 のリリースノートは、sm86 と sm89 でも cuDNN を優先するようにしたとする。読んだタグのコードの条件は major 9 と 10 だけで、合っていない。
- `torch.backends.cuda.enable_cudnn_sdp(False)` で cuDNN を外せる。内蔵の FlashAttention は2系（`_get_flash_version` は 2.5.7）。
- `torch.nn.attention.activate_flash_attention_impl("FA3")`、`("FA4")` で、FlashAttention の呼び出しを差し替えられる。`_fa3.py` の冒頭に PROTOTYPE、`_fa4.py` に UBER PROTOTYPE とある。
  - FA4 は `flash_attn.cute.interface`（`flash-attn-4`）を読み込む。major 9 と 10 だけで、head は256次元まで、dropout なし。
  - FA3 は `flash_attn_interface` を読み込み、`torch.ops.flash_attn_3` を探す。
- 2.15 のリリースノートに、cuDNN が Hopper の因果でない注意で NaN の勾配を返す問題の修正がある。

読んだのは上に挙げた部分。`sdp_utils.cpp` は優先順、cuDNN の条件の一部、FA4 の条件。GPU では動かしていない。
