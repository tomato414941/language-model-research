# FlashAttention-4

Zadouri ら、FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling（arXiv 2603.05451）と、`Dao-AILab/flash-attention` の README、`flash_attn/cute/interface.py`、PyPI の `flash-attn-4`（2026-11-01 に読んだ）。

- 論文の対象は Blackwell（B200、GB200）。B200 の bf16 で cuDNN 9.13 の最大1.3倍、Triton の2.7倍、最大 1613 TFLOPs/s（使用率71%）。FlashAttention-3 は主に H100 向けとしている。
- CuTeDSL（Python）で書かれ、コンパイルは C++ より20〜30倍速いとしている。
- `pip install flash-attn-4` で入る。CUDA 13 では `[cu13]` を勧める。読んだ時点は 4.0.0b37（2026-10-28）のベータ。依存は `nvidia-cutlass-dsl`、`apache-tvm-ffi`、`torch-c-dlpack-ext`、`quack-kernels` など。
- Hopper の kernel もある（`flash_fwd_sm90.py`、`flash_bwd_sm90.py`）。入口の `flash_attn_func` は `torch.autograd.Function` を使う。

H100 で FlashAttention-3 と比べた数字は、読んだ範囲にはなかった。読んだのは論文の要旨、README、`interface.py` の一部。論文の本文は読んでいない。
