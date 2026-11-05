# FlashAttention-4

Zadouri et al., FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling (arXiv 2603.05451), with the README of `Dao-AILab/flash-attention`, `flash_attn/cute/interface.py` and `flash-attn-4` on PyPI (read on 2026-11-01).

- The paper targets Blackwell (B200, GB200). In bf16 on B200 it is up to 1.3× cuDNN 9.13 and 2.7× Triton, up to 1613 TFLOPs/s (71% utilization). It describes FlashAttention-3 as mainly for H100.
- It is written in CuTeDSL (Python), and it says compilation is 20–30× faster than with C++.
- It installs with `pip install flash-attn-4`. `[cu13]` is recommended for CUDA 13. At the time of reading it was the 4.0.0b37 beta (2026-10-28). Its dependencies include `nvidia-cutlass-dsl`, `apache-tvm-ffi`, `torch-c-dlpack-ext` and `quack-kernels`.
- There are Hopper kernels too (`flash_fwd_sm90.py`, `flash_bwd_sm90.py`). The entry point `flash_attn_func` uses `torch.autograd.Function`.

What was read had no numbers comparing it with FlashAttention-3 on H100. The paper's abstract, the README and part of `interface.py` were read. The paper's main text was not read.
