# How PyTorch chooses its attention computation

`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, `torch/nn/attention/` (`__init__.py`, `_registry.py`, `_fa3.py`, `_fa4.py`) and `torch/backends/cuda/__init__.py` in PyTorch 2.15.0, the release notes for 2.13 to 2.15, and the metadata of the cu132 wheel (read on 2026-11-01).

- `scaled_dot_product_attention` checks the conditions in order of priority and uses the first implementation that passes. The implementations are cuDNN, FlashAttention, memory-efficient and math.
- In builds for CUDA 13 and later, when the runtime cuDNN is newer than 9.15.0 and the major of the GPU's compute capability is 9 or 10, cuDNN goes first (`check_prefer_cudnn_attention`). The environment variable `TORCH_CUDNN_SDPA_DEPRIORITIZED` turns this off. A comment says the cuDNN in the wheels for CUDA 12.9 and earlier is old.
- The `torch-2.15.0+cu132` wheel requires `nvidia-cudnn-cu13==9.26.0.51`.
- The 2.15 release notes say cuDNN is now preferred on sm86 and sm89 as well. The condition in the code at the tag read covers only majors 9 and 10, which does not match.
- `torch.backends.cuda.enable_cudnn_sdp(False)` takes cuDNN out. The built-in FlashAttention is the 2.x line (`_get_flash_version` gives 2.5.7).
- `torch.nn.attention.activate_flash_attention_impl("FA3")` and `("FA4")` swap out the FlashAttention call. The top of `_fa3.py` says PROTOTYPE, and that of `_fa4.py` says UBER PROTOTYPE.
  - FA4 loads `flash_attn.cute.interface` (`flash-attn-4`). Majors 9 and 10 only, heads of up to 256 dimensions, no dropout.
  - FA3 loads `flash_attn_interface` and looks for `torch.ops.flash_attn_3`.
- The 2.15 release notes include a fix for cuDNN returning NaN gradients for non-causal attention on Hopper.

Read: the parts listed above; in `sdp_utils.cpp`, the priority order, part of the cuDNN conditions and the FA4 conditions. Nothing was run on a GPU.
