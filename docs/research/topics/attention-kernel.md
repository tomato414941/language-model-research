# Methods for computing attention

Which method to run attention with on the H100. A view as of 2026-11, from [the first bundle's measurements](../experiments/first-bundle.md) and what was read; only one machine and one model shape (10 layers, width 640, context 1,024) were measured.

## Measured results

The time per update differed by at most 1.6%, and FlashAttention-3 was the slowest. The other difference was in the one-time preparation the first time a shape is new: FlashAttention-2 and 4 take about 23 seconds less than the default method and FlashAttention-3. For a single training run on a new machine, that is about 3% of the time, which converts to BPB falling by about 0.2%. On a second run of the same shape on the same machine, the preparation became shorter with the default method too.

| Method | Setup | Per update | Outside updates |
|---|---|---|---|
| PyTorch's default (`sdpa`) | As is | 0.557 s | 34 s |
| FlashAttention-2 (`fa2`) | Turn off cuDNN | 0.558 s | 11 s |
| FlashAttention-3 (`fa3`) | Call [`kernels-community/flash-attn3`](../sources/hf-kernels.md) directly; fetches about 550MB | 0.566 s | 33 s |
| FlashAttention-4 (`fa4`) | Put [`flash-attn-4`](../sources/flashattention-4.md) behind PyTorch's call; beta | 0.558 s | 11 s |

PyTorch 2.15's default, according to [the code we read](../sources/pytorch-sdpa.md), tries cuDNN first on the H100. The cuDNN in this environment is 9.26 and meets the conditions, but whether cuDNN actually ran was not checked directly. Turning off cuDNN removes the preparation time, so we think it is the time cuDNN takes to build what it makes for each shape, but this was not checked either.

With the shape of [the second bundle](../experiments/second-bundle.md) too, the default method and FlashAttention-2 took the same time per update (0.639 seconds), and 44 and 12 seconds outside updates.

## Differences from what was read

The figures of 1.5–2× for the attention computation alone in [the FlashAttention-3 paper](../sources/flashattention-3.md) and about 9% for nanochat's training ([LOG](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md), 2026-01-11) are comparisons against FlashAttention-2. Here it was the same as FlashAttention-2 or slightly slower. By the formula for the computation, attention is about 19% of it, so even if a difference appeared, it would be small overall. Which of the loaded build, CUDA 13.2, the short context or the interaction with compile is responsible was not investigated.

## Current view

FlashAttention-2, which needs no extra installation, is the easiest, and the differences are small. It is not treated as a candidate for a large improvement. Each time the shape or the context changes, a short measurement with `speed.py` is enough to check.

The routes not used were these.

- **`varunneal/flash-attention-3`:** nanochat and autoresearch use it, and nanochat says it is the better one on the H100 (with no figure for the difference). It covers Torch only up to 2.12 and is of the `model` type, so the current `kernels` cannot load it.
- **PyTorch's switch to FlashAttention-3:** it looks for `torch.ops.flash_attn_3`, so it needs a build from source (30–60 minutes, [TabPFN](../sources/tabpfn-fa4.md)). The `kernels-community` build has a different name.
- **The FLASH backend of [FlexAttention](../sources/flexattention-flashattention-4.md):** it is meant for custom attention, and for causal attention it is slower than the built-in FlashAttention-4.

## Conditions likely to change the view

- When the context is lengthened greatly. Attention's share of the computation grows. When trying a sliding window, according to nanochat's LOG, the PyTorch call gives no gain, while FlashAttention-3 handles it with `window_size`.
- When the number of heads is increased or the layer width is changed.
- When the PyTorch or cuDNN version changes.
- When the GPU is changed to Blackwell. There is no FlashAttention-3 kernel for it.
