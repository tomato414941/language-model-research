# FlexAttention and FlashAttention-4

PyTorch's post [FlexAttention + FlashAttention-4: Fast and Flexible](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) (updated 2026-03-04).

- It added a FlashAttention-4 backend (`kernel_options={"BACKEND": "FLASH"}`) to FlexAttention. It is 1.2–3.2× faster than the Triton implementation.
- Triton's FlexAttention reaches about 60% of FlashAttention-3's throughput on Hopper (about 80% at first). It calls FlashAttention-3 the state-of-the-art implementation on Hopper.
- With a causal mask on GB200, FlexAttention's FLASH backend has a slower forward pass than FlashAttention-4's built-in causal path.
- The H200 figures are for forms that only FlexAttention can express (ALiBi, document masks, sliding windows), compared with Triton: forward 1.30–1.89×, backward 1.35–2.01×. There is no direct comparison with FlashAttention-3.
- In training Llama 3 70B on 64 H100s at length 8192, it converged to the same loss as Triton.
- It says it is under development and may have breaking changes.

The numbers inside the figures were not read.
