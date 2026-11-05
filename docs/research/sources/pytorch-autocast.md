# PyTorch's automatic 16-bit computation

What PyTorch's code says.

- The list that GPU autocast runs in 32 bits (`AT_FORALL_FP32`) includes `pow`, `layer_norm` and `rms_norm`. The list it runs in 16 bits includes `scaled_dot_product_attention`, among others.
- `square` calls `at::pow(self, 2)`.
- `rms_norm` raises the input to 32 bits internally for the computation and returns the same type as its input.

CPU autocast has separate lists; tried with torch 2.14.1, `rms_norm` and `square` given 16-bit inputs stay in 16 bits. The product of a 32-bit 0-dimensional value and a 16-bit array is 16-bit. The behavior of GPU autocast was not tried.

Read: the lists in `autocast_mode.h` in the torch 2.14.1 distribution, and `square` in `UnaryOps.cpp` and `rms_norm` in `layer_norm.cpp` at v2.9.0.
