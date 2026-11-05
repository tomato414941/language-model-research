# Qwenputer

Divyansh Agrawal's blog post (2026-10-22, https://divagr.com/blog/qwenputer).

- It adds 11 layers and extra dimensions in front of Qwen 3.5 4B, and a compiler sets their weights to build a RISC-V CPU (RV32IM). The original weights are unchanged. It uses Gated DeltaNet's state as exact RAM, with gate 1, beta 1 and one-hot keys.
- On the benchmarks, multiplication (mul_big) goes from 55% to 97.5% (100% with a Python tool), division (div_big) from 72.5% to 97.5% (95% with the tool), and NUPA from 48.6% to 100% (100% with the tool).
- No mismatch with the reference machine over about 230,000 output positions. 18 million instructions were executed without error. On everyday text, the KL is 6e-4 (noise level 5e-4).
- The weights are 18.8 GB, about twice the original. Starting it needs a dedicated driver, and it is slow, at about 3,200 CPU tokens per second on an L4.

The whole post was read.
