# Hugging Face's kernels

`kernels/src/kernels/variants.py`, `docs/source/migration.md`, `security.md` and `kernel-requirements.md` in `huggingface/kernels`, and the FlashAttention-3 builds on the Hub (read on 2026-11-01). A library that loads compiled computations from the Hub.

- How builds are chosen: a build tagged with a Torch version can be used only when its major.minor matches the environment's Torch. A stable ABI build (such as `torch-stable-abi29`) can be used when the environment's Torch is that version or later. For CUDA, a build can be used when the major is the same and the minor is at or below the environment's.
- From 0.14 it loads only repositories of type `kernel`, not those of type `model`. It says type `model` is to be withdrawn from 2026-09-13.
- Because it loads remote code, by default it allows only code from trusted publishers. `kernels-community` is one of them. Builds carry a sigstore signature and a record of the source commit.
- It recommends writing kernels so that `torch.compile` does not break the graph.
- `kernels-community/flash-attn3` (type `kernel`, BSD-3-Clause):
  - version 2 has only stable ABI 2.9 builds for cu126, cu128, cu130 and cu134. The cu130 `.so` is about 550MB. version 1 has stable ABI 2.9 builds for cu126, cu128 and cu130 and builds specific to Torch 2.10–2.12, and its cu130 `.so` is about 821MB.
  - The target GPUs are 8.0 and 9.0a. bf16, fp16, fp8. The head size is rounded up to 64, 96, 128, 192 or 256.
  - The definition of `flash_attn_func` is the same in both versions. The shape is (B, T, H, D), and there is no dropout argument. There is a `fake` function for `torch.compile`.
- `varunneal/flash-attention-3` (type `model`): according to the README, Torch 2.8–2.12, CUDA 12.6–13.0, and bf16 and fp8 only. nanochat, autoresearch and modded-nanogpt use it.

Read: the relevant parts of the code and the documents, the file listing from the Hub's API, and the Python parts of the builds. The `.so` files were not run.
