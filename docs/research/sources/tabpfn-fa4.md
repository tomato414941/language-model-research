# TabPFN's switch to FlashAttention-4

[Pull Request #1291](https://github.com/PriorLabs/TabPFN/pull/1291) of `PriorLabs/TabPFN`. A proposal to replace FlashAttention-3 with FlashAttention-4 in a model for tabular data, unmerged as of 2026-11-01.

- H100 measurements: after writing that FlashAttention-4 is 4–10% faster overall (at 10,000 rows or more; below that they are the same), a re-measurement also found cases where FlashAttention-3 is faster (such as 57 milliseconds and 68 milliseconds at 100,000 rows).
- FlashAttention-4's `flash_attn_func` cannot be traced by `torch.compile`, so the graph breaks at every call. FlashAttention-3 can be traced. This costs about 4%, and it is still about 1.17× SDPA.
- FlashAttention-3 takes 30–60 minutes to build from source, while FlashAttention-4 installs with pip.

These are not numbers from language-model training. The description and the discussion were read through a summarization tool, and the original tables were not checked.
