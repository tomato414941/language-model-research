# Gated DeltaNet

Yang et al. (arXiv 2412.06464, 2024-12). It adds to linear attention a gate that erases memory and the delta rule, which rewrites memory selectively.

- At 1.3B, with 100 billion tokens of FineWeb-Edu and AdamW, it says it beats Mamba2 and DeltaNet on several tasks.
- Training speed is about the same as DeltaNet's and a little slower than Mamba2's. Within a 2,000-token context, a Transformer with FlashAttention-2 is the fastest, and hybrids with short-window attention fall in between.

The abstract and the section on training speed were read. The main body of the experiments was not read.
