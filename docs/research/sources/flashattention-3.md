# The FlashAttention-3 paper

Shah et al., FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (arXiv 2407.08608).

- FlashAttention-2 reaches only 35% utilization on H100.
- FlashAttention-3 uses the H100's new instructions and makes attention 1.5–2× faster in FP16. Up to 740 TFLOPs/s (75% utilization). In FP8 it approaches about 1.2 PFLOPs/s.
- The numbers are for the attention computation alone; the speed of training as a whole is not shown.
- The authors leave understanding the effect of low-precision attention on large-scale training to future work.

The abstract, the introduction, the conclusion, and the headings and figure captions of the experiments section were read. The details of the algorithm were not read.
