# Ceramic's post on LayerNorm

Costello et al., [Revisiting LayerNorm: aka Norms are Important](https://www.ceramic.ai/blog/revisiting-layernorm) (2025-05-08).

- At about 2 billion parameters, it compares LayerNorm, one-pass LayerNorm and RMSNorm, and the evaluation loss is 3.0–3.15 for all of them.
- In plain PyTorch the two differ in speed by up to 40%, but as Triton kernels they are almost the same, and it says the difference disappeared.

Whether they have weights, and where they are placed, is not stated. Nor are the data and the number of tokens. The detailed numbers are in a separate table, which was not read. Read: the body through a summarizing tool.
