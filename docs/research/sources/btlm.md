# BTLM-3B-8K

Dey et al., BTLM-3B-8K: 7B Parameter Performance in a 3B Parameter Model (arXiv 2309.11568).

On activation functions, replacing GELU with SwiGLU lowered the loss by 1.37%. To keep the compute equal, the inner width goes from 4× to 8/3×. The model uses Cerebras-GPT's μP settings, with 20 tokens per parameter.

ReLU² is not compared. Only the sections on activation functions and positions were read.
