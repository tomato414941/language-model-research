# Composer

Composer: A Search Framework for Hybrid Neural Architecture Design (arXiv 2510.00379, ICLR 2026). It searches for an arrangement of attention (A) and MLP (M) on small models, scales it up, and compares it with Llama 3.2 of the same width.

## Setup

- The width is the same as Llama 3.2's (at 1 billion: 2,048 and 8,192, 32 heads, 8 KV heads). There are fewer layers than Llama 3.2's 32 (16 blocks): 27 for the stacked form and 29 for the stretched form (§4.2). How the parameter counts are matched is not stated.
- The form found by the 6-layer search is 2A+4M (§4). It is written as counts only, and the order of attention and MLP is not given. Layers of the same kind are placed one after another (§2.4).
- The forms at 1 billion are 4×(2A+4M)+1A+2M for the stacked form and 4A+9M+4A+5M+2A+5M for the stretched form (Appendix B.1).
- The comparison at 1 billion uses 37.5 billion tokens, about 37 tokens per parameter (Appendix C.5).

## Results

- 1 billion (Table 2): Llama 3.2 has a loss of 2.80 and a mean score of 58.69. The stacked form has 2.77 and 60.83. The stretched form has 2.77 and 60.76.
- Mean score (Table 6): at 350 million, the stacked form gets 56.0 and the stretched form 56.7 against Llama 3.2's 55.4. At 3 billion, 61.0 and 62.0 against 59.2. The smaller the model, the smaller the gap.
- Searching at 100 million to 200 million parameters without narrowing the width, the best form was 3A+3M (1:1). 1:2 was found by a search with a narrowed width, and was better than 1:1 when scaled up (§3.3). There are no numbers for a comparison that changes only the ratio.
- Training is about 1.25× faster and inference about 1.33× faster, and the KV cache shrinks to about 1/1.69.

Read: the relevant parts of §2.4, §3.1–3.3, §4.2, Tables 2 and 6, and Appendices B.1, B.4, B.5 and C.5 of the arXiv HTML version (through summaries and quotes extracted with a tool).
