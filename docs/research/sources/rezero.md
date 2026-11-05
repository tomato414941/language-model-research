# ReZero

Bachlechner et al., ReZero is All You Need: Fast Convergence at Large Depth (arXiv 2003.04887).

Each layer becomes `x + α F(x)`, with the learned value α starting at 0. There is one α per layer, and LayerNorm is removed.

On enwiki8 (one character at a time), 12 layers, width 512, context 512, LAMB (constant learning rate 0.016), 20% dropout. The number of iterations until the validation BPB reaches 1.2:

| Form | Iterations |
|---|---|
| Post-Norm | Diverges |
| Post-Norm, with warm-up | 13,690 |
| Pre-Norm | 17,765 |
| GPT-2's Norm | 21,187 |
| ReZero, from α=1 | 14,506 |
| ReZero, from α=0 | 8,800 |

- In the same form, starting α at 0 instead of 1 takes about 0.6× the iterations.
- The final test BPB is 1.17 for every form, with no difference.
- What it compares are forms without normalization, not today's Pre-Norm form with a 0 put into it. It scales a whole layer by one value, which also differs from setting the weights themselves to 0.
- 64 and 128 layers can also be trained (the ordinary form diverges under the same conditions).

The abstract, the introduction, section 6, and Appendices B and C were read.
