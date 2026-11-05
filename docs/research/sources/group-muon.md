# Group Muon

Zhang, Zhou, Chen, Cheng, When and Why Grouping Attention Heads Accelerates Muon Optimization (arXiv 2605.08933, 2026-05).

It splits the rows of the attention matrices into groups of heads and orthogonalizes each group separately. The group size and the grouping (adjacent heads, spaced-out heads, random each time) are treated as settings. It presents the view that finer splits make each update lower the loss more, but also make the update larger, which costs.

## Settings

- A GPT-2 Small-style model (12 layers, width 768, twelve 64-dimensional heads, shared embedding and output, RoPE, RMSNorm) in modded-nanogpt's form. FineWeb, context 1,024, 524,288 tokens per update, 6,200 updates (about 3.25 billion tokens).
- Muon (Newton-Schulz, momentum 0.95) with no weight decay. The learning rate is constant and falls linearly over the last 1,800 updates. No warmup.
- Split matrices use a rule that transfers the learning rate according to their shape. It does not say the learning rate was searched for each grouping. The number of seeds is not stated either.
- The baseline orthogonalizes Q, K and V kept as one matrix.

## Results (final validation loss)

- Q, K and V kept as one: 3.2780. Split fully by head: 3.2799 (+0.06%).
- When only Q and K are grouped, forming random groups of six heads each time is the best, at 3.2722 (−0.18%). Grouping V as well brings no clear gain.
- Groups of one head should give the same update whatever the grouping, yet they range from 3.2759 to 3.2787, a spread of about 0.003.
- Splitting by head falls faster at first but is later overtaken by the unsplit form (Figure 1). Smaller groups are better early, and larger groups are better later.
- According to the paper, GLM-5 uses Muon split by head.

The HTML main text and Appendices C and D were read. The proofs (Appendix B) were not read.
