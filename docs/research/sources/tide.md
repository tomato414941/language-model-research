# TIDE

Jaiswal et al. (Apple), TIDE: Every Layer Knows the Token Beneath the Context (arXiv 2605.06216).

- A standard model uses the token ID only once, at the input. TIDE keeps K per-token tables; at each layer it mixes them with weights computed from the hidden state and adds the result to the residual. The tables go through RMSNorm. No matrix multiplications are added.
- The LLaMA architecture, DCLM, vocabulary 128,256, context 2,048, 400 billion to 500 billion tokens, Adam.
- In the 1 billion model, raising K from 2 to 24 improves the perplexity on Wikitext, PubMed and DCLM monotonically. With K at 2 to 4, it reaches the baseline's perplexity at 200 billion tokens in 100 billion tokens.
- Most of the effect is on rare tokens. At K=24, the loss on rare tokens goes from 6.671 to 6.250, while that on frequent tokens falls by 0.075. K=2 captures about 55% of the improvement on rare tokens.
- In the 750 million model, LAMBADA perplexity goes from 5.63 to 5.18 (K=8), and the benchmark mean from 59.7 to 60.7.
- Because the tables get large, it says they are quantized and kept on SSD at inference. There are no results with Muon or at 100 million or below.

The abstract, the introduction, the method in Section 3, the start of Section 4, Table 2's rows for 750 million and Table 7 were read.
