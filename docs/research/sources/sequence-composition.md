# Sequence Composition

Zhao et al., Analysing The Impact of Sequence Composition on Language Model Pre-Training (arXiv 2402.13991). It compares ways of packing documents, and attention across documents. It is not a comparison with starting partway through a row.

1.3 billion parameters, the LLaMA architecture, 150 billion tokens of SlimPajama. Test perplexity averaged over seven datasets (the loss differences are our own calculation from the averages):

| Packing | Context 2,048 | Difference | Context 8,192 | Difference |
|---|---|---|---|---|
| Pack random documents | 9.172 | | 9.065 | |
| Pack documents from the same source | 8.831 | −1.7% | 8.851 | −1.1% |
| Pack similar documents chosen with BM25 | 8.550 | −3.2% | 8.289 | −4.1% |
| Compute attention only within documents | 8.410 | −3.9% | | |

- It says attention across documents gets in the way. Attention only within documents costs 4% more compute (in the authors' implementation).
- It says the BM25 approach does not change the attention computation, so it does not lower training efficiency.
- It says downstream tasks improve as well.

The introduction, the start of Section 3, Table 1's rows at context 2,048 and part of those at context 8,192 were read. The within-document row at context 8,192 could not be read.
