# Infini-gram

Liu et al., Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens (arXiv 2401.17377, 2024).

## Results

- **Agreement with the next token (§4.1):** the ∞-gram, which uses the longest matching history, agrees with the next token of human-written text 47% of the time. When the matched length is 16 or more, it exceeds 75%. An ordinary n-gram with a small n gets 29% (introduction). The count of the matched history has little effect on the agreement rate (Appendix C).
- **Mixing with a model (§5.2, Table 1):** with the Pile's training data as the reference, Llama-2 70B's perplexity falls by 12%. Adding RedPajama as well gives 18–19%. For GPT-2 1.6B, 33–34%.
- **A model trained on the same data as the reference:** GPT-Neo 1.3B, trained on the Pile, gets 16%, less than GPT-2's 34%. The authors attribute this to GPT-Neo having been trained on the same Pile as the reference.
- **Overlap with the evaluation:** documents that share 80% or more of their 13-grams with the evaluation text were removed from the reference corpus (0.6% of Pile-train, Appendix B).

Read: the relevant parts of §4.1, §5.2 and Appendices B–D of the HTML version, from quotes extracted with a tool. The body as a whole was not read.
