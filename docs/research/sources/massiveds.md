# MassiveDS

Shao et al., Scaling Retrieval-Based Language Models with a Trillion-Token Datastore (arXiv 2407.12854, 2024).

## The datastore

- About 1.4 trillion tokens. General web 1.1917 trillion, STEM 97.7 billion, code 52.8 billion, encyclopedias 31.9 billion, books 26.3 billion, Q&A forums 20.2 billion, math 14.1 billion, medicine 6.5 billion (§3.1, Table 2).
- The retriever is Contriever-MSMARCO (177 million parameters). With chunks of 256 words, the top 3 are placed before the context at inference. The models are Llama-2, Llama-3, Pythia and OLMo.

## Results

- Growing the datastore improves the language-model loss and several tasks monotonically, with no clear saturation. At the largest scale, Llama-2 7B with retrieval beats 13B without it. MMLU rises monotonically; MedQA is unclear (§4.2).
- At the same training compute, the models with retrieval are better (§4.3, Figure 4).
- A datastore that mixes domains matches or beats a datastore of that domain alone (§5.1, Table 3). Llama-2 7B without → with retrieval: TQA 64.1→77.0, NQ 26.6→34.6, MedQA 36.6→39.4, MMLU 45.8→49.3.
- Removing documents that overlap the evaluation texts changes the language-model loss a lot but the task scores little (§5.3, Figure 7). Dolma's quality filters have a limited effect.

The relevant sentences and tables of §3.1, §4.1–4.3, §5.1 and §5.3 of the HTML version were read, from quotations extracted by a tool. The full main text and the latter half of the appendix were not read.
