# Olmo Hybrid

The Olmo team (arXiv 2604.03444, 2026-04). It studies a hybrid of Gated DeltaNet and full attention in the form of published scaling laws.

- 7B on up to 6 trillion tokens. Compared with OLMo 3 7B, it needs 35% fewer tokens to reach the same Common Crawl loss, and 49% fewer for the same MMLU. The shape is adjusted to match OLMo 3 in parameter count and training speed.
- From the scaling laws, the token savings at the same size are about 1.3× at 1B, 1.68× at 7B, 1.82× at 30B and 1.89× at 70B.
- In a comparison at a nominal 100 million parameters (Math, Code and QA loss), the Transformer gets 0.981, 0.804 and 1.326, and Gated DeltaNet 3:1 gets 0.934, 0.752 and 1.271. However, the actual parameters are about 103 million and 130 million, so they are not matched. Mamba2 is about the same as the Transformer.

Read: the abstract, the introduction, and the descriptions of the main tables and figures. The full text was not read.
