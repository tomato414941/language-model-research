# Weight Tying Biases Token Embeddings Towards the Output Space

Lopardo et al. (arXiv 2603.26663, ACL 2026 Findings). It studies how sharing the embeddings changes the contents of the table, by comparing public models and by retraining.

- A shared table resembles the output table of a model with separate tables. With OLMo-1B, the cosine similarity is 0.719 with the output table and 0.525 with the input table.
- Early in training, most of the gradient to the shared table comes from the output side. In shared models the first few layers work poorly (the tuned lens KL is high).
- In a shared OLMo-1B, multiplying the input-side gradient by 5 (up to 20 billion tokens) moves the table slightly toward a separate model's input table (0.216 to 0.222) and away from the output table (0.384 to 0.369). Downstream evaluation does not consistently improve, and the authors call it a trade-off.
- In small models the table takes a large share of the parameters (73% in Pythia 70M), which is a reason to share, and it recommends separating above 1B. Qwen3 shares at 4B and below and separates at 8B and above. OLMo 2 and 3 separate at every size. Llama 3 and DeepSeek-V3 also separate.
- There is no comparison of loss between sharing and separating with everything else matched.

Read: the body. The appendices were not read.
