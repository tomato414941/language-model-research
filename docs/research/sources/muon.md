# Muon

Keller Jordan et al., Muon: An optimizer for hidden layers in neural networks (blog post, 2024-12-08, https://kellerjordan.github.io/posts/muon/). It has not been made into a paper.

- For 2D weights, it applies 5 Newton-Schulz iterations to the SGD-momentum update to bring it close to orthogonal (corresponding to UV^T of the singular value decomposition). The coefficients are (3.4445, −4.7750, 2.0315), and it runs in bfloat16.
- Scalars, vectors, and the input and output layers are updated with AdamW or similar. For the embeddings this can be explained by the theory of the modular norm; for the output layer it is done on experimental grounds.
- In experiments, Nesterov-style momentum is slightly better, and applying it to Q, K and V as separate matrices is better.
- The upper bound on the extra compute is T·m/B (T is the number of iterations, m the width, B the tokens per update): 0.7% for the nanoGPT record (width 768, 524,288 tokens) and 0.5% for Llama 405B. The memory is the same as SGD-momentum's.
- Results: it made the record for the time to reach validation loss 3.28 on FineWeb (NanoGPT speedrun) 1.35× faster. At the time of the post, the 12 records since then were set by 7 people, and all of them use Muon. Reaching 94% on CIFAR-10 went from 3.3 seconds to 2.6 seconds on an A100. Bringing 1.5 billion parameters to GPT-2 XL's level on HellaSwag took 10 hours on 8×H100 (AdamW 13.3 hours).
- On why it works, it stops at a guess: Transformer updates are skewed toward a few directions (nearly low-rank), and orthogonalization may enlarge the weak directions.
- Arguing that many papers tune their AdamW baselines too little, it asks for evidence in the form of competitive records. With a record, if AdamW were faster, switching back to it would by itself set a new record.
- Open questions at the time of the post: whether it works at 20 billion parameters or more, whether it can be distributed, and whether it works beyond pretraining.

The whole main text was read. The figures were not looked at.
