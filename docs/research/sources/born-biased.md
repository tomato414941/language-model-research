# Transformers Are Born Biased

Transformers Are Born Biased: Structural Inductive Biases at Random Initialization and Their Practical Consequences (arXiv 2602.05927).

- A model at its random initialization strongly prefers particular tokens. In GPT-2's shape at 1.2 billion, the most preferred token takes 10.2% at a context of 1,024. The causes are the MLPs' asymmetric activations and attention's aggregation, and it happens with the MLPs alone as well.
- The bias of each seed weakens but remains after training. The paper uses it as the model's fingerprint.
- The initialization is GPT-2's, with the output weights at 0.02/√(2×number of layers). There is no comparison with the output weights set to 0.

Sections 1, 3 and 4.1, the start of section 5, 6.2.2, section 7 and the appendix's part on initialization were read.
