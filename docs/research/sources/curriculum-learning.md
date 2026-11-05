# The conditions of the studies on the order in which documents are read

Zhang et al. (arXiv 2506.11300) and arXiv 2601.21698. The main figures are in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/curriculum-learning.md); only what is not there is written here.

- **Zhang et al.:** The evaluation is the mean accuracy over 8 benchmarks; there are no loss or BPB results. Loss is not used, on the grounds that it captures generalization poorly. The main comparison is 500 million parameters on 10 billion tokens, 20 tokens per parameter. It says the scores that set the order are cheap to compute and do not add to the training compute.
- **2601.21698:** It trained 14 million to 1 billion parameters on 300 billion tokens. A random order has large gradient noise; the difference is strong at 14 million–70 million and small for larger models. Reversing the order loses most of the accuracy benefit.

The main text of Zhang et al.'s results and conclusion, and the abstract and introduction of 2601.21698, were read.
