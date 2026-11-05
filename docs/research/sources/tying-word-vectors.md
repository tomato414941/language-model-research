# Tying Word Vectors and Word Classifiers

Inan et al., Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (arXiv 1611.01462, ICLR 2017). LSTM language models. It derives the use of the input embedding table for the output as well from a loss framework.

Comparison adding only the tying (test perplexity):

| Model | PTB | Wikitext-2 |
|---|---|---|
| 200 units | 87.3 to 85.1 | 105.9 to 100.5 |
| 650 units | 77.7 to 73.9 | 95.3 to 87.7 |

Both are small datasets, in a setting where tying cuts the number of parameters sharply.

The abstract, the introduction and Table 1 were read.
