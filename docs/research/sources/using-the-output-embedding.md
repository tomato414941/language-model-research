# Using the Output Embedding

Press, Wolf, Using the Output Embedding to Improve Language Models (arXiv 1608.05859, EACL 2017). One of the papers that recommended sharing the embeddings, on LSTM language models.

- The updates of the shared table are closer to those of the separate model's output table than to those of its input table.
- The test perplexity of the small model (LSTM 200, no dropout) goes from 114.5 to 112.4 on PTB, 195.3 to 187.1 on text8, 110.8 to 106.8 on BBC, and 94.3 to 94.8 on IMDB.
- On the training data it often does not improve (90.4 to 95.6 on text8), which the authors attribute to less overfitting.

The abstract, the introduction, the experiments section and Tables 5–7 were read.
