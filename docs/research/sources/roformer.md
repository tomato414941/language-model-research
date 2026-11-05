# RoFormer

Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv 2104.09864).

- It rotates pairs of query and key dimensions by an angle proportional to the position. The score depends only on the relative distance, and there are no learned parameters.
- Machine translation (WMT) BLEU goes from 27.3 to 27.5. On GLUE it wins 3 and loses 3, with QQP from 71.2 to 86.4 and MNLI from 84.6 to 80.2. On long Chinese texts: BERT-512 67.77, RoFormer-512 68.29, RoFormer-1024 69.79.
- There is no direct comparison with learned position embeddings in a GPT architecture. Neither the variance nor the cost is shown. The authors say they cannot explain why it works.

The main text and the experiment tables were read.
