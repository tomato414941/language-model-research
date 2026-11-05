# Rethinking Embedding Coupling

Chung et al., Rethinking embedding coupling in pre-trained language models (arXiv 2010.12821). A comparison of separating the input and output tables in multilingual pretrained models (BERT-shaped).

- Just separating them without changing their shapes takes the downstream task average from 62.3 to 62.7 (with the parameters going from 177 million to 269 million). It wins on some tasks and loses on others, and the authors say the difference is not large.
- The advantage is that the shapes can be set separately. The input table can be made smaller, and a larger output table improves both the pretraining task and downstream results.

The comparison is of evaluations after fine-tuning, not of pretraining loss. Read: the abstract, the introduction, the start of Section 4, and Table 2.
