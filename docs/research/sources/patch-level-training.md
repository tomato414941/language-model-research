# The scale sections of Patch-Level Training

Shao et al. (arXiv 2407.12665). The method and the main tables are in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/patch-level-training.md); this note records only what is not there.

- The output layer stays a single layer and predicts the K tokens of the next patch at once.
- The patch stage consumes data quickly, so it needs a lot of data.
- The advantage shrinks as the model gets larger and grows faster as the data increases (Tables 3 and 4).
- The smallest comparison is 370 million with 360 billion tokens, about 970 tokens per parameter.

The text of the method and of Sections 3.3 and 3.4 was read. The numbers in Tables 3 and 4 were not read.
