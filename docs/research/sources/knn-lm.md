# kNN-LM

Khandelwal et al., Generalization through Memorization: Nearest Neighbor Language Models (arXiv 1911.00172, ICLR 2020).

## Method

- Pairs of a context vector produced by a trained model and the word that follows it are stored in a datastore. At inference, it looks up nearby contexts and mixes the distribution of their next words with the model's distribution. The model is not retrained.
- The model has 16 layers, width 1,024 and FFN 4,096 (the shape of Baevski & Auli 2019). Except for WikiText-103, it uses BERT's vocabulary (about 29,000).
- The evaluation text is kept apart from the training text by article or by book.

## Results

- **Using the training data itself (§4.1):** WikiText-103 perplexity goes from 18.65 to 16.12, and Books from 11.89 to 10.89.
- **More text than the training data (§4.2, Table 3):** training on 100 million tokens (WIKI-100M) and using WIKI-3B, 30× that size (about 2.87 billion tokens), as the datastore takes test from 19.59 to 13.73. A model trained on all of WIKI-3B gets 15.17.
- **Datastore size (Figure 2):** a larger datastore improves results monotonically, without saturating even at about 3 billion. At 1.6 billion it already beats the model trained on all 3 billion. The best mixing weight rises as the datastore grows.
- **Another domain (Table 4):** adding a Books datastore to a model trained on Wikipedia takes Books test from 34.84 to 20.47. A model trained on Books gets 11.89.

Read: the description of the data in §3, §4.1, §4.2, Tables 1–4 and the caption of Figure 2 in the PDF.
