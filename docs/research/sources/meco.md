# MeCo

Gao, Wettig, He, Chen, Metadata Conditioning Accelerates Language Model Pre-training (arXiv 2501.01956, ICML 2025).

For the first 90% of training, the URL's domain (for example, `en.wikipedia.org`) is put at the head of each document, and the last 10% trains without it. No loss is applied to the metadata tokens.

## Settings

- A Llama-style model, Llama-3's tokenizer, DCLM-Baseline. 600 million, 1.6 billion and 3 billion on 160 billion tokens, and 8 billion on 80 billion tokens.
- Packed into a context of 8,192. AdamW (β 0.9, 0.95), a peak learning rate of 3e-3 (5e-4 at 8 billion), 5% warmup and then cosine down to 10% of the peak. Weight decay 0.033 (0.1 at 8 billion), 4 million tokens per update.
- One run each. The noise across three runs continued from one intermediate checkpoint is ±0.1 in the mean score (Table 13).

## Results

| Size | Plain | MeCo |
|---|---|---|
| 600 million | 51.5 | 51.7 |
| 1.6 billion | 55.7 | 56.7 |
| 3 billion | 59.2 | 59.8 |
| 8 billion | 57.8 | 58.4 |

- Mean over 10 tasks (Table 17). MeCo at 1.6 billion matches the plain model trained on 240 billion tokens (56.7), which is the basis for "33% less data" (Table 1).
- Perplexity at 1.6 billion is 13.2 for plain and 13.3 for MeCo (Table 1). The paper says perplexity does not correlate with the downstream scores.
- The authors say the gain appears larger for larger models, but the trend is inconsistent.

## Kinds of metadata (1.6 billion, Table 5)

Domain 56.7, full URL 56.8, the URL's suffix (such as org) 56.2, only the top 0.2% of URLs 56.4, only the top 2% 56.3, hashed URLs 56.7, model-generated topics 56.6. Since hashing gives the same result, it says what helps is the grouping of documents rather than the meaning. Generating the topics took about 1,500 GPU hours.

## The final phase

- Keeping the metadata to the end drops the score to 50.3 when the model is used without it (Table 4).
- A final phase of 10% gives 56.7, 20% gives 56.8 and 30% gives 56.2 (Table 14).
- Training on a mix of 90% documents with URLs and 10% plain documents gives 56.4, lower than training in two stages.

The HTML main text and tables were read through a summary.
