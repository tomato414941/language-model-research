# Training data

**Rule: from 2026-11-04, the training data is FineWeb.** The uniform 10B-token sample of FineWeb that modded-nanogpt reads (`sample/10BT` of `HuggingFaceFW/fineweb`, revision `9bb295d`). The license is ODC-By 1.0, and commercial use is allowed ([FineWeb](../sources/fineweb.md)).

- **Previous data:** ClimbMix was dropped because its source is non-commercial (climbmix.md). language-model's v6 Japanese-English data was also considered as a replacement, but FineWeb was chosen to keep handling simple. It is English only, so effects in Japanese cannot be measured with this data.
- **Tokenizer:** Keeps the vocabulary of 8,192 and is retrained on FineWeb's training text (`prepare.py`). The distributed GPT-2 tokens (vocabulary of about 50,000) are not used, since they would change the model's shape.
- **Split:** Of the 15 files, the last file goes to validation (20 million tokens), and training uses the files before it, going backward (2 billion tokens). The text for the COUNT tables is taken from the first file (`count_data.py`), so the tables do not count the training text.
- **Comparing scores:** The data and the tokenizer changed, so scores are not compared with those of the baseline and the bundles on ClimbMix. The second and third bundles were retrained on this new data and compared ([the third bundle](../experiments/third-bundle.md)).

## The data made (2026-11-04)

It took about 3 hours 11 minutes on the processing machine (CPU). No extra cost.

- **Training and validation:** 2 billion training tokens, 20 million validation tokens, vocabulary 8,192. Downloading, the tokenizer and tokenizing took 35 minutes.
- **COUNT text:** 2 billion tokens from 2,475,100 documents read from the first file, minus 1,157 documents (0.047%) that contain evaluation questions. For the same 2 billion tokens, 2.7% more documents are read than with the tokenizer trained on ClimbMix (2,410,588 documents). 43 minutes with 5 processes.
- **COUNT tables:** With a minimum count of 4 and the top 8 candidates, 165.91 million rows, 5.31GB. Rows for histories 1–5: 8,142, 9.27 million, 49.97 million, 63.74 million, 42.92 million. 1 hour 54 minutes on one thread.
- **Transfer time:** To the H100 in Japan, 187 seconds for the 4.0GB other than the tables, and 232 seconds for the tables.
