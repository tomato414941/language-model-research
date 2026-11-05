# The size of COUNT's table, and evaluation questions in ClimbMix

On 2026-11-03, to check whether COUNT could be added, we counted 2 billion tokens of ClimbMix's training data on the processing machine (CPU, free). Evaluation questions were found in 0.114% of the documents, and removing them did not change the gap with v6's English data. With the top 8 candidates and a minimum of 4 occurrences, the table comes to about 5.4GB. A table built from the text that training reads predicts that text better than it really can. The third bundle's table was built from 2 billion tokens of FineWeb and came to 5.3GB. Its hit rate on ClimbMix's validation text is lower than that of the table built from ClimbMix.

## Evaluation questions

We searched with the same rules as v6 in `language-model`. Text is NFKC-normalized, lowercased and stripped of symbols, and a question counts as found when its middle 64 characters (all of it if shorter) appear in a document. Questions under 24 characters cannot be checked. MMLU's opening instruction was left out. All 2,683,950 documents were checked.

| Task | Questions | Questions checkable | Questions found | Of these, answer also in the document |
|---|---|---|---|---|
| MMLU | 14,042 | 13,685 | 284 | 129 |
| ARC-Easy | 2,376 | 2,357 | 54 | 34 |
| ARC-Challenge | 1,172 | 1,158 | 21 | 13 |
| HellaSwag | 10,042 | 10,042 | 53 | 22 |
| PIQA | 1,838 | 1,166 | 23 | 2 |
| SciQ | 1,000 | 1,000 | 43 | 38 |
| Total | 30,470 | 29,408 | 478 | 238 |

- 3,071 documents (0.114%) contain questions. This is about the same as the share v6 removed from its English data (0.17–0.31%), but v6 checks more tasks.
- The same question can appear in many documents. 326 questions appear in only one document, 72 in six or more, and the most in 813 documents.
- "Answer also in the document" means the answer string appears somewhere in the document. Short answers like "4" can match by chance. Restricted to answers of 8 characters or more, there are 186 questions.
- Only exact matches are searched for, so paraphrases are not found.

## Effect on the scores

The two models of v6-english-data.md were compared again on the same questions. The mean over 6 tasks, with equal task weights.

| Questions | Trained on ClimbMix | Trained on v6's English data | Difference |
|---|---|---|---|
| All | .4521 | .4269 | +.0252 ± .0035 |
| Excluding the questions found | .4521 | .4272 | +.0249 ± .0035 |
| Only the questions found | .4772 | .4453 | +.0319 ± .0266 |

Excluding the questions found does not change the difference. The difference on the found questions alone is also about the same within the error, so there is no sign that training on ClimbMix gives particularly high scores on the found questions. The cause of the score drop after switching the data is not exactly matching evaluation questions.

## The size of the table

We kept only 1/16 of the history hashes, counted them exactly and multiplied by 16. Histories are 1–5 tokens, and BOS can only be at the start of a history. The size is rows × (2 bytes × history length + 3 bytes × K), calculated for fixed-width rows. Filling candidates from shorter histories is not included.

| Minimum occurrences of a history | K=8 | K=4 |
|---|---|---|
| 2 | 15.57GB | 9.78GB |
| 3 | 8.00GB | 5.00GB |
| 4 | 5.38GB | 3.35GB |
| 6 | 3.24GB | 2.01GB |
| 8 | 2.32GB | 1.43GB |
| 16 | 1.08GB | 0.66GB |
| 32 | 0.51GB | 0.31GB |
| 64 | 0.25GB | 0.15GB |

Hit rates at a minimum of 4. The validation text is the first 16 million tokens of `val.bin`; the answer text is the tokens of the 6 tasks' correct answers, given as the range over tasks.

| History length | Rows | Found in the validation text | Next token in the top 8 | Found in the answer text |
|---|---|---|---|---|
| 1 | About 8,000 | 100% | 41.6% | 100% |
| 2 | 8.51 million | 99.0% | 51.0% | 99–100% |
| 3 | 48 million | 83.6% | 45.6% | 65–85% |
| 4 | 65.15 million | 51.8% | 28.7% | 40–52% |
| 5 | 47.63 million | 25.5% | 14.6% | 7–24% |

The share found at different minimums (validation text).

| Minimum | History 3 | History 4 | History 5 |
|---|---|---|---|
| 2 | 87.9% | 59.4% | 32.2% |
| 4 | 83.6% | 51.8% | 25.5% |
| 8 | 78.7% | 44.5% | 20.0% |
| 16 | 72.9% | 37.6% | 15.5% |

## Overlap between the table's text and the training text

When the text that training reads is counted into the table, the table knows that text's continuations. How large the difference is was measured with tables that counted the first N tokens of the training data (1/16 of the hashes). The text looked up was the first 16 million tokens of the counted text, the following 16 million tokens that were not counted, and the first 16 million tokens of `val.bin`.

With the table of 2 billion tokens and a minimum of 4, the share where the next token is in the top 8 when a row is found.

| History length | Counted text | Validation text |
|---|---|---|
| 3 | 65.9% | 54.5% |
| 4 | 76.2% | 55.5% |
| 5 | 83.4% | 57.3% |

- On the counted text, the table looks more reliable the longer the history. On the validation text, it barely changes with length.
- As a share of all positions, at history 5: counted text 24.2%, validation text 14.6%.
- Raising the minimum narrows the gap. At history 5, a minimum of 16 gives 71.5% and 60.9%, and a minimum of 64 gives 65.8% and 61.9%.
- With tables of 250 million, 500 million and 1 billion tokens, the difference between the uncounted continuation and the validation text was within 0.3 points for histories 3–5 at every minimum. In terms of hit rates, the uncounted part of the training data can be treated the same as the validation text.
- The counted text is 16 million contiguous tokens, not the very rows starting at document heads that training reads.

## How much text to count

On the validation text, the share where the next token is in the top 8 (over all positions). Tables of about the same size were chosen.

| Amount counted | Minimum | Size (K=8) | History 3 | History 4 | History 5 |
|---|---|---|---|---|---|
| 250 million | 2 | 2.19GB | 35.9% | 18.9% | 8.3% |
| 500 million | 3 | 2.11GB | 38.9% | 21.3% | 9.6% |
| 1 billion | 4 | 2.75GB | 42.0% | 24.3% | 11.4% |
| 1 billion | 6 | 1.64GB | 40.6% | 22.4% | 10.1% |
| 2 billion | 8 | 2.32GB | 43.4% | 25.3% | 11.9% |

- At the same size, each doubling of the text raises it by 2–3 points at history 3, about 2 points at history 4 and about 1 point at history 5. The gains shrink gradually.
- Raising the minimum in proportion to the text (from 2 at 250 million to 16 at 2 billion) shrinks the table from 2.19GB to 1.08GB and raises history 3 from 35.9% to 40.4%.
- Only whether the next token is in the top 8 is looked at; the precision of the scores was not measured.
- One validation shard is 63.37 million tokens, so ClimbMix as a whole comes to about 410 billion tokens by calculation. The training data is about the first 32 shards, about 0.5% of the whole.

## Tables built from text other than ClimbMix

Tables were built the same way from 250 million tokens of FineWeb (the first file of the 10BT sample, revision `9bb295d`) and from the first 250 million tokens of ClimbMix's training data, and compared. With the baseline's tokenizer, FineWeb has 3.70 bytes per token, fewer than ClimbMix's 4.02.

At a minimum of 2 (about 2.1–2.2GB), the share where the next token is in the top 8 (over all positions). From the left, the preceding 3, 4 and 5 tokens.

| Text looked up | ClimbMix's table | FineWeb's table |
|---|---|---|
| ClimbMix's validation | 35.9 / 18.9 / 8.3% | 29.6 / 13.3 / 4.9% |
| v6's English data, stable-phase validation | 31.0 / 13.9 / 5.9% | 22.9 / 9.8 / 3.8% |
| v6's English data, decay-phase validation | 31.7 / 13.6 / 6.4% | 20.0 / 8.7 / 3.2% |
| FineWeb's continuation | 29.5 / 13.3 / 4.9% | 35.0 / 18.0 / 7.9% |

- Each table hits best on text from its own data. Their hit rates on each other's text are about the same (29.6% and 29.5%).
- On v6's English data, which is neither, ClimbMix's table hits more. The reasons were not separated.
- The text looked up was 16 million tokens for ClimbMix and FineWeb, and all of the validation data for v6 (about 5.9 million and about 2.4 million). The order is the same with minimums of 1, 4 and 8.
- Converting FineWeb into 266 million tokens took 161 seconds. 2 billion tokens would take about 20 minutes by calculation, about 3 files of the 10BT sample (about 2.1GB per file).

## The table built from 2 billion tokens of FineWeb

The table that was to be used in the third bundle. On 2026-11-04 the tokenizer was retrained on FineWeb, so the table was rebuilt too (training-data.md). The numbers below are with the tokenizer trained on ClimbMix. It was built with `count_data.py` and `count_table.py` on the processing machine (for the time, see "Time and cost" below).

- **Text:** 2 billion tokens, from reading 2,410,588 documents from the start of the 10BT sample (revision `9bb295d`) and removing the 1,123 documents (0.047%) that contain evaluation questions. 3.70 bytes per token. The share of documents containing questions is less than half of ClimbMix's (0.114%).
- **Table:** a minimum of 4 occurrences per history, the top 8 candidates. 5.29GB, 165.17 million rows.

The first roughly 16 million tokens of `val.bin` were cut into rows of length 1,024 and looked up the same way the model looks them up. The same text as for the ClimbMix table above.

| History length | Rows | Found | Next token among the candidates | Among them when found |
|---|---|---|---|---|
| 0 | 1 | 100% | 16.0% | 16.0% |
| 1 | 8,126 | 99.0% | 35.1% | 35.5% |
| 2 | 8.42 million | 96.4% | 46.3% | 48.1% |
| 3 | 47.62 million | 77.2% | 38.4% | 49.8% |
| 4 | 64 million | 42.7% | 21.0% | 49.2% |
| 5 | 45.12 million | 18.0% | 9.0% | 49.9% |

- The next token is among the candidates of some length at 56.9% of positions.
- Compared with ClimbMix's 2-billion-token table (minimum 4), at histories 3–5 the share found is 6–9 points lower, the share among the candidates 6–8 points lower, and the share among them when found 5–7 points lower. The share among the candidates is 16% lower at history 3 and 38% lower at history 5, about the same as in the comparison at 250 million tokens (18–41%).
- **Paragraph breaks:** FineWeb's text separates paragraphs with a single newline, and no token contains two newlines. In ClimbMix's validation text, 0.95% of tokens contain two newlines. Histories containing them are not in the table, and they are never candidates for the next token either. Positions with such histories are 2.8% at history 3 and 4.7% at history 5, and part of the gap in the share found comes from this. FineWeb's text does not distinguish paragraph breaks from line breaks, so substitution cannot make them match.
- Loading took 11 seconds, with peak memory of 5.6GB. Looking up 65,536 positions on the CPU took 0.07 seconds.

## Time and cost

- Searching for evaluation questions took about 14 minutes with 4 processes. Counting 1/16 of the table took about 14 minutes on 1 thread. It ran on the processing machine, so there was no extra cost.
- The full table is counted in 16 partitions. One partition of 2 billion tokens measured 174 seconds at order 6 (140 seconds of it computing hashes), 118 seconds at order 3, and peak memory of about 11GB (including loading the token file). Memory allows only one at a time, so 16 partitions were estimated at about 3 hours, but it actually took about 1.6 hours (below). Most of the time goes to computing the hashes of all positions in every partition, so changing how the partitions are split would likely shorten it further.
- Building from 2 billion tokens of FineWeb, preparing the text (downloading, searching for evaluation questions, tokenizing) took about 34 minutes with 4 processes, and counting about 1.6 hours. On disk, the tokens take 4.0GB and the table 5.3GB.
- The overlap check took about 13 minutes, running 4 at once.
- Transferring 5.3GB at the last speed (3.7GB in 164 seconds) takes about 4 minutes, about $0.16. It is paid on every training run.

## Not verified

- How much effect COUNT has with a 2-billion-token table. The paper's table is built from about 6 trillion tokens.
- How much the overlap between the table and the training text moves the scores. What was measured is the difference in the table's hit rates, not the model's scores.
- How much the scores differ between the table built from FineWeb and the one built from ClimbMix.
- The hit rates of tables counted from more than 2 billion tokens. The tables above are measured up to 2 billion.
- The number of paraphrased evaluation questions.
- The number of evaluation questions in the unread part of the 400B.
