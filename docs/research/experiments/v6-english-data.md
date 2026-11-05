# Trained the second bundle's form on v6's English data

On 2026-11-02, the same form as the second bundle was trained once on `language-model`'s v6 English data instead of ClimbMix, and measured on every question of the 6 tasks. The mean accuracy over the 6 tasks fell from .452 to .427 (difference paired by question −.025 ± .004; ± is the standard error, here and below). The data alone moves it this much. Which difference mattered was not separated.

## How to read the results, decided before seeing them

If the mean falls by .01 or less, the difference the data makes is small. If it falls by more, the data matters. This time it fell by more than .01, so the reading is that the data matters.

## Conditions

- Model and training: the same as the second bundle (second-bundle.md). 54.4 million parameters used in the computation, 1,281 updates of 524,288 tokens each, seed 1337, and the same learning rates, untuned. The settings are in `configs/bundle2_v6.py`. The second bundle's number of updates was used as is, and the speed was not measured again.
- Machine: one H100 SXM 80GB in Japan on Vast.ai, $2.41/hour. Driver 615.71.09. The image is the same as the baseline's.
- Data: `train_en.bin` and `validation_en.bin` of `corpus-stable` and `corpus-decay` from v6's corpus. They were read read-only from R2, and their checksums and document counts were confirmed to match v6's records. The patch corpus was not read.
- Conversion (`prepare_packed.py`): each document was turned back into text with v6's tokenizer and re-tokenized with the baseline's tokenizer (vocabulary 8,192, trained on ClimbMix), and the order of the documents was shuffled. Changing the vocabulary would change the model's form, so v6's tokenizer (vocabulary 32,768) was not used. For the first 70,000 documents of the files, turning them back into text and re-tokenizing them with v6's tokenizer matched the original tokens exactly.
- Reading: at the 641st update, where the learning rate starts to fall, the data was switched from the stable-phase data to the decay-phase data. v6's settings also switch the mix where the learning rate starts to fall. The tokens read are split half and half. v6 is 80:20 overall, but 40:60 counting each token only once, and half and half lies between the two.
- Scoring: every question (30,470) of the 6 tasks in the harness 0.4.13, bf16. MMLU in its authors' format was not measured. The scores of the models compared against are from harness-milestone.md.

## Scores

Accuracy on the 6 tasks. Each difference is this model minus the other, paired by question.

| Task | Second bundle | Trained on v6 English | Difference from the second bundle | v5 | Difference from v5 |
|---|---|---|---|---|---|
| MMLU (continuation form) | .264 | .259 | −.005 ± .003 | .274 | −.015 ± .003 |
| ARC-Easy | .439 | .420 | −.019 ± .009 | .454 | −.034 ± .009 |
| ARC-Challenge | .247 | .232 | −.015 ± .010 | .263 | −.031 ± .011 |
| HellaSwag | .308 | .294 | −.014 ± .004 | .347 | −.053 ± .004 |
| PIQA | .635 | .567 | −.067 ± .010 | .611 | −.044 ± .010 |
| SciQ | .820 | .788 | −.032 ± .011 | .818 | −.030 ± .012 |
| Mean | .452 | .427 | −.025 ± .004 | .461 | −.034 ± .004 |

- All 6 tasks fell. The drops clearly larger than the error are PIQA, HellaSwag and SciQ.
- The mean of the 3 tasks used to choose ClimbMix's mix (PIQA, ARC-Easy, HellaSwag) was −.033, and that of the other 3 tasks −.017. With one measurement, this does not go as far as showing a difference.
- The mean correct-answer BPB rose from 1.342 to 1.460 (+8.8%). By task: MMLU +1.8%, ARC-Easy +5.8%, ARC-Challenge +6.3%, HellaSwag +13.2%, PIQA +14.6%, SciQ +14.2%. In this comparison, BPB and accuracy pointed the same way.
- The validation BPB on ClimbMix from `scoring.py` rose from 1.046 to 1.188. This validation text is from the data the second bundle trained on, not from the data this model trained on.

## Checking that training was not broken

Since the scores fell, both models were measured again after training on the processing machine (CPU, float32), on three validation texts. BPB over 96 rows that start at the heads of documents.

| Validation text | Second bundle | Trained on v6 English |
|---|---|---|
| ClimbMix | 0.954 | 1.088 |
| v6 stable phase | 1.090 | 0.982 |
| v6 decay phase | 1.074 | 0.792 |

Each model does better on the text closer to its own training data. The model trained on v6 English has indeed learned v6's text.

## Differences from ClimbMix

Candidate causes of the drop. None was measured separately.

| | ClimbMix | v6 stable phase | v6 decay phase |
|---|---|---|---|
| Documents | 2.68 million | 750,000 | 690,000 |
| Tokens (baseline tokenizer) | 2 billion | 660 million | 1.00 billion |
| Median document length | 617 | 463 | 940 |
| Rows read | 655,872 | 328,192 | 327,680 |
| Share of rereading the same document (calculated) | 11% | 19% | 20% |
| Bytes per token (sample of 4,000 rows) | 4.02 | 3.43 | 3.25 |

- Evaluation questions: v6 removes documents that contain the text of evaluation questions. In English, it removed 1,271 documents from the stable phase (0.17% of the documents kept) and 2,138 from the decay phase (0.31% of those kept). ClimbMix does not remove them, and how many it contains has not been counted.
- Mix: v6's English is 20% code and 16% math even in the stable phase. ClimbMix's mix was chosen by performance on the validation questions of 3 of the 6 tasks.
- Amount of text read: the baseline tokenizer was trained on ClimbMix, so on v6's English it gives fewer bytes per token. For the same number of tokens, about 17% less text is read. This difference is separate from the data's content, and which way and how far it moved the scores was not measured.
- Reading from the heads of documents: documents are chosen with equal probability, so sources with many short documents are read more. In v6's stable phase, 58% of row starts are FinePhrase, more than its share in the mix (30% by tokens). In the decay phase, general-llmjp is 18% (8% in the mix). v6 itself does not read this way.
- Rereading: rows are drawn from fewer documents, so the share of rereading the same document is higher.

## Time and cost

| Stage | Measured |
|---|---|
| Converting the data (the processing machine, free) | 25 min |
| From renting to ready to use | 59 s |
| Transferring the code and the data (3.7GB) | 5 s, 164 s |
| Adding packages | 48 s |
| Training | 823 s |
| Scoring with `scoring.py` | 18 s |
| Every question of the 6 tasks | 144 s |
| Fetching the results | 13 s |

The rental of about 23 minutes cost about $0.93 (calculated from the time and the rate), of which training was about $0.55. One update took 0.604 s, no different from the second bundle's 0.615 s. The training's wall time was longer than the second bundle's 796 s, but that is because the first update took 38 s. The second bundle came after the speed measurement, so for it this took only 6.6 s.

## What was learned

- Even with the same form, changing only the data moved the mean accuracy by .025. This is larger than the +.003 ± .003 from the first bundle to the second, and smaller than the +.060 from the baseline to the first bundle.
- The .452 from training on ClimbMix is not a score under the same conditions as v5, which was trained on data with the evaluation questions removed. The difference from v5 was −.009 with ClimbMix and −.034 with v6's English. But v5's and v6's data are not the same either.
- This comparison does not show that v6's data is bad. The amount of text read and reading from the heads of documents are conditions that differ from v6's own training, and which way they moved the scores was not measured. From this one run, it cannot be told whether the data is bad or not.

## Not checked

- Which of the candidates above mattered, and how much. For the evaluation questions, they were counted later, and removing exact matches was confirmed not to change the difference (count-table-check.md).
- The scores when using a tokenizer trained on v6's English, and when not reading from the heads of documents.
- Variation from the seed. The difference between the baseline's two seeds is +.002 ± .003 on the mean.
- Whether the learning rates suit v6's data.
