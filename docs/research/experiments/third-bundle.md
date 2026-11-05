# Training the third bundle on FineWeb

On 2026-11-04, with the training data switched to FineWeb, the second and third bundles were each trained once at the same cost. With the third bundle, validation BPB read from document starts fell by 1.05%, and the mean accuracy over the 6 tasks rose from .418 to .422. Correct-answer BPB rose by 0.5% (got worse). The differences in accuracy and in correct-answer BPB are both within the movement across seeds. The weights made by mixing the last weights with their average were worse than the last weights.

## Setup

- Machine: one H100 80GB (HBM3) on Vast.ai in Japan, $2.41/hour. Driver 615.71.09.
- Image: the same as the baseline's (torch 2.15.0+cu132). cuDNN 9.26.
- Data: FineWeb's 10BT sample, and a tokenizer with a vocabulary of 8,192 trained on it (training-data.md). The data and the tokenizer differ, so the scores are not compared with those on ClimbMix.
- Second bundle: `configs/bundle2.py` (second-bundle.md). 54.4 million parameters used in computation.
- Third bundle: the second bundle plus 8 items: COUNT, Dispersion Loss, mixing the last weights with their average (TSA), tables of token tuples, attention gates, shifting the keys, MUDD and backout (speedrun-parts.md). 54.48 million parameters used in computation. COUNT's table was built from 2 billion tokens of FineWeb that training does not read. Settings in `configs/bundle3.py`.
- The same cost: the baseline was run briefly on the same machine, and the number of updates was set from the time per update and the time outside updates (`speed.py`). Both bundles use FlashAttention-2, values flowing through the layers in bf16, and 4 passes of 128 rows added up. One update took 0.816 seconds for the baseline, 0.604 for the second bundle and 0.830 for the third, giving 1,275 updates for the second bundle and 910 for the third. 524,288 tokens per update.
- Of the third bundle's slower update (+37% from the second bundle), removing COUNT gives 0.704 seconds, and removing Dispersion Loss 0.782 seconds.
- One seed (1337).
- Scoring: accuracy and correct-answer BPB on all questions of the 6 tasks (30,470). The mixed weights were scored too.

## Scores

| | Second bundle | Third bundle | Third bundle (mixed weights) | Second to third |
|---|---|---|---|---|
| Mean accuracy over the 6 tasks | .4179 | .4223 | .4194 | +.0044 ± .0032 |
| MMLU | .2551 | .2605 | .2595 | +.0054 ± .0030 |
| ARC-Easy | .3472 | .3506 | .3451 | +.0034 ± .0081 |
| ARC-Challenge | .2176 | .2261 | .2244 | +.0085 ± .0096 |
| HellaSwag | .2962 | .3014 | .2996 | +.0053 ± .0030 |
| PIQA | .6153 | .6202 | .6170 | +.0049 ± .0089 |
| SciQ | .776 | .775 | .771 | −.0010 ± .0104 |
| Correct-answer BPB (mean of 6) | 1.5576 | 1.5660 | 1.5842 | +0.54% (+.0084 ± .0022) |
| MMLU | 1.829 | 1.865 | 1.877 | +2.0% |
| ARC-Easy | 1.769 | 1.768 | 1.797 | −0.1% |
| ARC-Challenge | 1.758 | 1.776 | 1.790 | +1.0% |
| HellaSwag | 1.059 | 1.075 | 1.075 | +1.5% |
| PIQA | 1.389 | 1.381 | 1.378 | −0.6% |
| SciQ | 1.542 | 1.532 | 1.588 | −0.7% |
| Validation BPB (`scoring.py`, fixed-position rows) | 1.2146 | 1.1633 | 1.1634 | −4.2% |
| Validation loss (`train.py`, random positions) | 3.2442 | 3.1000 | | −4.4% |
| Final training loss (training data, from document starts) | 2.9986 | 2.9627 | | −1.2% |

± is the standard error of the differences on the same questions. On ClimbMix, the difference between the baseline's two seeds was +.002 ± .003 in mean accuracy and 1.3% in mean correct-answer BPB (harness-milestone.md).

- Accuracy rose on 5 of the 6, but the difference in the mean is 1.4× the standard error and cannot be told apart from the movement across seeds.
- Correct-answer BPB rose on MMLU, ARC-Challenge and HellaSwag, and fell on PIQA and SciQ. The difference in the mean is smaller than the movement across seeds.
- The mixed weights were worse than the last weights, by −.0029 ± .0011 in mean accuracy and +1.2% in correct-answer BPB. The mixing happens after training, so it does not affect the last weights.

## Validation BPB split by how rows are cut

After training, the saved last weights were measured again on 128 rows of validation data each, on the processing machine (CPU, float32), in two ways: rows at the same fixed positions as in `scoring.py`, and non-overlapping rows starting at document starts. Losses are in nats per token.

| Rows | | Second bundle | Third bundle | Change |
|---|---|---|---|---|
| Fixed positions | BPB | 1.1988 | 1.1416 | −4.8% |
| | First 64 tokens of a row (6%) | 5.136 | 4.067 | −20.8% |
| | Rest of the row | 3.090 | 2.997 | −3.0% |
| From document starts | BPB | 1.1342 | 1.1223 | −1.05% |
| | First 64 tokens of a row | 3.343 | 3.246 | −2.9% |
| | Rest of the row | 3.021 | 2.994 | −0.9% |

- Most of the difference on the fixed-position rows comes from the beginning of rows that start mid-document. The second bundle badly mispredicts this part (second-bundle.md), and the third bundle mispredicts it less. Even so, it is 0.8 nats per token worse there than on rows starting at document starts.
- The difference on rows starting at document starts (−1.05%) is close to the difference in the final training loss (−1.2%). It is larger than the movement of validation BPB across seeds (about 0.3%). The error from the choice of rows was not computed.
- With the conversion of about 3.9% per doubling of compute, this corresponds to about 1.2× the compute (our own calculation).

## Decision

As of 2026-11-04, the third bundle, without the mixing, becomes the base of the next bundle.

- Validation BPB read from document starts fell by more than the movement across seeds. Accuracy rose on 5 of the 6, though by small differences.
- Correct-answer BPB went the other way, but the difference is smaller than the movement across seeds. On MMLU and HellaSwag, it rose clearly beyond the standard error of the differences on the same questions.
- As the reading decided beforehand (weight-averaging.md) prescribes, the mixing is removed, since it was worse than the last weights.
- The differences are small, and there is one seed. This is to be revisited with the next bundle's results or with measurements that add seeds. The weights of the second and third bundles are kept.

## Not verified

- Which items worked. Apart from the mixing, the effects were not separated one by one.
- Why correct-answer BPB rose on MMLU and HellaSwag.
- How the scores on FineWeb move when the seed changes. It is estimated from the baseline's two seeds on ClimbMix.
- Whether the strength on rows starting mid-document remains when the way training reads rows is changed.

## Time and cost

| Stage | Measured |
|---|---|
| From renting to ready | 37 s |
| Transferring code and data (4.0GB besides the table, and the table's 5.3GB, relayed from the processing machine) | 5 s, 187 s, 232 s |
| Adding packages | 49 s |
| Speed measurement (9 runs) | 10 min 5 s |
| Second bundle: training, scoring, 6 tasks | 13 min 7 s, 21 s, 2 min 28 s |
| Third bundle: training, scoring, 6 tasks | 13 min 4 s, 23 s, 2 min 31 s |
| Mixed weights: scoring, 6 tasks | 23 s, 2 min 32 s |
| Fetching the results | 69 s |

About $2.3 for the roughly 56 minutes of the rental (calculated from time and price). Training was about $1.05 of it, for the two runs. Building the data and the analysis were done on the processing machine, so there was no extra cost.
