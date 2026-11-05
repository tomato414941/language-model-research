# Training the second bundle on an H100

On 2026-11-01, the second bundle, which changes the model's shape, was trained once at the same cost as the baseline (baseline.md). Compared with the first bundle, the mean correct-answer BPB fell by 4.4%, and validation BPB as currently measured rose by 2.2%. The rise comes from how rows are cut: compared on rows read from document starts, it falls by 4.2%.

## Setup

- Machine: one H100 SXM 80GB on Vast.ai in Japan, $2.41/hour. Driver 615.71.09. A different machine from the baseline's.
- Image: the same as the baseline's (torch 2.15.0+cu132). cuDNN 9.26.
- Model: the first bundle (10 layers, width 640, 5 heads, context 1,024, vocabulary 8,192, Value embeddings) plus RoPE, RMSNorm without weights, QK-norm, ReLU², zero initialization of the outputs, untied embeddings, x0 and resid weights, and softcap 15. 54.4 million parameters used in computation, 85.85 million in total. Settings in `configs/bundle2.py`.
- Training: Muon and AdamW together. Learning rates untuned: 0.66 for the input table, 0.0044 for the prediction table, 0.04 for matrices, 0.5 for the x0 weights, 0.005 for the resid weights, 0.1 for Value embeddings. Rows are read from document starts. Values flowing through the layers in bf16, attention computed with FlashAttention-2 (both chosen by `speed.py`). 1,281 updates of 524,288 tokens each (4 passes of 128 rows added up), about 670 million tokens in total (about 12 tokens per parameter). One seed (1337).
- The same cost: the baseline was run briefly on the same machine, and the number of updates was set from the time per update and the time outside updates. Training took 796 seconds of wall-clock time; the baseline took 830 and 808 seconds, the first bundle 810.
- Data and scoring: the same as the baseline's. All questions (30,470) were scored.

## Scores

| | Baseline seed 1 | Baseline seed 2 | First bundle | Second bundle | Change from the first bundle |
|---|---|---|---|---|---|
| Validation BPB | 1.2536 | 1.2571 | 1.0237 | 1.0461 | +2.2% |
| Validation loss (`train.py`, random positions) | 3.4646 | 3.4834 | 2.8410 | 2.8974 | +2.0% |
| Correct-answer BPB (mean of 6) | 1.6921 | 1.6700 | 1.4037 | 1.3420 | −4.4% |
| MMLU | 1.953 | 1.990 | 1.662 | 1.628 | −2.1% |
| ARC-Easy | 1.769 | 1.716 | 1.463 | 1.384 | −5.3% |
| ARC-Challenge | 1.736 | 1.705 | 1.452 | 1.392 | −4.2% |
| HellaSwag | 1.292 | 1.296 | 1.066 | 1.026 | −3.7% |
| PIQA | 1.615 | 1.636 | 1.339 | 1.311 | −2.1% |
| SciQ | 1.787 | 1.676 | 1.441 | 1.311 | −9.0% |

Correct-answer BPB fell on all 6, by well more than the standard error of the differences on the same questions (0.0006–0.011). The fall is also larger than the movement of mean correct-answer BPB across seeds (about 1.3%). The final training loss was 2.6740 on the training data (read from document starts).

We compared on the same questions that OpenAI's GPT-2 (120 million parameters) was measured on (the first 200 questions of each of 5 benchmarks).

| | GPT-2 | Baseline seed 1 | Baseline seed 2 | First bundle | Second bundle |
|---|---|---|---|---|---|
| ARC-Easy | 1.704 | 1.750 | 1.691 | 1.473 | 1.391 |
| ARC-Challenge | 1.518 | 1.710 | 1.681 | 1.431 | 1.370 |
| HellaSwag | 0.996 | 1.341 | 1.352 | 1.124 | 1.076 |
| PIQA | 1.328 | 1.585 | 1.600 | 1.302 | 1.271 |
| SciQ | 1.511 | 1.756 | 1.613 | 1.377 | 1.233 |
| Mean | 1.411 | 1.628 | 1.587 | 1.341 | 1.268 |

## Why validation BPB rose

After training, the saved weights were measured again on 96 rows of validation data on the processing machine (CPU, float32), in two ways: rows at the same fixed positions as in `scoring.py`, and rows starting at document starts.

| Rows | Baseline seed 1 | Baseline seed 2 | First bundle | Second bundle | Change from the first bundle |
|---|---|---|---|---|---|
| Fixed positions (the first 96 rows, read the same way as in `scoring.py`) | 1.2256 | 1.2311 | 0.9908 | 1.0090 | +1.8% |
| From document starts | 1.2157 | 1.2193 | 0.9768 | 0.9360 | −4.2% |

Loss on the fixed-position rows, split by token position (per token, nats):

| Token position | Share of tokens | First bundle | Second bundle | Change |
|---|---|---|---|---|
| First 64 tokens of a row | 6% | 3.327 | 4.901 | +47.3% |
| Rest of the row | 94% | 2.727 | 2.677 | −1.9% |
| The part of a document past its first row (row starts included) | 16% | 2.562 | 2.657 | +3.7% |

On rows starting at document starts, the change is an even −3.5 to −4.3% at every position.

The second bundle trained only on rows starting with BOS, so it badly mispredicts the beginning of rows that start mid-document. This part alone pushes the overall loss up by about 3.6%, which explains the rise in validation BPB. The questions for correct-answer BPB all start with BOS, so this weakness does not show there. It is a side effect of aligning rows to document starts and was not in the estimate (0–0.2%).

Whether other items contribute to this weakness was not separated. Mixing rows that start partway into training and measuring again was not done.

## Speed

Measured on the same machine by running 26 updates. The update time is the median, excluding the first 5.

| Shape and method | Rows per pass | One update | Outside updates |
|---|---|---|---|
| Baseline, PyTorch's default | 64×8 | 0.834 s | 34 s |
| Bundle, PyTorch's default | 128×4 | 0.639 s | 44 s |
| Bundle, FlashAttention-2 | 128×4 | 0.639 s | 12 s |
| Bundle, FlashAttention-2, values in bf16 | 128×4 | 0.615 s | 41 s |
| Same as above | 64×8 | 0.625 s | 50 s |
| Same as above | 256×2 | Out of memory (with 78GB in use, 8GB could not be allocated) | |
| Same as above, no QK-norm | 128×4 | 0.591 s | 33 s |
| Same as above, no x0 and resid weights | 128×4 | 0.597 s | 39 s |
| Same as above, no softcap | 128×4 | 0.599 s | 40 s |
| Same as above, GELU instead of ReLU² | 128×4 | 0.632 s | 39 s |

- Cost per component (per update): QK-norm +4.2%, x0 and resid weights +3.1%, softcap +2.7%, ReLU² −2.6% against GELU, values in bf16 −3.7%. All are below the thresholds for considering removal (5% for QK-norm and the x0 and resid weights, 3% for softcap).
- The time per update relative to the baseline rose from the first bundle's 0.662 to 0.738 (11.5% slower). Updates fell accordingly from 1,441 to 1,281, worth about +0.66% by conversion. The scores above include this loss.
- Because of the difference in time outside updates (the first compile of a new shape), the form with values in bf16 and the default form were calculated to get almost the same number of updates (1,281 and 1,279). The real run's first update took 6.6 seconds, and it finished in less wall-clock time than the measured budget.
- The estimated utilization during training, from what `train.py` displays (about 111.7% relative to the A100), is about 35% of the H100's peak.
- FlashAttention-2 and the form with values in bf16 matched the default in the outputs and gradients of a small model (`attention_check.py`).

## Time and cost

| Stage | Measured |
|---|---|
| From renting to ready | 21 min (the first machine did not start within 20 minutes; the second started right away) |
| Transferring code and data (4GB, relayed from the processing machine) | 4 s, 200 s |
| Adding packages | 8 s |
| Speed measurement (9 runs and the check) | 525 s |
| Training | 796 s |
| Scoring | 19 s |
| Saving the weights | 3 s |
| Fetching the results | 12 s |

About $1.13 for the roughly 28 minutes the second machine was rented (calculated from time and price). Training was about $0.53 of it. The charge for the first machine ($2.15/hour) while it was loading was not checked separately.

## Findings

- Correct-answer BPB and validation BPB read from document starts fell within the estimate for the 10 items (2–6%). Which items worked was not measured separately.
- "Reading from document starts", the item with the smallest expected effect, had a side effect that moved the yardstick. Looking at validation BPB alone, the bundle would have been judged a loss. It became clear from looking at both validation BPB and correct-answer BPB, and from splitting by position.
- Some components cost more than estimated, and together they cut updates by 11%.

## What happened

- The first machine did not finish pulling the image within 20 minutes, and was automatically replaced by a second.
- The analysis was done on the processing machine after the rented machine was deleted, so there was no extra cost.

## Not verified

- The quality effect of each item. In particular, the effect and side effect of reading from document starts alone were not measured separately.
- The learning rates. They may not suit the current shape.
- The noise with other seeds.
- Whether the spike at the start of training caused by the x0 and resid weights also occurred with the real batch. Logs are every 10 updates, and the loss at the 10th was 5.56.
