# Training the first bundle on an H100

On 2026-11-01, a bundle of changes was trained once at the same cost as the baseline (baseline.md). Validation BPB fell by 18.5% and the mean correct-answer BPB by 16.5%.

## Setup

- Machine: one H100 SXM 80GB on Vast.ai in Japan, $2.41/hour. Driver 615.71.09.
- Image: the same as the baseline's (torch 2.15.0+cu132). cuDNN 9.26.
- Model: 10 layers, width 640, 5 heads (128 dimensions per head), context 1,024, vocabulary 8,192, Value embeddings. 54.41 million parameters used in computation, 81.3 million in total.
- Training: Muon and AdamW together (Polar Express, NorMuon, cautious weight decay). Learning rates untuned. 1,441 updates of 524,288 tokens each (4 passes of 128 rows added up), about 760 million tokens in total. bf16, torch.compile, attention computed with PyTorch's default. One seed (1337). Settings in `configs/bundle.py`.
- The same cost: the baseline was run briefly on the same machine, and the number of updates was set from the time per update and the time outside updates (`speed.py`). Training took 810 seconds of wall-clock time; the baseline took 830 and 808 seconds.
- Data and scoring: the same as the baseline's. All questions (30,470) were scored.

## Scores

| | Baseline seed 1 | Baseline seed 2 | Bundle | Change from the baseline mean |
|---|---|---|---|---|
| Validation BPB | 1.2536 | 1.2571 | 1.0237 | −18.5% |
| Validation loss (`train.py`, 20 draws) | 3.4646 | 3.4834 | 2.8410 | −18.0% |
| Correct-answer BPB (mean of 6) | 1.6921 | 1.6700 | 1.4037 | −16.5% |
| MMLU | 1.953 | 1.990 | 1.662 | −15.7% |
| ARC-Easy | 1.769 | 1.716 | 1.463 | −16.1% |
| ARC-Challenge | 1.736 | 1.705 | 1.452 | −15.6% |
| HellaSwag | 1.292 | 1.296 | 1.066 | −17.7% |
| PIQA | 1.615 | 1.636 | 1.339 | −17.7% |
| SciQ | 1.787 | 1.676 | 1.441 | −16.7% |

The final loss of training was 2.8437 on the training data and 2.8410 on the validation data, with almost no gap.

For reference, we compared on the same questions that OpenAI's GPT-2 (120 million parameters) was measured on (the first 200 questions of each of 5 benchmarks; MMLU is left out because its questions could not be aligned).

| | GPT-2 | Baseline seed 1 | Baseline seed 2 | Bundle |
|---|---|---|---|---|
| ARC-Easy | 1.704 | 1.750 | 1.691 | 1.473 |
| ARC-Challenge | 1.518 | 1.710 | 1.681 | 1.431 |
| HellaSwag | 0.996 | 1.341 | 1.352 | 1.124 |
| PIQA | 1.328 | 1.585 | 1.601 | 1.302 |
| SciQ | 1.511 | 1.756 | 1.613 | 1.377 |
| Mean | 1.411 | 1.629 | 1.588 | 1.341 |

ClimbMix's mixture was chosen by results on some benchmarks (climbmix.md), so this comparison may favor the bundle.

## Speed

On the same machine, we measured the time per update and the time outside updates (Python startup, the first evaluation, compile and so on). The update time is the median of 26 updates, excluding the first 5.

| Shape and method | Rows per pass | One update | Outside updates |
|---|---|---|---|
| Baseline (12 layers, width 768), PyTorch's default | 64×8 | 0.822 s | 34 s |
| Bundle, PyTorch's default | 64×8 | 0.557 s | 34 s |
| Bundle, FlashAttention-2 | 64×8 | 0.558 s | 11 s |
| Bundle, FlashAttention-3 | 64×8 | 0.566 s | 33 s |
| Bundle, FlashAttention-4 | 64×8 | 0.558 s | 11 s |
| Bundle, PyTorch's default | 32×16 | 0.594 s | 36 s |
| Bundle, PyTorch's default | 128×4 | 0.544 s | 35 s |
| Bundle, PyTorch's default | 256×2 | Out of memory | |
| Bundle, PyTorch's default, no compile | 64×8 | 0.733 s | 5 s |
| Bundle, PyTorch's default, context 2,048 | 32×8 | 0.599 s | 29 s |

The time outside updates includes preparation that happens only on the first update of a new shape. With the default method, the first update took about 27 seconds every time the rows or the context changed; with FlashAttention-2 and 4 it took about 4 seconds. The second time with the same shape is short: the real training had the same shape as the `128×4` measurement, and its first update took 6.3 seconds rather than 28.3. The measurement without compile, also the second at `64×8`, took 1.5 seconds. We think this is caching, but it is unverified.

FlashAttention-2, 3 and 4 matched PyTorch's default in the outputs and gradients of a small model (`attention_check.py`).

The estimated utilization during training, from what `train.py` displays (123.6% of the A100's 312 TFLOPs), is about 39% of the H100's peak, about the same as the baseline.

## Time and cost

| Stage | Measured |
|---|---|
| From renting to ready | 38 s |
| Transferring code and data (3.9GB, relayed from the processing machine) | 4 s, 181 s |
| Adding packages and fetching FlashAttention-3 (run by hand) | 21 s, 23 s |
| Speed measurement (10 runs and the check) | 427 s |
| Training | 810 s |
| Scoring | 18 s |
| Saving the weights | 2 s |
| Fetching the results | 12 s |

About $1.2 for the roughly 29 minutes of the rental (calculated from time and price). Training was about $0.54 of it, the same as the baseline.

## Findings

- The effect of the whole bundle is larger than the sum of the expected effects of its items. Which items worked was not measured separately. The baseline's AdamW learning rates are considered a weak setting (fantastic-pretraining-optimizers.md), so the gain includes what tuning AdamW alone would give. It is not the effect of the new methods alone.
- It is well above the noise across seeds (about 0.3% in validation BPB). The bundle had only one seed.
- The attention methods differed by at most 1.6% per update, and FlashAttention-3 was the slowest. FlashAttention-2 and 4 do not have the roughly 23 seconds of preparation for a new shape.
- For rows per pass, 128 was 2.4% faster than 64, and 32 was 6.6% slower. 256 did not fit in memory (with 75.9GB in use, 4GB could not be allocated).
- Compile shortens each update by 24%. The compile itself takes no more than about 4 seconds, since that is FlashAttention-2's first update. Most of the "about 31 seconds for compile and so on" in the baseline's record is likely the preparation for a new shape.
- Context 2,048 is 7.5% slower per update than 1,024 at the same number of tokens.

## What happened

- The `cryptography` that `kernels` needs could not replace the image's Debian package, so the first attempt stopped at the first install. We noticed it by checking over SSH, fixed it with `--ignore-installed` and resumed in about 2 minutes.
- The command that started the script held the SSH connection until the script finished. It did not affect the run and was fixed later.
- `speed.py` chose the method by the update time alone and picked the default. The gap with FlashAttention-2 was within the error, and the preparation time was gone in the real run after the measurement, so this run lost almost nothing. Training on a new machine without the measurement would differ by the preparation time, so it was later changed to also count the time outside updates when choosing.

## Not verified

- The effect of each item. The learning rates are untuned, and better values may exist.
- Why FlashAttention-3 was not faster. Whether it is the build loaded, CUDA 13.2, the short context or the interaction with compile was not investigated.
- The bundle's noise with other seeds.
