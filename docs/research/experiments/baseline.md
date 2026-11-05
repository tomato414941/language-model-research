# Trained the baseline nanoGPT on an H100

On 2026-10-31, nanoGPT as it is was trained on 500 million tokens of ClimbMix and run twice with different seeds. A third run was stopped partway.

## Conditions

- Machine: one H100 SXM 80GB in Japan on Vast.ai, $2.41/hour. Driver 615.71.09.
- Image: `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` (torch 2.15.0+cu132).
- Model: 12 layers, width 768, 12 heads, context 1,024, vocabulary 8,192, 91.25 million parameters (`configs/baseline.py`).
- Training: 954 updates of 524,288 tokens each (8 passes of 64 rows added together), about 500 million tokens in all. bf16, torch.compile.
- Data: ClimbMix (`karpathy/climbmix-400b-shuffle`, `915333b`). A BPE with a vocabulary of 8,192 was trained on this data, and 2 billion tokens were prepared. Validation used the last file (63.37 million tokens).
- Scoring: the correct-answer BPB on every question (30,470) of six English benchmarks, and the BPB of about the first 1.05 million tokens of the validation data.

## Time and cost

| Stage | Measured |
|---|---|
| From renting to ready to use | 162 s |
| Data transfer (3.9GB, relayed from the processing machine) | 197 s |
| Adding packages | 8 s |
| Training (seed 1, seed 2) | 830 s, 808 s |
| Of which, until the first update (compile and so on) | about 31 s |
| One update | about 0.83 s (about 630,000 tokens/s) |
| Scoring (seed 1, seed 2) | 20 s, 18 s |
| Saving the weights | 2 s |

The rental of about 41 minutes cost $1.97 (the difference in the Vast balance). During training, GPU utilization was 99–100%, memory about 28GB and power draw about 690W.

## Scores

| | Seed 1 | Seed 2 | Difference |
|---|---|---|---|
| Validation BPB | 1.2536 | 1.2571 | 0.0035 |
| Validation loss (`train.py`, 20 draws) | 3.4646 | 3.4834 | 0.019 |
| Correct-answer BPB (mean of the six) | 1.6921 | 1.6700 | 0.022 |
| MMLU (14,042 questions) | 1.953 | 1.990 | 0.037 |
| ARC-Easy (2,376 questions) | 1.769 | 1.716 | 0.053 |
| ARC-Challenge (1,172 questions) | 1.736 | 1.705 | 0.030 |
| HellaSwag (10,042 questions) | 1.292 | 1.296 | 0.004 |
| PIQA (1,838 questions) | 1.615 | 1.636 | 0.021 |
| SciQ (1,000 questions) | 1.787 | 1.676 | 0.111 |

The standard error of the differences paired by question was 0.0006–0.0083, much smaller than the differences between the seeds. The seed, not the number of questions, decides the variation. There were only two seeds, so the size of the variation is rough.

For reference: OpenAI's GPT-2 (vocabulary 50,257, 120 million parameters), scored on 200 questions from each of the same benchmarks, had a mean correct-answer BPB of 1.450 (measured with the scoring software; the questions differ, so the two cannot be compared directly).

## What happened

- The image's Python refuses packages added with `pip`, so `--break-system-packages` was needed.
- Written to a file, `train.py`'s output was buffered, so its progress could not be seen. It was changed to `python -u`.
- The command to stop the third run stopped itself and failed, so the run went on for about 6 minutes until the machine was deleted.
