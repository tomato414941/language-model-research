# Measuring the baseline and two bundles by accuracy on all questions

On 2026-11-02, the baseline, the first bundle and the second bundle were measured on all questions of lm-evaluation-harness and set beside `language-model`'s v5, 8k version, after pretraining. The mean accuracy over the 6 tasks was .452 for the second bundle and .461 for v5, a difference on the same questions of −.009 ± .003 (± is the standard error, here and below). The difference from the first bundle to the second was +.003 ± .003, within the error.

## Setup

- Models: the saved weights of the baseline's two seeds (baseline.md), the first bundle (first-bundle.md) and the second bundle (second-bundle.md). Not retrained.
- Tool: lm-evaluation-harness 0.4.13's default prompting (`harness.py`). All 30,470 questions, in bf16. The metric is `acc` for SciQ and `acc_norm` for the other five.
- Machine: one RTX 4090 on Vast.ai in Brazil, $0.368/hour. Driver 595.71.05. The image is the same as the baseline's (torch 2.15.0+cu132). The measurement does not compare costs, so an H100 was not used.
- Compared with: the "8k 事前学習の後" (after 8k pretraining) column of `docs/results.md` in `language-model`. It was measured on all questions with the same version of the harness, in bf16, with the same reading of the metrics. Recomputing the means from the per-question records matched the recorded values. MMLU is the mean over all questions, not the plain mean of the 57 subjects, the same as in `harness.py`.
- Checking the scoring: the per-task means of correct-answer BPB matched the values `scoring.py` measured on the H100, with a difference of at most .0005. No effect of changing the machine is visible.

## Scores

Accuracy on the 6 tasks. "Difference from v5" is the second bundle minus v5, on the same questions.

| Task | Baseline seed 1 | Baseline seed 2 | First bundle | Second bundle | v5 | Difference from v5 |
|---|---|---|---|---|---|---|
| MMLU (continuation form) | .257 | .256 | .264 | .264 | .274 | −.010 ± .003 |
| ARC-Easy | .353 | .350 | .430 | .439 | .454 | −.015 ± .009 |
| ARC-Challenge | .213 | .221 | .251 | .247 | .263 | −.016 ± .011 |
| HellaSwag | .268 | .276 | .297 | .308 | .347 | −.039 ± .004 |
| PIQA | .567 | .562 | .627 | .635 | .611 | +.024 ± .009 |
| SciQ | .675 | .679 | .827 | .820 | .818 | +.002 ± .011 |
| Mean | .389 | .391 | .449 | .452 | .461 | −.009 ± .003 |

Differences in the mean over the 6 tasks, on the same questions:

| Compared | Accuracy difference | Mean correct-answer BPB |
|---|---|---|
| Baseline seed 1 → seed 2 | +.002 ± .003 | 1.692 → 1.670 |
| Baseline seed 1 → first bundle | +.060 ± .004 | 1.692 → 1.404 |
| Baseline seed 2 → first bundle | +.059 ± .004 | 1.670 → 1.404 |
| First bundle → second bundle | +.003 ± .003 | 1.404 → 1.342 |
| v5 → second bundle | −.009 ± .003 | 1.336 → 1.342 |

From the first bundle to the second, only HellaSwag clearly rose (+.011 ± .003); the other five were within the error.

On some tasks, correct-answer BPB and accuracy do not agree in the comparison with v5.

| Task | Correct-answer BPB (second bundle, v5) | BPB difference | Accuracy difference |
|---|---|---|---|
| MMLU (continuation form) | 1.628, 1.470 | +10.7% | −.010 |
| ARC-Easy | 1.384, 1.434 | −3.5% | −.015 |
| ARC-Challenge | 1.392, 1.409 | −1.2% | −.016 |
| HellaSwag | 1.026, 0.992 | +3.4% | −.039 |
| PIQA | 1.311, 1.317 | −0.5% | +.024 |
| SciQ | 1.311, 1.394 | −6.0% | +.002 |

## MMLU in its authors' format

The form that gives 5 examples and answers with a letter (`harness.py --authors-mmlu`). Not included in the means above. There is no record of v5 at the same stage, so no comparison was made.

| | Baseline seed 1 | Baseline seed 2 | First bundle | Second bundle |
|---|---|---|---|---|
| Accuracy (14,042 questions) | .255 ± .004 | .250 ± .004 | .248 ± .004 | .252 ± .004 |

None can be told apart from the .25 of random guessing. None of the four models shows an ability to pick the letter in this format.

2,725 questions (19%) do not fit in the 1,024-token context, so their front is cut off and examples are lost. They are common in the high school history and law subjects; for example, none of the 165 European history questions fit.

## Differences when comparing with v5

- Evaluation questions: v5 removes documents containing the text of evaluation questions from its training data (all 6 tasks are covered). The English documents removed, including those for evaluation tasks other than the 6, were 12,298 in the stable phase (0.29% of the documents kept) and 1,514 in the decay phase (likewise 0.19%). This research uses ClimbMix as is, without removal. How many are in it was not counted. If it has an effect, it raises this research's scores.
- How the data was chosen: ClimbMix's mixture was chosen by results on the validation questions of PIQA, ARC-Easy and HellaSwag, three of the 6 tasks (climbmix.md in `language-model`).
- Amount of training: v5 trained on 11.7 billion tokens of Japanese and English, about 5.9 billion of them English. The second bundle trained on about 670 million, English only.
- Size: v5 has 300 million parameters. The second bundle has 54.4 million used in computation and 85.85 million in total.
- Training cost: v5 cost $28.71 for Japanese and English (A100). The second bundle cost about $0.53 (H100).
- Seeds: one for each bundle. The difference between the baseline's two seeds is +.002 ± .003 on the mean.

## Time and cost

| Stage | Measured |
|---|---|
| From renting to ready | 4 min 30 s |
| Transferring code and data (894MB, through Vast's proxy) | 32 s, 211 s |
| Adding packages | 1 min 7 s |
| 6 tasks (1 model, about 10.5 million tokens) | About 3 min 55 s |
| Authors' format (1 model, about 36 million tokens) | About 7 min |
| Fetching the results | 33 s |

About $0.38 for the roughly 62 minutes of the rental (calculated from time and price). With batches of 65,536 tokens, about 14GB of GPU memory was used. On the processing machine's CPU, the authors' format alone was estimated at about 14 hours per model (not measured).

## Findings

- The gain from the baseline to the bundles was clearly visible in accuracy too.
- The second bundle's changes lowered correct-answer BPB by 4.4%, but the mean accuracy moved within the error. A difference of a few percent in BPB may not show up as a rise in accuracy.
- The second bundle does not reach v5 on the mean over the 6 tasks. The direction differs by task: below on HellaSwag and MMLU, above on PIQA.

## What happened

- The machine's SSH port could not be reached directly, and the data transfer failed once. `rental.py` was changed to try the direct route and then the proxy.

## Not verified

- How many evaluation questions ClimbMix contains, and how much they move the scores.
- The bundles' noise across seeds.
- v5's score at the same stage in the authors' format.
