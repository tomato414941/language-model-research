# Evaluation for the line that improves a small model

The evaluation for the line that improves Qwen3-1.7B with reinforcement learning and other methods. Decided on 2026-11-03, before measuring. It follows the rules in language-model's `docs/evaluation.md`.

## Decisions

- **Format:** Tasks whose authors define a procedure are asked with the authors' code and settings. The Swallow evaluation (swallow-evaluation-instruct) remakes the formats for thinking models, so it is not used for tasks that have an authors' procedure. The task-by-task check and the tools used are in "Format of each task" below. The scores cannot be compared with Swallow's tables.
- **How to compare:** The original model and the improved model are measured in the same environment and with the same settings. Both are measured with thinking and without. The length of the answers (number of tokens) is compared too.
- **Passing bar:** With thinking, the mean over the Japanese tasks rises by +0.05 or more. And, both with and without thinking, none of the tasks below falls by more than 3 times the standard error of the difference between the two models.
- **Japanese tasks (in the mean):** JamC-QA, WMT20 English→Japanese, WMT20 Japanese→English, M-IFEval-Ja, MMLU-ProX (Japanese), BenchMAX Science (Japanese), PolyMath (Japanese), JHumanEval.
- **English tasks:** HellaSwag, IFBench, MMLU-Pro, GPQA Diamond, MATH-500, AIME 2024–2025, LiveCodeBench.
- **Other languages:** MMLU-ProX's lite version (27 languages other than Japanese and English), MGSM (9 languages other than Japanese and English), PolyMath (16 languages other than Japanese and English). PolyMath's authors' scoring also counts whether the answer is in the same language as the question.
- **Safety:** AnswerCarefully v2.0 and borderline-v1.0, XSTest.
- **Tool calling:** BFCL.
- **Long context:** RULER. Measured only when additional pretraining on Japanese text is done.
- **When to measure:** The Japanese and English tasks are measured every time. The others are measured alongside the original model when the Japanese mean reaches +0.05.
- **What cannot be measured:** MT-Bench. GPT-4 (gpt-4-0613), which its authors used for scoring, was retired on 2026-10-23.
- **Not checked:** The authors' procedures for XSTest and HellaSwag.
- **Not decided:** Whether to use OpenAI's paid API to score AnswerCarefully.

## Format of each task

On 2026-11-03, each task's README and dataset card, MMLU-Pro's evaluation code and lm-evaluation-harness's definition of MMLU-ProX were read. The papers' main text was not read.

The Swallow evaluation remakes the tasks into a form that has the model think at length without examples. For example, MMLU-Pro and MMLU-ProX use no examples, temperature 0.6 and up to 32,768 tokens, unlike the authors' specification (5 examples, temperature 0, output up to 2,048 tokens). IFBench is measured with per-instruction strict at temperature 0.6, and LiveCodeBench with the prompt and the way code is extracted changed.

| Task | Authors' specification | How it is asked |
|---|---|---|
| MMLU-ProX (Japanese, and the lite version for other languages) | The card specifies lm-evaluation-harness's `mmlu_prox_{language}`. For Japanese, 5 examples, temperature 0, output up to 2,048 tokens, with the answer extracted from "答えは (X) です" ("the answer is (X)") | harness. For the lite version, every question of `mmlu_prox_lite_{language}`, published by the authors |
| MMLU-Pro | The authors' `evaluate_from_local.py`. 5 examples, temperature 0, output up to 2,048 tokens, with the answer extracted from `answer is (X)` | The authors' code |
| IFBench | The authors' `run_eval`. Temperature 0; for thinking models, the output limit is raised and the thinking is removed before scoring. The paper reports per-prompt loose | The authors' code |
| LiveCodeBench | The authors' `lcb_runner`. pass@1 and pass@5 | The authors' code. The problems are the 175 added in LiveCodeBench's v6 |
| PolyMath | The authors' prompts (`instruction.py`) and scoring code. Gives accuracy and whether the answer is in the same language as the question | The authors' code |
| JamC-QA | Has the model output the choice's text rather than its label, and checks for an exact match. Can be measured with FlexEval's `jamcqa` | FlexEval |
| BenchMAX Science | Measured with lm-evaluation-harness and BenchMAX's code | The authors' code |
| GPQA Diamond | The authors' code has the types zero_shot, 5_shot, zero_shot_chain_of_thought and chain_of_thought | zero_shot_chain_of_thought from the authors' code |
| JHumanEval | As in HumanEval, the input is the function's header and its description | The same completion form as HumanEval |
| M-IFEval-Ja | The official scoring code | The official scoring code (the same as language-model's v6) |
| MT-Bench (English, Japanese) | GPT-4 scores it. The Japanese version is Stability AI's FastChat | Cannot be measured. gpt-4-0613 was retired on 2026-10-23 |
| AnswerCarefully | The authors' llm-jp-judge. Scoring is through the OpenAI or Azure API, with gpt-4o-2024-08-06 as the example | llm-jp-judge |
| BFCL | The authors' `bfcl generate` and `bfcl evaluate` | The authors' code |
| RULER | The authors' generation and scoring scripts | The authors' code |

Tasks whose authors define no procedure are asked with public tools.

- MGSM: harness (the same as language-model's v6)
- MATH-500, AIME 2024–2025, WMT20 English→Japanese and Japanese→English: the Swallow evaluation

Tasks whose authors' procedure could not be checked:

- HellaSwag: the authors' repository cannot be viewed because of a DMCA notice (since 2026-09-14). As in language-model v6's comparison with outside models, it is asked with harness.
- XSTest: the authors' description could not be fetched.
