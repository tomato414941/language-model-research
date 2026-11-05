# Integrating the OLMES evaluation

On 2026-11-05, README.md's evaluation (AI2's three OLMES suites `olmo3:base_easy:qa_bpb`, `math_bpb` and `code_bpb`) was built into `frame.py`, and the requests and tokens were counted on the processing machine (CPU) before renting. Later the same day, it was changed to run only the measurements the scores use (the last section). research-v4 was measured with the earlier form.

## Conditions

- olmes is at revision `5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`. olmes pins torch 2.8 and others, so it runs in an environment separate from the training one (torch 2.15). The versions are in `olmes.txt`.
- The suites come to 109 tasks (83 QA, 7 math, 19 code). olmes's aggregation is nested: MMLU's 57 subjects, basic_skills's 6 and multilingual MBPP's 17 languages each enter the average as one item.
- The tokenizer is the current one with a vocabulary of 8,192. It was passed to olmes's HuggingFace wrapper as a tokenizer that puts BOS first.

## Counts

The requests total 654,536. The tokens the model reads, with rows cut at the context of 1,024, are as follows.

| Suite | Tokens |
|---|---|
| QA | 180.72 million |
| Math | 5.03 million |
| Code | 7.09 million |
| Total | 192.83 million |

- The longest row is 5,978 tokens, and the longest answer 2,960 tokens. 101,995 rows (15.6%) are longer than the context, and 264 answers are longer than the context (lab_bench_dbqa, Minerva's math, multilingual MBPP).
- The large tasks are SQuAD (42.2 million tokens), DROP (34.58 million), HellaSwag (22.83 million) and CoQA (19.42 million). Even for BPB tasks, olmes makes requests for every choice and requests without the context, so the requests number several times the examples.

## Decisions made in the integration

The tasks, examples, prompts and scoring are olmes's, unchanged.

- **Requests are made beforehand:** For each task, olmes reads all of the few-shot source data row by row for every example, so making the requests takes time that grows with the square of the task's size (280 of 283 s for basic_skills's arithmetic). The requests do not depend on the model (on HumanEval, the requests with and without a model matched). So olmes's own runner makes them on the processing machine (about 13 minutes with 5 processes, 1.7GB), and the evaluation loads them instead of making them. olmes's multiple-choice tasks are written so that `make_metrics` also does the setup needed when the step that makes requests is skipped.
- **Rows whose answer is longer than the context:** olmes's wrapper stops when an answer exceeds the maximum length. The wrapper's maximum length was set to 4,096, and the model reads rows longer than the context in windows that step back by half from the row's end. The last 512 tokens get the same values as when the wrapper cuts at 1,024. Only the windows that reach the task's longest answer are read. In this form, the GPU reads 208.51 million tokens.
- **ARC's average:** The QA suite counts ARC (Easy and Challenge) as one item, but this revision of olmes has no rule for averaging the two, so olmes's aggregation gives no score for ARC or for the QA suite. It was given a macro average, the same as every other nesting.
- **Data format:** The data of some tasks has been re-saved with `datasets` 4 and cannot be read by `datasets` 3, which olmes pins. 4's `List` was made to read as 3's `Sequence`. ARC-Easy's test has 2,376 examples, and the form of the choices is the same as before.
- **pytrec_eval:** olmes's task list imports it, but these three suites do not use it. It cannot be built without a compiler, so it was replaced with `pytrec_eval-terrier`, which installs the same module.

## Time estimate

Measured in the first run (2026-11-05, the run measured by CORE): 162 s from renting to ready to use, 232 s for the data transfer (the longer of the parallel transfers), and about 31.5 minutes of training including preparation.

For the evaluation, olmes takes about 2 s per task on the processing machine to read each task's data (222 s for 109 tasks), and this is split over 3 processes. The GPU computation is 208.51 million tokens, 2–4 minutes judging from the training speed. Python's handling of each request is around 0.3 ms. Altogether, the evaluation was estimated at 6–12 minutes.

One rental is expected to take about 46–52 minutes, within the hour. Setting up olmes's environment and loading the data happen in the background during training.

## Not checked

- The evaluation's speed on the GPU. The estimate is calculated from the training speed.
- Speed and memory when 3 processes share the GPU.

## The form that measures only the correct answers (later on 2026-11-05)

README's evaluation changed to "how the scores are computed is free, and measurements the scores do not use may be skipped", so only the measurements the scores use were kept.

- **Selection:** For every task of the three suites, olmes scores with MCAccuracy and uses only the bits per byte of each question's one correct answer. MCAccuracy drops the requests without the question (the second half), groups by question and takes the request at the label's position (the first one if the label is not a number). This was applied to the requests olmes made, keeping only the answers (`frame.py answers`). On all 109 tasks, the scores given random log-likelihoods matched olmes's calculation (relative difference 2×10⁻¹⁶). The nested averages also matched olmes's aggregation.
- **Amount:** The requests went from 654,536 to 99,886, and the tokens the model reads from 208.51 million to 58.39 million (QA 45.23 million, math 5.84 million, code 7.32 million).
- **Measuring without olmes:** The way rows are made (moving the trailing whitespace of the context into the answer, putting BOS first, up to 4,097 tokens from the end) is the same as olmes's wrapper (lm_eval), and a test in olmes's environment confirmed that they match. Rows longer than the context are read in the same windows as before.
- **Tokenization beforehand:** It takes about 8 minutes on one core, so it is done on the processing machine (`frame.py rows`, 158MB) and used only when the tokenizer's contents are the same. With a different tokenizer, the rows are made on the spot.
- **What is sent:** Only the answers (251MB) and the rows (158MB) are sent to the rented machine. olmes's environment (torch 2.8), the task data (1.8GB) and the requests (1.7GB) are no longer needed.
- **Agreement with the earlier form:** With a small model (3 layers, context 256), the first 4 questions of each task were measured with the earlier form (every request through olmes's runner) and the current form. The relative differences of the three scores were 10⁻⁸ or less, and 6×10⁻⁸ or less per task (float32 error). On the CPU, the earlier form took 18 minutes and the current form 46 seconds.
- **Not checked:** The speed on the GPU. The tokens read are 28% of before, and olmes's per-task loading and Python's per-request handling are gone, so it is expected to go from 20 minutes to a few minutes.
