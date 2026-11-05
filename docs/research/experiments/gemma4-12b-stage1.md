# Measured Gemma 4 12B's generation settings and a public distilled version

On 2026-11-03, whether Gemma 4 12B's answers that never finish thinking decrease without training was measured on part of the Swallow tasks. Two things were tried: changing the generation settings in two ways, and having a public distilled 12B solve the same questions. The original 12B and the distilled version were also compared on tau2-bench's telecom. This is the first stage of a plan to rebuild the 12B by distillation; nothing was trained. The scripts and the per-question score tables are in `open-models/gemma4-12b-stage1/`.

## Conditions

- Machine: one RTX PRO 6000 Blackwell (96GB) on Vast.ai, $1.36/hour (GPU $1.333, disk $0.026). Driver 595.84, CUDA 13.2.
- Inference: vLLM 0.32.0, bf16, `--reasoning-parser gemma4`, 90 concurrent sequences (not specified when the comparison was measured). The context limit was 32,768 for the Swallow tasks and 65,536 for tau2, and for tau2 tool-call parsing was enabled.
- Scoring: v202604 of [swallow-evaluation-instruct](https://github.com/swallow-llm/swallow-evaluation-instruct) (lighteval's endpoint). Temperature 0.6, top_p 0.95, thinking on. The setting changes were added to the requests by a proxy placed between the evaluation software and vLLM.
- Questions: only part of each task was used. JamC-QA 100 questions, IFBench 100, M-IFEval (Japanese) 60, PolyMath (Japanese) high and top 20 each (10 each for c2), AIME 2024 and 2025 10 each, 4 times.
- Score: for JamC-QA and PolyMath, the match of the extracted answer; for M-IFEval and IFBench, the per-instruction accuracy under strict judging; for AIME, the mean accuracy over the 4 times.
- Comparison: the original 12B (`google/gemma-4-12B-it`), measured on 2026-11-01 with the same evaluation software, the same vLLM and the same generation settings. The comparison was made on the same questions. For JamC-QA alone, the 100 questions chosen change from run to run, so the comparison used the 26–33 questions that overlapped with the original 12B's 400.
- Answer that never finishes thinking: an answer that ended with its thinking part not closed, that is, one where the answer the evaluation software received starts with `<think`.
- Conditions tried:
  - c2: `repetition_penalty` 1.08. AIME was not measured.
  - c3: `logit_bias` −2.0 on the 110 forms of rethinking words (wait, maybe, actually, however, but and so on) that are a single token.
  - c4: the public distilled version [yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2) (revision `af644ab`), used with the settings unchanged.
  - The temperature 1.0 condition (c1) was not measured.
- tau2: text-v1 of [tau2-bench](https://github.com/sierra-research/tau2-bench) at `19f13ea`. The 50 English telecom questions were run once each. The questions were narrowed down, so these are not official scores.
  - The agent role used temperature 0.6, top_p 0.95, thinking on. The customer role was the original 12B, temperature 0, thinking off.
  - A conversation was cut off at 10 tool errors or 200 steps. 32 conversations ran at once.
  - For the distilled version, two servers, one for the agent role and one for the customer role, each took half the GPU. For the original 12B, one server took both roles.
  - Only for the original 12B, a limit of 20 minutes per conversation (`--timeout 1200`) was set.

## Time and cost

| Stage | Time (UTC) | Length |
|---|---|---|
| From renting until setup and downloading the models finished | 04:55–06:02 | 67 min |
| Waiting for approval, with the machine doing nothing | 06:02–07:34 | 92 min |
| c3, first run (unusable because of a GPU error) | 07:34–08:16 | 42 min |
| c4 | 08:16–09:07 | 50 min |
| c2 | 09:07–09:31 | 24 min |
| c3, second run (frozen for 34 minutes of it) | 09:31–10:49 | 78 min |
| tau2, distilled version | 10:49–11:35 | 46 min |
| tau2, original 12B | 11:35–12:11 | 36 min |
| tau2, remeasuring 14 questions of the original 12B | 12:11–13:22 | 71 min |

The rental of 8.45 hours cost $11.64: GPU $11.27, disk $0.22 and $0.15 of traffic for the 55.6GB downloaded, according to Vast's billing records. The setup itself (the evaluation software, vLLM, tau2 and downloading the two models) finished in 9 minutes.

## Scores

### c2: suppressing repetition

| Task | Questions | Share that never finished thinking | Score | Difference (± standard error) |
|---|---|---|---|---|
| JamC-QA | 33 | 15.2% → 6.1% | 0.424 → 0.485 | +0.061 ± 0.106 |
| M-IFEval | 60 | 6.7% → 1.7% | 0.778 → 0.797 | +0.019 ± 0.027 |
| IFBench | 100 | 16.0% → 8.0% | 0.635 → 0.665 | +0.030 ± 0.037 |
| PolyMath high | 10 | 40% → 20% | 0.500 → 0.400 | −0.100 ± 0.100 |
| PolyMath top | 10 | 50% → 10% | 0.300 → 0.800 | +0.500 ± 0.224 |

On the three tasks other than math, the share of answers that never finished thinking fell to half or less. On these three tasks, every score difference was smaller than twice its standard error. PolyMath had 10 questions each; top rose and high fell.

### c3: making rethinking words less likely

| Task | Questions | Share that never finished thinking | Score | Difference (± standard error) |
|---|---|---|---|---|
| JamC-QA | 31 | 16.1% → 16.1% | 0.452 → 0.419 | −0.032 ± 0.073 |
| M-IFEval | 60 | 6.7% → 1.7% | 0.778 → 0.781 | +0.003 ± 0.022 |

The other 4 tasks could not be measured because the server froze.

### c4: the public distilled version

| Task | Questions | Share that never finished thinking | Score | Difference (± standard error) |
|---|---|---|---|---|
| JamC-QA | 26 | 7.7% → 11.5% | 0.538 → 0.346 | −0.192 ± 0.124 |
| M-IFEval | 60 | 6.7% → 5.0% | 0.778 → 0.700 | −0.078 ± 0.052 |
| IFBench | 100 | 16.0% → 18.0% | 0.635 → 0.500 | −0.135 ± 0.055 |
| PolyMath high | 20 | 40% → 30% | 0.450 → 0.150 | −0.300 ± 0.105 |
| PolyMath top | 20 | 45% → 45% | 0.300 → 0.100 | −0.200 ± 0.117 |
| AIME 2024 | 10 (4 times each) | 0% → 12.5% | 0.975 → 0.600 | −0.375 ± 0.130 |
| AIME 2025 | 10 (4 times each) | 27.5% → 25.0% | 0.625 → 0.225 | −0.400 ± 0.125 |

The score fell on all 7 items. The share of answers that never finished thinking rose on some items and fell on others, with no clear direction.

### tau2's telecom

| | Original 12B | Distilled version |
|---|---|---|
| Conversations that got a score (of the 50 questions) | 48 | 47 |
| Correct on the 45 questions scored for both | 11 (24.4%) | 6 (13.3%) |
| Questions only one of the two got right | 6 | 1 |
| How the 45 questions ended | 44 ended by the customer, 1 cut off by errors | 23 ended by the customer, 22 cut off by errors |
| Tool calls by the agent per conversation | 9.6 | 15.9 |
| Of these, calls to tools the agent does not have | 3.4 | 7.8 |

The imbalance in the questions only one of the two got right had p = 0.12 by a sign test.

- **Tools the agent does not have:** Both models called tools the agent does not have in 44 of the 45 questions.
  - The most common were tools that check the customer's phone (`check_status_bar`, `check_network_status`, `check_apn_settings`, `run_speed_test` and so on).
  - The distilled version also called `send_message`, which tau2 does not have, in 13 conversations.
  - The distilled version made more calls, and more of its conversations were cut off on reaching 10 errors.
- **The author's report:** The distilled version's author writes in the [model card of the GGUF version](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF) that it rose from about 15% to about 55% on 20 questions of tau2's telecom. Whether the customer-role model and other conditions were the same as here has not been checked.

## What happened

- **GPU errors stopped the server 4 times:** This also happened to a server with unchanged settings.
  - 08:01: crashed partway through c3's first run with `CUDA error: unspecified launch failure`.
  - 10:15: froze without an error partway through c3's second run. It was noticed and stopped 34 minutes after the output stopped.
  - 11:00: tau2's customer-role server crashed with `CUDA error: unspecified launch failure`. 2 questions of the distilled version could not be measured.
  - 12:07: the original 12B's tau2 server crashed with `CUDA error: an illegal memory access was encountered`. 14 questions were measured again.
- **c3's results:** The first run's results were not used, since they included empty answers from the crash. Of the second run, too, only the 2 tasks that finished before the freeze were used.
- **Conversations that did not end:** The distilled version's last conversation had not ended after 40 minutes and was stopped by hand. In the original 12B's remeasurement, 2 conversations had not ended after an hour, even with `--timeout 1200` set. The machine was deleted on reaching the cost limit, so those 2 questions have not been measured.
- **M-IFEval's auxiliary data:** M-IFEval first crashed reading a broken zip of its auxiliary data (NLTK). It was downloaded again and run once more.
- **Loading the distilled version:** Downloaded with a revision specified, it could not be loaded offline because `refs/main` was not found. The name of the downloaded snapshot was written into `refs/main`, and it then loaded.
- **Waiting:** While approval was awaited, the rented machine did nothing for about an hour and a half. Separately, the execute permission was left off the chaining script, which cost about 20 minutes of waiting.
