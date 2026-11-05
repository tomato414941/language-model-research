# Research rules

Research is the line of work that picks techniques from papers and other people's records, bundles changes, and tests them. The evaluation, the training time, the GPU and the acceptance threshold are the same as the `language-model-autoresearch` loop's. Their training data differs (research reads FineWeb, the loop FineWeb-Edu; both may also read FineMath and Common Pile's code). The evaluation does not depend on the training data, so the two lines' scores can be set side by side, keeping in mind that the data differs.

## The frame (fixed; only the user changes it)

Everything trained and measured in this repository, from any window, follows this frame. Only the part being tested may differ.

- **Data:** FineWeb, and FineMath's `finemath-3plus` and `finemath-4plus` and Common Pile's `stackv2_edu_filtered` at the revisions pinned in the code.
- **Evaluation:** AI2's three OLMES suites `olmo3:base_easy:qa_bpb`, `olmo3:base_easy:math_bpb` and `olmo3:base_easy:code_bpb`: each is the macro average over its tasks of the bits per byte of the correct answer. Take the tasks from AI2's `olmes` at a pinned revision, with every task, example, prompt and score formula unchanged; the one approved addition is `primary_metric: macro` for the ARC-Easy/ARC-Challenge group, which upstream lacks. How the scores are computed is free, and measurements that the three scores do not use may be skipped. Validation BPB on the held-out FineWeb data is measured unchanged in every run as the common yardstick. Never change the evaluation for our own convenience: do not drop tasks or examples, change formats or move the threshold afterwards.
- **Training:** 30 minutes per run, from scratch to the end. Never continue an earlier run.
- **Other models:** Do not use other models in training: no distillation, and no other model's weights or outputs.
- **GPU:** One H100 SXM. Each rental lasts at most 1 hour and holds the setup, the training and the evaluation. Every rental needs the user's approval.
- **Acceptance:** Compare a candidate with the current best only on the evaluations both were measured with. When both have the three suites, adopt the candidate when the rates by which its BPB falls below the current best's, averaged over the suites, are positive (each suite counts equally); otherwise adopt it when its validation BPB is lower. Measure each candidate once and accept the noise: never repeat a run, add seeds or average runs to make a number more certain. Later bundles and retried changes correct wrong calls.

## What is free

- The tokenizer, the sequence length and how the data is read are free. Use the evaluation unchanged whatever the tokenizer. Never train on the validation data, the tokenizer included. Do not filter the allowed data by its content.
- Anything made beforehand is allowed if the CPU alone can make it, such as a tokenizer or tables. It does not count toward the 30 minutes.
- Changes may be tested together as a bundle. List every change.
- A discarded change is not discarded forever. Try it again when the comparison, the combination or the settings change, or when new evidence appears.

## Records

- Keep one row per run in a table.
- Write notes only when they will help later.
- Do not copy the [records](https://github.com/tomato414941/language-model/tree/main/docs/research) of `language-model`; link to them. Leave out chat remarks, and back claims with public papers and records.
