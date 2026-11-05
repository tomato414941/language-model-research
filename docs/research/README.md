# Research rules

Research is the line of work that picks techniques from papers and other people's records, bundles changes, and tests them. The evaluation, the training time, the GPU and the acceptance threshold are the same as the `language-model-autoresearch` loop's. Their training data differs (research reads FineWeb, the loop FineWeb-Edu). The evaluation does not depend on the training data, so the two lines' scores can be set side by side, keeping in mind that the data differs.

## The frame (fixed; only the user changes it)

Everything trained and measured in this repository, from any window, follows this frame. Only the part being tested may differ.

- **Data:** FineWeb. Only the user adds allowed data.
- **Evaluation:** AI2's OLMES suite `olmo3:base_easy:qa_bpb`: the macro average, over its 21 tasks, of the bits per byte of the correct answer. Run it with AI2's `olmes` at a pinned revision, with every task, example, prompt and scoring unchanged; only connect it to the model and tokenizer. Before the first run, count its tokens on the CPU and check that a run with the complete suite fits within the rental; if it does not, tell the user instead of dropping tasks or examples. Validation BPB on the held-out FineWeb data is recorded as a reference. Never change the evaluation for our own convenience: do not drop tasks or examples, change formats, move the threshold afterwards, or rerun until a better number appears. Only the user changes the evaluation.
- **Training:** 30 minutes per run, from scratch to the end. Never continue an earlier run.
- **Other models:** Do not use other models in training: no distillation, and no other model's weights or outputs.
- **GPU:** One H100 SXM. Each rental lasts at most 1 hour and holds the setup, the training and the evaluation. Every rental needs the user's approval.
- **Acceptance:** Adopt a candidate when its qa_bpb is lower than the current best's. Measure each candidate once and accept the noise: never repeat a run, add seeds or average runs to make a number more certain. Later bundles and retried changes correct wrong calls.
- When the measurement itself changes (the data, the evaluation, the training time or the GPU), measure the current best once under the new measurement before comparing candidates.
- Keep the frame in one place in the code.

## What is free

- Keep a single copy of the model and training code, the current best, and test changes by editing it directly. Do not add a config file per bundle.
- The tokenizer, the sequence length and how the data is read are free too. Use the evaluation unchanged whatever the tokenizer. Never train on the validation data, the tokenizer included. Do not filter the allowed data by its content.
- Anything made beforehand is allowed if the CPU alone can make it, such as a tokenizer or tables. It does not count toward the 30 minutes.
- Changes may be tested together as a bundle. List every change.
- A discarded change is not discarded forever. Try it again when the comparison, the combination or the settings change, or when new evidence appears.

## Records

- Keep one row per run in a table: commit, qa_bpb, validation BPB, cost, adopted or not, what changed, sources. Write the reasons in the commit message.
- Write notes in `topics/` (the current judgment on a theme), `sources/` (what one source says) and `experiments/` (one investigation or experiment) only when they will help later. Link only from `topics/` to the other two, and end each source note with what was read.
- Do not copy the [records](https://github.com/tomato414941/language-model/tree/main/docs/research) of `language-model`; link to them. Leave out chat remarks, and back claims with public papers and records.
- Do not write environment details such as hosts, keys or IP addresses.
