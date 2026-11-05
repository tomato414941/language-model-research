# Runs

One row per run under the frame (README.md), in the order they ran. The reasons for each change are in the message of its commit. QA, math and code BPB are the three OLMES suites' scores, each the macro average over its tasks of the bits per byte of the correct answer; the validation BPB is the yardstick every run shares, measured as scoring.py measures it, over the first 2^20 tokens of FineWeb's held-out file.

| Commit | QA BPB | Math BPB | Code BPB | Validation BPB | Cost | Adopted | What changed | Sources |
|---|---|---|---|---|---|---|---|---|
| a0da8b4 | NA | NA | NA | NA | $2.00 | No: evaluated by CORE, then the frame's evaluation, which did not finish within the hour (18 of its 22 tasks) | Nothing: the current best (the third bundle without the mixing of the last weights), trained 2,181 updates in 30 minutes | — |
| e77674d | 1.5155 | 0.9386 | 1.1122 | 1.1255 | $2.07 | Yes: its validation BPB, the only measure that the third bundle also has, is 1.1255 against 1.1633, but it trained 30 minutes against 13 | research-v4, eight changes: attention within documents; tables of pairs and triples with 32 rows per token; STEM in half of the MLPs; xIELU; 262,144 tokens per update in passes of 64 rows; the learning rate falling over the last 75%; Cautious AdamW; a tenth of the rows each from FineMath 4plus and Common Pile's code. 3,375 updates; evaluation 20 minutes; weights not kept | Llama 3; arXiv 2402.13991, 2501.16975, 2601.10639, 2411.13010, 2401.02954, 2502.15938, 2411.16085, 2502.02737, 2602.02522 |
