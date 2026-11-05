# modded-nanogpt's Muon per pair of heads (PR #253)

PR #253 of `KellerJordan/modded-nanogpt` (2026-04-08, merged as a record on 04-19).

Until then, Q, K, V and O were each orthogonalized as one matrix. This PR splits K and V into groups of two heads and orthogonalizes each group (to match the attention that pairs two heads in some layers). Q and O are unchanged. The optimizer is NorMuon, with six heads of 128 dimensions.

| Condition | Runs | Steps | Time | Loss |
|---|---|---|---|---|
| Original | 4 | 1,490 | 93.02 s | 3.2782 |
| Original | 8 | 1,480 | 92.55 s | 3.2800 |
| This PR | 8 | 1,480 | 92.63 s | 3.2788 |

- At the same 1,480 steps, the mean loss over 8 runs each is −0.0012. The standard deviation within one condition is 0.0009–0.0017.
- The steps needed to reach the target of 3.28 drop by 10 (−0.7%), and the time is about 0.4 seconds shorter. The time per step rises by about 0.1%.
- The author says that splitting more finely should reduce the compute.

The PR's description and README, and the parts of `train_gpt.py` on the attention's shape and the assignment of optimizers, were read. The logs were not read.
