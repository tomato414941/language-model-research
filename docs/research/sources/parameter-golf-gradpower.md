# Parameter Golf's GradPower (PR #1682)

PR #1682 of `openai/parameter-golf` (2026-04-16, in the non-record track, unmerged as of 2026-11-04). It applies GradPower to Muon and compares values of p.

## Settings

- 17 million parameters (the default architecture is 9 layers, width 512 and vocabulary 1,024, with shared embedding and output). FineWeb, context 1,024, 524,288 tokens per update.
- The hidden layers' matrices use Muon (learning rate 0.04), with the gradient turned into `sign(g)·|g|^p` before the momentum. The embeddings and scalars stay on Adam, without the power.
- 1,200 seconds on one H100 (about 2,400 updates). The wide comparison of p ran on one RTX 4080, at 1,200 and 600 seconds.

## Results

| Setting | p | BPB | Difference from p=1.0 |
|---|---|---|---|
| H100, 1,200 seconds, 1 seed | 0.9 | 1.28338 | −0.00353 (−0.27%) |
| Same | 1.0 | 1.28691 | |
| RTX 4080, 1,200 seconds, mean of 3 seeds | 0.9 | 1.36673 | −0.00191 (−0.14%) |
| Same | 1.0 | 1.36864 | |
| RTX 4080, 600 seconds (2–3 seeds) | 1.05 | 1.51246 | +0.00632 |
| Same | 1.1 | 1.51651 | +0.01037 |
| Same | 1.2 | 1.52744 | +0.02129 (+1.4%) |
| Same | 1.0 | 1.50615 | |

- At 1,200 seconds on the RTX 4080, every p from 0.85 to 0.975 is lower than 1.0, and 0.85 to 0.925 are nearly the same. The per-seed differences at 0.9 are −0.0028, −0.0025 and −0.0005, and the standard deviation across seeds within one p is 0.0016–0.0030.
- The two H100 runs had the same time but different numbers of updates: 2,391 and 2,355.
- The author says the paper's p=1.2 goes the wrong way in this setting, and that p needs to be matched to the optimizer and the gradient noise.

The PR's description and README, and the Muon and settings parts of `train_gpt.py`, were read. The logs were not read.
