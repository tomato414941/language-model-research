# The 2026-11-05 survey

Before choosing the eighth change in the fourth bundle, we checked 37 new papers and posts. Most were read through search summaries; only IMU-1 and the Parameter Golf analysis were read in full.

## What bore on the bundle

| Source | Findings |
|---|---|
| IMU-1 (arXiv 2602.02522) | On a 70 million proxy model, QK-norm −0.63%, normalized value residual −0.38%, LayerNorm scaling −0.36%, per-head gates −0.05%. NorMuon and cautious weight decay −5.2%. For WSD, a decay period of 20% equals cosine, and 10% and 30% are worse. The training data has about 19% code and about 13% math (estimated from the amounts collected) |
| Parameter Golf: What Really Works? (arXiv 2607.01517) | An analysis of the records of a contest on FineWeb BPB with small models. Within the top entries, what worked was n-gram mixing at evaluation, Gated DeltaNet, value residual and windowed attention. Across all entries, per-document attention (VarLen), tables of token tuples, MuonEq-R, weight averaging and others also lean the right way. A single method usually gives under 1%, and the gains come from stacking. It is a correlational analysis, not controlled experiments |
| SmolLM2 (arXiv 2502.02737) | Mixes web, code and math. 75/20/5 in the later stages. Built FineMath and Stack-Edu |
| FineMath's comparison (Hugging Face) | The quality of math data changes the scores on math tasks |

## Others checked

- **Optimization:** MUON+ (2602.21545; claims to beat NorMuon and AdaMuon), HTMuon (2603.10067; better than NorMuon at 60 million and 135 million, and better still combined), MuonEq (2603.28254), Newton-Muon (2604.01472), MALT (2608.05088), SOAP, Muon, and Beyond (2607.20548; SOAP slightly ahead at 8B), a survey of optimization (2608.28557), Benchmarking Optimizers (2509.01440), AdaMuon (2507.11005), MARS (2411.10438).
- **Training procedure:** Small Batch Size Training (2507.07101), Multi-Power Law (2503.12811), the Marin 8B retrospective (WSD-S), Marin's issue #764 on the learning-rate schedule.
- **Data and objectives:** Data-Constrained Pretraining (2606.06888; about the side where data runs short), Bridging Compute- and Data-Optimal Pretraining (2607.25271; validation BPB predicts downstream scores well), Self-Play Pretraining (2609.30063), ESLM (2505.19893), SwallowMath/SwallowCode (2505.02881; unusable, since they are rewritten by other models), Scaling Inherently Interpretable LMs (2608.07594).
- **Tables and memory:** Engram (2601.07372), Memory Grafting (2605.20948), Lngram v2 (2609.03426), Qwen's design of a new architecture (2608.30320; includes n-gram tables).
- **Attention:** a survey of the evolution of attention (2609.39661), Deconstructing Attention (2510.11602), Language Models Can Control Their Own Attention (2609.02737; about inference time), Tucker Attention (2603.30033), a comparison of small-model architectures (2512.20877).
- **Tokenizers:** a comparison of 46 settings (COLM 2026, 2608.18062), the effects per language (2610.12144), simulation in bytes (2604.27263), distillation into byte models (2609.12303), ByteFlow (2603.03583), tokenizer extension (2607.15232), Boundless BPE (2504.00178).
- **Parameter Golf:** the official repository and a post on the leaderboard.

## Decisions

- The eighth change mixes in FineMath (4plus) and code (Common Pile), each at 10% of the training rows. It acts directly on OLMES's math and code suites. The mix was guided by SmolLM2 and IMU-1.
- Value residual leans the right way in both IMU-1 and Parameter Golf, and is kept as a candidate for the next bundle. The current bundle has Value embeddings, which may overlap with it.
- In IMU-1, a 30% decay period is worse than 20%, the opposite direction from the fourth bundle's 75%. The bundle tries it anyway, but if the result is bad, this is the first thing to revisit.
