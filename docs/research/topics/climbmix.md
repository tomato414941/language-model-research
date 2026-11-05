# ClimbMix

**Not used from 2026-11-04 on.** The original Nemotron-ClimbMix is non-commercial (CC BY-NC 4.0), so it was decided not to use it in research either. A rule was made that the training data is FineWeb ([training data](training-data.md)). The tokenizer trained on ClimbMix and the COUNT tables counted with it were also rebuilt. The baseline's and the bundles' scores are compared again, dataset by dataset. The H100 rented that same day to train the third bundle was deleted partway through the transfer (about 19 minutes, about $0.75).

The baseline's training data as of 2026-10. It is `karpathy/climbmix-400b-shuffle` (`915333b`), the same as `autoresearch` and nanochat use, so results are easy to compare. How the data is chosen is one of the candidates; if it changes, score the baseline's weights again with the new yardstick.

- English only. About 92MB per file, 6,543 files.
- The distributor lists MIT, but the original Nemotron-ClimbMix is CC BY-NC 4.0.
- It is used as is, with no processing to remove documents that contain evaluation questions. In the 2 billion tokens read, evaluation questions were found in 0.114% of documents (3,071) (478 questions, [the count](../experiments/count-table-check.md)). These are exact matches only; paraphrases were not counted. The difference from v5 is written up in [the measurement on every question](../experiments/harness-milestone.md).
- With the second bundle's shape, replacing only the training data with v6's English lowered the mean accuracy over the 6 tasks from .452 to .427 ([comparison](../experiments/v6-english-data.md)). Even with the found evaluation questions removed, the difference stayed at +.025, so they were not the cause. Which of the mixing, the fit with the tokenizer or reading documents from their start was responsible has not been separated. As of 2026-11, the baseline's data stays ClimbMix, and when it is set beside models trained on other data, a note is added that the scores may move because the data differs. Revisit this once the causes are measured separately.

For the data's origin, its effects and the fact that its mixture was chosen by the scores on some benchmarks, see `language-model`'s [climbmix.md](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/climbmix.md).
