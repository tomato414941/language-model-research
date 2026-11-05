# Scoring

What to measure to see whether a change worked.

- Normally, measure the correct-answer BPB on public benchmarks and validation BPB. This takes about 20 seconds on the H100, small compared with the cost of training ([the baseline's training](../experiments/baseline.md)).
- Just changing the seed moved validation BPB by about 0.3% and the mean correct-answer BPB by about 1.3%. A single benchmark moved by up to 6%. With only two seeds, this is still rough.
- From this, as of 2026-11, we take a difference in a one-run comparison of more than 0.01 in validation BPB, and of more than 0.05 in the mean correct-answer BPB, to be an effect. Revisit after measuring with more seeds.
- From 2026-11-02, every time a bundle is trained, the accuracy on all questions of six tasks is also measured with the scoring harness. The extra cost is about $0.1 per run (about 4 minutes per model on an RTX 4090), and it saves deciding when the milestones are.
- In [the measurement on all questions](../experiments/harness-milestone.md), the gain from the baseline to the first bundle was clearly visible in accuracy too, but the difference from the first bundle to the second bundle was within the error (correct-answer BPB fell by 4.4%). As of 2026-11, a difference of a few percent in correct-answer BPB is not read as a rise in accuracy.
- The format of MMLU's authors (five examples, answering with a letter) is not measured for now. In the same measurement, none of the four models could be told apart from guessing's .25, and there was no difference between them. Measure again when building a model large enough that the ability to choose a letter might show.
- Validation BPB reads rows cut at fixed positions. Many rows start mid-document, which puts models trained to read from the start of documents at a disadvantage. [The second bundle](../experiments/second-bundle.md) was +2.2% by this measure and −4.2% on rows read from the start of documents. Do not judge by a single yardstick.
