# Survey of the latest research

On 2026-11-02, 2026 research and public records that might help make a better model at the same cost were searched for. The records of `language-model` up to October were read first, and what they lacked was added. The results served as material for ordering the candidates.

## How the search was done, and its limits

Search, modded-nanogpt's table of records and its PRs, and arXiv's new submissions in cs.LG and cs.CL on 2026-11-02.

- The arXiv new submissions are one day's worth, and only the first 45 or so that could be fetched were looked at.
- Many papers were read through summaries. Numbers inside figures were not read.
- Recursive's post and OpenAI's Parameter Golf post could not be fetched. For awards other than ICML's, the official lists were not checked.
- Comparisons at the same cost (compute) are mixed with comparisons by the number of updates needed to reach a loss.

## What was read

| Work | When | What was learned | Depth of reading |
|---|---|---|---|
| speedrun record #92 (modded-nanogpt-anvil2.md) | 2026-08 | Time to reach 3.28 on 8 H100s from 73.9 s to 39.9 s. Has results with each technique removed | The full record |
| Recursive (recursive-first-steps.md) | 2026-06 | BPB from .9372 to .9109 on a task of the same form as ours | README, results, part of the code |
| COUNT (count-ngram.md) | 2026-10 | Adding counted tables to the input is worth 46–89% more compute in validation loss and 6–56% more in task loss | The body through a summary; numbers only as overall ranges |
| Low-Bit N-Gram (arXiv 2610.31646) | 2026-10 | Learned tables in 4 bits: memory −75%, with over 99% of the effect kept. DeepSeek-V4.1-Flash has 196 billion parameters in Engram | Abstract |
| speedrun PR #379 (CPLM) | 2026-10 | Mixes into the output a pointer that copies earlier tokens of the document; time −11.25%. Not adopted | The PR's description |
| Muown (muown.md), T-Muon (t-muon.md), FARO (faro.md) | 2026-05 to 10 | Improvements to Muon; loss about 0.4–1.1% | Abstract and part of the body |
| Fast Catch-up (fast-catch-up.md) | 2026-02 | Holds that the batch is best switched late | Part of the body |
| A list of what was tried and did not work (modded-nanogpt-wr-blog.md) | 2026-05 | mHC, xIELU, AdEMAMix and others lost in the speedrun | The post |
| AMUSE (arXiv 2605.22432) | 2026-05 | A Muon that needs no learning-rate schedule. No numbers in the abstract | Abstract |
| Newton-Muon (arXiv 2604.01472) | 2026-04 | 6% fewer updates, about 4% less time | Abstract |
| Lngram (arXiv 2605.24869) | 2026-05 | Looks up n-gram tables with symbols made from the hidden state. No numbers in the abstract | Abstract |
| Extra-Merge (arXiv 2605.26484) | 2026-05 | Moves the weights past the average, in its direction. Results with AdamW; numbers only inside figures | Part of the body |
| A study separating the effects of subwords (arXiv 2604.27263) | 2026-04 | The main causes of the gap with byte models are the amount read and word boundaries. Results from a 1.7-billion-parameter model | The body through a summary |

Reread in the records of `language-model`: COUNT, Low-Bit N-Gram, record #92, the proposals after #92, Rethinking Pre-pretraining, When Does Muon Actually Win, The Best Optimizer Depends on Batch Size, Terminal Shrinkage Averaging, Rewriting Scaling Laws, Compute Optimal Tokenization, and the notes on how to lower the learning rate and on batch and context schedules.

## What was learned

- **n-gram tables are the biggest trend:** Looking up a table costs almost no computation; what grows is mainly memory. The speedrun records, Recursive's solution, COUNT and DeepSeek's models use them independently. Within this survey, only COUNT reported multiple-choice scores, but a broader search finds them in STEM, L³ and others too (broad-research-survey.md).
- **There are examples ahead of the current bundle:** Recursive reached .9109 on a task of the same form. A B200's peak is about 2.3× an H100's, so its 5 minutes are about 0.8× the current bundle's compute (our own rough conversion). The current bundle's BPB reading from the head of a document is .936, about the same as the community's best, .9372. But the row length (1,024 and 2,048), the packing of documents and the quantity measured differ, so this is only a rough guide.
- **Improvements to the update rule are small:** In record #92, most of the difference the optimizer makes comes from how the final weights are made; the update rule itself is worth 4.9 millinats.
- **Pretraining on synthetic data seems unnecessary:** In Rethinking Pre-pretraining, it never beat training on natural text alone at the same compute.
- **Material for judging the data:** In Rewriting Scaling Laws, the best share of FinePhrase to mix in at 3×10¹⁸ FLOPs is under 10% (Paloma). The current bundle's compute is 2.7×10¹⁷, and in v6's English stable phase FinePhrase was 30% by the mixing and 58% by the starts of the rows read. This cannot be said to be why the score fell in the v6 comparison.
- **Most report losses:** In the second bundle, correct-answer BPB fell but accuracy did not clearly rise, so loss numbers are not read directly as score gains.

## Not checked

- None of the techniques was measured at the current scale (54 million parameters, vocabulary 8,192, one H100).
- COUNT's numbers for 60M–190M. The size and effect of the tables when built from ClimbMix.
- Which changes in Recursive's solution worked.
