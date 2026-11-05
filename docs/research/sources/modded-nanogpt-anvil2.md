# modded-nanogpt record #92 (ANVIL2)

`records/track_1_short/2026-08-30_ANVIL2` of `KellerJordan/modded-nanogpt`. The table measured with the methods removed one at a time is in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/modded-nanogpt-anvil2.md); only the numbers that are not there are written here.

## Table rows and validation loss

Measurements during development that vary only the number of rows. Two runs each. The table has two kinds, one for 2 tokens and one for 3 tokens.

| Rows | Time | Validation loss |
|---|---|---|
| 84,602,880 | 39.52 s | 3.2791 |
| 21,150,720 | 39.35 s, 39.36 s | 3.2825, 3.2863 |
| 377,280 | 39.00 s, 38.98 s | 3.3149, 3.3152 |

- The largest table has width 768 and 6.5e10 parameters. It is split over 8 GPUs, 15.13 GiB each. Each update fetches only the rows its tokens look up, and updates only those rows with Adam.
- The table starts at 0 and is learned within the training time.
- In the PR, several people questioned whether the table's size fits the spirit of the record, and no conclusion was reached. KellerJordan regards the change as legitimate, but had not reproduced it personally at the time of the PR, and the merge came after a rough check (2026-09-28).

## The shape of training

- Over three stages, the sequences per GPU rise through 8, 16 and 24 and the cap on document length through 896, 2048 and 3072, and at the end the number of sequences is reduced. With 8 GPUs, an update grows from about 57,000 tokens to about 590,000 tokens (our own calculation, reading the counts as per GPU). 1,194 updates in all.
- The learning rate falls over the last 80%, with a floor at 20% of the peak.
- How the final weights are made: an EMA of the last 298 steps is mixed into the table at 0.65, an average of 250 steps is mixed into the MLPs and values, and more.
- It keeps seven attention layers, removing one layer entirely and the attention of two layers.
- Sampled softmax is used only in the first ~93% of training, with 10,240 candidates out of the vocabulary of 50,304 (14,336 from step 681, 24,576 from step 961). Validation always uses the whole vocabulary.

## What was read

The record's README was read in full. The PR's description and discussion were read through summaries. The code was not read.
