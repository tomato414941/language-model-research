# Mixing the last weights

A method that averages the last few weights of training and uses the average mixed with the final weights. The training computation does not change. It was put into the third bundle on 2026-11-03, and when measured on 2026-11-04, the mixed weights were worse than the final weights (see "Measured result" below). It is dropped from the next bundle on.

## Reports

- [Terminal Shrinkage Averaging](../sources/terminal-shrinkage-averaging.md) (TSA), on nanochat's 12 layers, mixed the average of 8 checkpoints (every 32 updates, including the final weights) with the final weights, and BPB fell by about 0.0008 (about 0.1%). Using the average alone is worse than the final weights. The higher the learning rate's floor, the worse the final weights and the better the mixed weights. A schedule falling to 0 was not tried.
- In the speedrun's [record #92](../sources/modded-nanogpt-anvil2.md), removing the way the final weights are made (mixing in an EMA or an average) makes the loss 20.2 millinats (about 0.6%) worse. The learning rate stops at 20% of the peak.
- IMU-1 applied an EMA over the last 10, and the mean score was +0.014. In nanochat's LOG, EMA and SWA did not help ([`language-model`'s note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/checkpoint-averaging.md)).

## The form in the third bundle

The settings are in `configs/bundle3.py`, and the mixing in `averaging.py`.

- **How to mix:** Uses the values TSA's authors had fixed before their experiments. Average 8 checkpoints taken every 32 updates (including the final weights), and add the average at 55% and the final weights at 45%. Only the parameters are averaged.
- **Span:** The current bundle makes 1,281 updates, so the 8 checkpoints span the last 224 updates (about 17% of training), over which the learning rate falls from about 35% of its maximum to 0. The paper had 3,000 updates, and the span was 7.5% of training.
- **The learning rate stays as it is:** It falls linearly to 0 over the second half. A higher floor was not put in, because there is a study finding a linear fall to 0 best (Straight to Zero, [`language-model`'s note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/lr-schedule.md)), and in the TSA paper too, the mixed weights' difference between floors of 5% and 15% is small, about 0.0001.
- **How to measure:** Score both the final weights and the mixed weights of the same training run (`rented.sh`). Since the difference is within the same run, it can be read even if it is smaller than the seed noise (about 0.3%). Across the paper's five training runs, the noise in the difference was ±0.00009.
- **How to read:** As of 2026-11-03, the bundle's result is read from the mixed weights. If the mixed weights are worse than the final weights, this bundle's result is read from the final weights, and the next bundle drops the mixing.
- **Cost:** Training time barely changes (the weights are added 8 times and saved once at the end). Scoring grows by one round: about 3 minutes at the last H100 run's timing, about $0.11. The short measurement in `speed.py` does not reach the updates that are averaged, so it does not average.

## Measured result

In the same training run of [the third bundle](../experiments/third-bundle.md), the mixed weights were worse than the final weights by −.0029 ± .0011 in mean accuracy over six tasks and by +1.2% in mean correct-answer BPB, and validation BPB was the same (1.1633 and 1.1634). This is larger than the standard error of the differences on the same questions.

- Matched to the same cost, the third bundle made 910 updates, and the 8 checkpoints spanned the last 224 updates (about 25% of training). Over that span the learning rate falls from about 49% of its maximum to 0. That is wider than expected when it was decided (about 17%, from about 35%) and than in the paper (7.5%).
- Whether the span being too wide was the cause was not isolated.

## What would change the view

- Shrinking the mixing span to 10% of training or less. This time it was about 25%.
- Raising the learning rate's floor. There is a report that it makes mixing more effective.
- A large change in the number of updates. The share of training that the 8 checkpoints span changes.
