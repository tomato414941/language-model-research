# CatBoost

Prokhorenkova et al., CatBoost: unbiased boosting with categorical features (arXiv 1706.09516, NeurIPS 2018). A paper on a decision-tree method, not about language models. It describes what happens when values aggregated from the data are used as features and the same data serves both the aggregation and the training.

## Leakage through the aggregated values (§3.2)

On using the mean of the answers per category as a feature (target statistic).

- **Aggregating as is:** an example's feature contains that example's own answer (target leakage). The feature's distribution shifts between the training examples and the evaluation examples.
- **An extreme case:** when every category value is different and the answers are split half and half, training gets every example right with one split, while evaluation gets an accuracy of 0.5.
- **Splitting the data (holdout):** aggregate on one part and train on the other. The shift disappears. The drawback is that both the aggregation and the training lose much of the data.
- **Leaving out only oneself (leave-one-out):** does not prevent the leakage. When there is only one category, an example's own answer can be read from the aggregated value, so training can get every example right.
- **Fixing an order (ordered):** shuffle the data, aggregate only over the examples before each one, and use all of them for evaluation. There is no shift, and all the data is used. The aggregates of the early examples vary widely, so several orders are used.

Read: §3.2 of the PDF, and the sentences in §1 and §4 that mention leakage. Ordered boosting (§4 onward) and the experimental results were not read.
