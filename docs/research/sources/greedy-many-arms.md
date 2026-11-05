# The Unreasonable Effectiveness of Greedy Algorithms in Multi-Armed Bandit with Many Arms

Bayati et al. (arXiv 2002.10121, NeurIPS 2020).

- It covers the case where the number of candidates k is at least the square root of the budget T. Trying each candidate once already costs a regret proportional to k, so standard UCB explores too much. Narrowing down to about √T candidates makes it optimal in how the regret grows.
- In the experiments, a greedy method that keeps choosing whichever candidate looks best did better than the narrowed UCB. The authors explain this by good candidates being found early when there are many of them (free exploration).
- The candidates' rewards are assumed to be independent.

The abstract, the introduction and the captions of the main figures were read. The proofs were not read.
