# How to run experiments

What to measure, and in what order, when there are many candidates. This too is a problem of optimizing on a limited budget. The views here are provisional as of 2026-11, and the only round we have run ourselves is [the first one](../experiments/first-bundle.md).

## Current plan

Candidates and yardsticks both keep growing, so instead of making an overall plan up front, we fix how each round chooses. Holding the best model and the growing list of candidates (candidates.md), repeat the following.

1. Read and filter. Reading a paper is an experiment cheaper than training.
2. Choose the next bundle, starting from the highest expected effect per cost.
3. Measure each bundle once as a whole, at the same cost.
4. If it wins, update the baseline; if it loses, split it in half to find the cause. Look into which change worked only when it needs to be known.
5. Keep the records and the weights of losing bundles too.
6. Record, then decide whether to dig deeper or widen the scope.

## Rules of thumb

- **Do not measure differences smaller than the noise:** Changing the seed moves validation BPB by about 0.3% (scoring.md). Smaller effects cannot be told apart in one comparison, so decide them on the papers' evidence.
- **Bundle size:** If each change does harm with probability q, the chance of passing in one run falls to even at about 0.7/q changes (our own calculation). For well-researched changes with q at 10%, that is about 7. Candidates tried without reading rarely hit ([Auto Research](../sources/auto-research.md): about 3.8% of all trials, about 7% of valid trials), so large bundles presuppose reading and filtering. The first bundle was nine changes, read and filtered, and won big in one run. The second bundle was ten changes; it won on correct-answer BPB, and on validation BPB it appeared to lose because of one change's side effect.
- **Narrow the candidates:** With a budget of T runs, keep the candidates on the order of √T ([Bayati et al.](../sources/greedy-many-arms.md), [Carpentier et al.](../sources/simple-regret-infinite-arms.md)). This is a theory of how the number scales, not an exact count, and it does not consider how candidates interact. Within the narrowed set, the simple method of using whichever looks best did well in experiments.
- **Records make the search more efficient:** In Auto Research, using the records of past trials gave 16 improvements, and removing them gave 3. For each trial, keep the hypothesis, the changes, the result and the reason for failure.
- **Keep stepping stones:** In [Darwin Gödel Machine](../sources/darwin-godel-machine.md), the best result later came from a branch that had temporarily declined, and the variant keeping only the latest lost. [Lehman et al.](../sources/novelty-search.md) and [POET](../sources/poet.md) also show that chasing only the objective leads into dead ends. These are results in mazes, robots and coding, not confirmed with language models.
- **People set the direction:** Auto Research went only as far as combining existing methods, and what [the preference judge](../sources/ai-scientist-no-drift.md) changed was also the direction of the search. [Rzhetsky et al.](../sources/choosing-experiments.md) recommend taking more risks in mature fields (summary only).
- **Do not compare runs cut short:** According to [Fantastic](../sources/fantastic-pretraining-optimizers.md), the ranking by loss partway through can flip by the end.
- **Confirm with a bundle at the end:** Effects do not multiply. In [Marin](../sources/marin-moe.md), the product of the individual effects was about 2.27×, and the stacked result was 2.1× (the compared compute differed).
- **Stay comparable when the yardstick changes:** Keep the baseline's weights and rescore them with the new yardstick.
- **Do not decide by a single yardstick:** The second bundle lost on validation BPB alone, and won on correct-answer BPB. When they split, keeping the weights makes it possible to break the result down at no cost.

## Comparing a closed set of candidates

Tools for when it is necessary to know which change worked. They assume eight changes at $0.56 per run, not counting the preparation time. The errors are rough values derived from the difference between one pair of seeds (0.3%).

| Method | Training runs | Cost | What it tells | Error in one change's effect |
|---|---|---|---|---|
| All in | 1 | About $0.6 | Only the whole bundle's effect | About 0.3% for the whole bundle |
| Add or remove one at a time | 9 | About $5 | Each change's effect | About 0.3% |
| Half on each time (Plackett-Burman) | 12 | About $7 | Each change's average effect | About 0.12% |
| All combinations | 256 | About $142 | Down to the effects of combinations | About 0.03% |

The half-on method needs code that can switch each change on and off. Interaction effects between two changes appear mixed with other changes' effects. Unless the learning rate on the AdamW side is matched too, Muon's effect is overestimated.

## What is not known

- How to split the budget between promising candidates and those whose payoff cannot be predicted. We provisionally use 80% and 20%, but have found no studies supporting it.
- Auto Research is about the only study that directly addresses the conditions for comparing language models at the same cost, and it is one team's result, not yet peer-reviewed.

## What would change the view

- Bundles losing often in actual runs, or the hit rate differing from what was expected.
- A run of bundles whose effects are buried in the noise. Then stop digging into the same kind of candidate, and widen the scope to the kind of model and the task setup.
