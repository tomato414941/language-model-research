# Auto Research with Specialist Agents

Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes (arXiv 2605.05724, 2026-05). It studies, in three environments, a loop in which AI proposes, implements, measures and revises training recipes.

- 1,197 trials in total over the three main runs. 45 were kept as improvements, and 592 were valid but did not improve.
- Parameter Golf (FineWeb, 16MB, 10 minutes): BPB from 1.0810 to 1.0722 in 900 trials (−0.81%).
- NanoChat-D12: CORE from 0.1618 to 0.2244 in 200 trials (+38.7%), with 5 improvements. The largest gain (+0.0334) came from a change that fixed the attention path and spent the time saved on more tokens.
- CIFAR-10: from 26.356 seconds to 25.1464 seconds in 97 trials (−4.59%).
- The effect of a record of past trials (the same 200 trials of Parameter Golf): with the record, 16 improvements and 1.0731; without it, 3 improvements and 1.0774, then no improvement over the next 125 trials.
- Splitting the roles widens the variety of proposals (effectively 74.1 groups against 39.3; 91 past records).
- The authors say that what it produced went no further than combining and modifying existing methods, and that no fundamentally new architecture emerged.

Read: the abstract, the introduction, the method, and the main parts of the experiments and the discussion. The appendix was not read.
