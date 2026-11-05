# Darwin Gödel Machine

Zhang et al. (arXiv 2505.22954, 2025). A coding agent rewrites its own code, checks it on benchmarks, and keeps every past agent so that it can branch from any of them.

- Over 80 iterations, SWE-bench goes from 20.0% to 50.0% and Polyglot (50 problems) from 14.0% to 38.0%.
- A variant that keeps only the latest agent struggled to improve after a bad change and lost to the variant that keeps them all.
- The best agent later came from a branch that had temporarily fallen below its parent.

The introduction and the main part of the experiments were read. The appendix was not read.
