# FARO

Wu et al., FARO: Function-Space Adaptive Return-to-Risk Optimization (arXiv 2610.40858, 2026-10).

An optimizer that redistributes the per-layer update budget during training. 11 layers, width 768, vocabulary 50,257, FineWeb.

- Over the full training, FARO-Adaptive reached 3.2780, and NorMuonAndAdam, the best of the hand-assembled combinations, 3.2794.
- In an 800-step comparison, adding ERC, which allocates for equal risk, gives 3.5881 instead of 3.6017 (−0.0136). It was better on all 6 seeds. The difference per step is about 2 milliseconds, and memory is the same.

Read: the abstract, and about the first 75% of the HTML body through a summary.
