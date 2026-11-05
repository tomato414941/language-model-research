# Terminal Shrinkage Averaging

Terminal Shrinkage Averaging Reveals a Schedule-Estimator Interaction in LLM Pretraining (arXiv 2609.25482, 2026-09). It returns a mix of the last weights and the average of the last few weights, with a coefficient α (TSA).

## Setup

- The returned weights are (1−α)θ_T + αθ̄_T, a mix of the last weights θ_T and the average θ̄_T of the K weights θ_T, θ_{T−s}, …, θ_{T−(K−1)s}, which include θ_T (§3.1). α=0 gives the last weights, and α=1 the uniform average (LAWA). What is averaged is the weights; the optimizer state is not handled.
- The main experiments use nanochat's 12 layers, with nanochat's Muon and AdamW as the optimizers. The learning rate falls along the original schedule, stops at the floor ρ once it reaches it, and does not go down to 0. The baseline floor is 5% of the maximum (Appendix A.1, §4.4).
- Training is 3,000 updates, with the first 40 updates as warmup. 8 weights are averaged, 32 updates apart. The 8 span the 224 updates ending at update 3,000 (about 7.5% of training) (Appendix A.1).

## Results

- 12 layers, a 5% floor, 5 training runs (§4.1): with α=0.40, BPB falls by 0.000794 ± 0.000087 from the last weights. With α=0.55, fixed in advance, by 0.000711. α=1 (LAWA), which uses only the average, is worse by 0.000734.
- Across 9 variants of the number of weights averaged and their spacing, every one was better than the last weights, and the best was 16 weights 32 updates apart, at 0.002363 (Appendix B.2).
- A comparison of floors (Table 1; differences from the last weights at a 5% floor; α=0.55):

| Floor | Last weights | TSA |
|---|---|---|
| 10% | +0.000915 | −0.000487 |
| 15% | +0.003757 | −0.000844 |

  Raising the floor makes the last weights worse and the mixed weights better. With AdamW alone, the direction is the same and the size smaller. 0% (falling to 0) was not tried.
- 22 layers, 3 seeds (Table 2): with a 15% floor and α=0.70, validation BPB goes from 0.720235 to 0.719580 (a difference of 0.000654 ± 0.000184). DCLM CORE goes from 0.2491 to 0.2579, but the noise is large, and the authors also treat it as provisional.

Read: the abstract on arXiv, and the relevant parts of §3.1, §4.1–4.5, Tables 1 and 2, and Appendices A.1 and B.2 of the HTML version (from quotes extracted with a tool). The theoretical part (a local quadratic approximation) was not read.
