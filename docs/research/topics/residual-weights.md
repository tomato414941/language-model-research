# x0 and resid weights

Two numbers learned per layer, applied before each block as `x = resid[i] * x + x0[i] * x0`. `x0` is the value right after the embedding, and it brings the token's own information to deep layers too. The only added weights are twice the number of layers. `language-model` has short notes ([x0](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/x0-lambdas.md), [resid](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/resid-lambdas.md)).

## What is known

The differences are our own calculations.

| Source | Conditions | What was added | Result |
|---|---|---|---|
| [nanochat](../sources/nanochat-log.md) (01-11) | Muon, 8–20 layers, BPB | x0 and resid | 8 layers −0.95%, 12 layers −0.79%, 16 layers −0.63%, 20 layers −0.46% |
| [modded-nanogpt's record 9](../sources/modded-nanogpt-records.md) | Muon, 12 layers, width 768, 3,200 updates | Shortcut (after value residual) | −0.28%. Each update went from 146 to 152 milliseconds (4.1% slower) |
| [Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md) | 150 million to 280 million, added one at a time | resid, then x0 | −0.25%, −0.08% |

- **nanochat's figures run high:** they come from sweeping the learning rate and picking the best value, and the comparison predates value embeddings. Value embeddings also bring token information to deep layers, so the two may overlap (our reading). Record 9 comes after value residual, and its difference is small.
- **Cost:** on the H100 of [the second bundle](../experiments/second-bundle.md), each update is 3.1% slower (0.597 and 0.615 seconds; +0.17% when converted). nanochat says computation barely increases, and record 9 is 4.1% slower.
- **Learning rate:** nanochat says 0.5 is fine for the added x0, while the multiplying resid does not work well unless it is 1/100. For x0, a beta1 of 0.96 is best, and 0.98 is much worse (d20). [autoresearch](../sources/autoresearch.md) starts x0 at 0.1 and resid at 1.0, with learning rates of 0.5 and 0.005. The current nanochat starts x0 at values falling from 0.20 to 0.05 and resid from 1.15 to 1.05, layer by layer.
- **The best learning rate depends on depth:** nanochat's x0 values are 0.2 at 8 layers, 0.6 at 12, 0.2 at 16 and 0.1 at 20. The current 0.5 is untuned.
- **The start of training:** in a trial with the second bundle's shape at a batch about 1/500 of the real one, the version with the weights had its loss jump at the 4th update (9.36 against 8.04) and had nearly caught up by around the 30th ([the check's record](../experiments/second-bundle-code-check.md)). It is not a comparison of quality.
- **Learned values:** in [a post that examined one training run](../sources/modded-nanogpt-lambdas.md), the x0 weights are large in many layers. It is not a with-and-without comparison.

**Adoption:** it is used by nanochat, autoresearch and modded-nanogpt (code of the same lineage) and in Marin's comparison; no adoption in a large model has been found. modded-nanogpt's optimization track leaves it out as an "unusual component".

**Other forms of a similar idea:**

- [TIDE](../sources/tide.md) (Apple): holds several per-token tables, mixes them and adds the result at each layer. It works at 750 million to 3 billion, mostly on rare tokens. With the current vocabulary the tables are small, but there are no results with Muon or at the current scale.
- [KromHC](../sources/kromhc.md), mHC: split the residual stream into several. Against nanochat's x0 and resid form, −0.48% at 6 layers and −0.35% at 12 layers (mHC). Computation grows with the number of streams. DeepSeek-V4 adopts mHC.
- [Attention Residuals](../sources/attention-residuals.md): selects earlier layers' values with attention.

## Current view

The view as of 2026-11 is to put it in the second bundle. The settings are the same as autoresearch's (x0 from 0.1, learning rate 0.5, beta1 0.96; resid from 1.0, learning rate 0.005). We think the effect at the same cost is 0–0.5%. The reasons are these.

- Every comparison points the good way, and no report of it making things worse has been found.
- It is a one-line change and adds almost no weights.
- A bundle is measured only once, so it follows an example that worked with Muon. autoresearch's set of learning rates also runs with this form.

`x0` is the value right after the embedding, so this assumes the form with separate embeddings and RMSNorm applied (embeddings.md).

TIDE, mHC and Attention Residuals stay candidates.

## Conditions likely to change the view

- When changing the shape changes how much slower it is. If it is more than 5% slower, consider removing it. 5% converts to about 0.27%, close to the middle value of the quality benefit.
- When the overlap with value embeddings is measured.
- When the loss jumps sharply early in training even with the real batch. Lower x0's learning rate.
- When results for TIDE with Muon at the current scale come out.
- When the effect is measured item by item.
