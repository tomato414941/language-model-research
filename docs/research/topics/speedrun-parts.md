# Parts from the speedrun

Four of the parts adopted in modded-nanogpt's records, put into the third bundle on 2026-11-04. On the same day the bundle was about 1% better than the second bundle in validation BPB read from the start of documents ([the third bundle](../experiments/third-bundle.md)), but the effect of each one was not separated. Bundles are measured as a whole, so parts likely to help and cheap to add went in even if each one's effect is smaller than the seed noise (about 0.3%). Their details are in [the records' history](../sources/modded-nanogpt-records.md). The settings are in `configs/bundle3.py`, and each can be switched off.

The expected effects convert the time or the number of updates saved in the speedrun to the current scale, using the conversion that 2× compute is worth about 3.9% (our own calculation).

## Attention gate (`attention_gate`)

- **Form:** Multiplies each head's output by a gate between 0 and 1, made from the first 12 channels of the attention input. Its weights start at 0, and at first every head is half open. The weights are updated with Muon, like the value embeddings' gate (as in record 28).
- **Reason:** In record 28, the updates needed to reach 3.28 fell by about 3% (about 0.15%). The author says that since a head can choose to do nothing, there is less need to use BOS as a place to direct attention. The weakness on rows that start mid-document (document-rows.md) might shrink too.
- **Weaknesses:** Only one record, and the time per update rises slightly.

## Partial Key Offset (`key_offset`)

- **Form:** Leaves half of each head's channel pairs unrotated (the other half covers the whole range of rotation), and takes the keys of the unrotated channels from the previous position. This lets a single layer see "what came after the same token appeared before". Record 49 uses it only in the long-window layers, but says using it in all layers gave the lowest loss, so it is used in all layers. Adds no weights.
- **Reason:** Record 49 got about 1.8% shorter with five changes, and says most of that is due to this (about 0.1%). [Marin's post](../sources/marin-moe.md) puts it at 1.2× the compute, and says half the effect comes from rotating only some dimensions (rope.md).
- **Weaknesses:** Marin's number is on top of MoE. Rotating only some dimensions was slightly worse in nanochat.

## MUDD (`dense_connections`)

- **Form:** After the second-to-last layer, a small MLP (width 64) decides, per position, weights for the stream right after the embeddings and for the outputs of the layer two before and of its own layer. One mixture is added to the stream, and another to the last layer's V. After the last layer, the stream right after the embeddings and the second-to-last layer's output are added to the stream. The exit weights start at 0, and at first it does nothing. The choice of layers is record 81's 11-layer form adapted to 10 layers. The weights use AdamW, at a quarter of the other weights' learning rate, with β of 0.9 and 0.99 (the same ratio and values as record 81).
- **Reason:** In record 81, the loss fell by 0.26% at the same number of updates, and the time per update rose by 1.8% (about 0.15% net). It is the original [MUDDFormer](https://arxiv.org/abs/2502.12170) cut down to fit this scale.
- **Weaknesses:** Record 81's author is one of MUDDFormer's authors. The choice of layers and the coefficients (1.15, 0.2/√L) are the record's, not re-chosen for our shape.

## backout (`backout`)

- **Form:** Keeps the stream after the 7th layer (of 10), and before the final normalization subtracts it multiplied by a learned coefficient (starting at 0.5). The idea is to remove from the prediction the information added only as context for later layers. The coefficient has the same learning rate as the x0 weights.
- **Reason:** In a record of the medium track (PR #139), the updates to reach the target loss went from 5,590 to 5,550 (about 0.04%). It is also in the short track's record 40 and in record 81's form.
- **Weaknesses:** The effect is small, and in the short track it came together with other tuning, without being separated.

## What would change the view

- The bundle scoring worse when measured. The parts were not measured separately, so first remove these four and review.
- The time per update rising much more than expected. At the same cost, the number of updates falls.
