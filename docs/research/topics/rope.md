# RoPE

Conveying position by rotating the query and key instead of with a learned embedding. The baseline nanoGPT uses learned position embeddings.

- **Direct comparison:** in [EleutherAI's post](../sources/rotary-embeddings-blog.md), against learned position embeddings in the GPT form, the validation loss is −1.8% at 125 million and −3.0% at 1.4 billion. The 1.4 billion has about 6 tokens per parameter, close to the current scale (about 14 tokens). It is a 2021 post, a comparison by the people who popularized RoPE, and the number of seeds is also unknown. The optimizer is not Muon.
- **Paper:** [the RoFormer paper](../sources/roformer.md) has no comparison in the GPT form.
- **Speed record:** record 2 of [modded-nanogpt's records](../sources/modded-nanogpt-records.md) is worth 1.43× together with a learning-rate adjustment, with no separate figure.
- **Cost:** computation rises 1–3% (the post's figure). At the current scale that is equivalent to a loss of about 0.1%. The position table (about 660,000 parameters in the first bundle) goes away.
- **Adoption:** autoresearch, nanochat and many frontier models use it.
- **Rotating only some dimensions:** covered in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/partial-rope.md). About 0.1% in Marin, slightly worse in nanochat. For Partial Key Offset in [Marin's post](../sources/marin-moe.md), half of the effect is said to come from this form. A post says Qwen 3.6 rotates 25% of the dimensions, but the original research behind this comparison was not read. The third bundle leaves half of each head's channel pairs unrotated for Partial Key Offset ([parts from the speedrun](speedrun-parts.md)).
- **Alternatives:** a search in 2026-11 found no comparison in which something beats RoPE in language-model training. There is a [paper](../sources/rope-not-relative.md) that questions RoPE, but it proposes no replacement.

## Current view

The view as of 2026-11 is to put it in the second bundle. The expected effect is about 2%.

## Conditions likely to change the view

- When a comparison against learned position embeddings with Muon and the current shape is found, or when we measure it ourselves.
- When the context is made longer. How extended contexts are handled becomes an issue.
