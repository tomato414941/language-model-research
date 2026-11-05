# Normalization

The computation that evens out the size of each layer's input. The baseline, nanoGPT, uses LayerNorm, with learned weights (scales). The candidate forms and computations are these three:

- LayerNorm: subtracts the mean and divides by the standard deviation.
- RMSNorm: does not subtract the mean, and divides by the mean square. Slightly lighter.
- For each, with or without learned weights. In the first bundle these weights number 13,440 in all, and hardly matter for compute.

## What is known

- **LayerNorm and RMSNorm:** About the same quality ([the RMSNorm paper](../sources/rmsnorm.md), [Ceramic](../sources/revisiting-layernorm.md)). On speed, the paper's Transformer shows a 6.9% difference, and Ceramic up to 40% in plain PyTorch and 0 with good kernels. Whether the difference is in the normalization alone or in the whole model differs between sources.
- **Results on the weights are split:**
  - In ordinary settings, having weights is better. In [OLMoE](../sources/olmoe.md) gradient spikes decreased, and [Lingle's post](../sources/parametric-rmsnorm-blog.md) has 2.240 against 2.253 (a 0.6% difference). Both use AdamW, and OLMoE clips gradients.
  - In μP settings that initialize `lm_head` small, no weights is better ([μ-Transfer](../sources/mu-transfer.md)).
- **Small-scale examples:** autoresearch, nanochat and modded-nanogpt (code of the same lineage) all use RMSNorm without weights. LLaMA, Qwen and DeepSeek use weights (according to Lingle's post). We have not found comparisons at the current scale combined with Muon.
- **FlashNorm (arXiv 2407.09577):** According to the abstract, it is about faster inference, and not a candidate for training.

## Current view

In the view as of 2026-11, use RMSNorm without weights. The reasons:

- The result that no weights is better in μP settings is for setups that start `lm_head` small. If the second bundle separates the embeddings, it comes close to this setup (embeddings.md).
- The report that raised gradient spikes as a problem is about gradient clipping, and the Muon bundle does not use clipping.
- autoresearch's set of learning rates runs in this form.

We think the effect on quality is near 0 (around ±0.6%), with speed slightly better at most. It is not a candidate for a large improvement; it goes in to match the shape.

## What would change the view

- Gradients becoming unstable. Then bring the learned weights back.
- Results comparing with and without weights at the current scale.
