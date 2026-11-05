# Candidates

Candidates not yet decided on for lowering the BPB of the current best (`train.py`'s default: [the third bundle](../experiments/third-bundle.md) without the mixing of the last weights) on the three OLMES suites further with 30 minutes of training. Higher entries take priority. The order is provisional as of 2026-11-02 ([the survey of the latest research](../experiments/latest-research-survey.md), [the broad survey](../experiments/broad-research-survey.md)), and most have not been measured at the current scale. The order gave heavy weight to having reports of multiple-choice scores, to an amount read close to ours, and to small effort. Numbers written through 2026-11-04 are expectations for training at the same cost as the baseline (about 13 minutes). New candidates are added, and those decided to go in are removed. What went in is in `train.py`'s default, with the reasons in the respective `topics/` notes and in the commit messages.

- Candidates are not excluded by kind. Besides the model and the training method, speed, price, data and the task setup are included too. Those unlikely to work are only postponed, with reasons, not removed.
- The order is set by the expected effect, the strength of the evidence and the effort. Those with a large effect but weak evidence, and those heavy to implement, are lower down.
- The numbers are rough expectations of the rate by which BPB falls at the same cost, estimated against the baseline before the bundles were measured. Some shrink on top of a bundle. Some overlap, so they cannot be added up. "Unknown" marks those with no basis for a number.
- Most are judgments from past notes or abstracts, without reading the original papers in full. Read them before deciding.

## Measured and found small

| Candidate | Measured result |
|---|---|
| How attention is computed (cuDNN, FlashAttention-2, 3, 4) | Differences per update within 1.6%. About 0.2% for the preparation when a shape is new |

## To look into next

| Candidate | Expected effect | Evidence | Effort |
|---|---|---|---|
| Fewer attention layers ([Composer](../sources/composer.md) has attention and MLP at 1 to 2; record #92 has 7 of 10 layers) | Mean score +0.6–2.8 (350 million to 3 billion; smaller at smaller sizes). At the current size, 1 to 2 cuts compute by about 4% | When Composer searched at 100 million to 200 million, 1 to 1 was best. Record #92 has no number for removing this alone. AIRA-Compose: abstract only. [Survey](../experiments/broad-research-survey.md) | Medium (re-decide where value embeddings and the like go) |
| Tables looked up by token between layers ([L³](../sources/l3.md); engram.md; [STEM](../sources/stem.md), a table inside the MLP, went into the fourth bundle on 2026-11-05) | Mean score +0.8–2 | At the 800 million class the amount read is about the same as ours (about 12 tokens per parameter), but the tables hold 710,000 embeddings per layer, the speed is about 87%, and running fast needs dedicated computation | Large |
| How long the learning rate falls and its floor (now to 0 over the second half; the fourth bundle tries the last 75%; Recursive goes to 5% over 95%, record #92 to 20% over 80%) | Unknown | Split. There is a report that a higher floor makes mixing the last weights more effective (weight-averaging.md). [Note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/lr-schedule.md) | Small |
| Mixing into the output a pointer that copies earlier tokens of the document ([CPLM](../sources/modded-nanogpt-cplm.md)) | About 0.7% (time to reach the same loss −11%). Expected to shrink in our setup | Only the speedrun's PR #379. Not adopted, paper unpublished, a loss-only report, and no comparison varying the pointer's range. There, 5 of the 7 attention layers have windows of 768 tokens or fewer, and the pointer reaches up to 8,192 tokens back in validation. Currently, every layer sees all 1,024 tokens of the row. The scoring code would also need to change to the mixed distribution. Reread when the paper is out | Medium |
| Tuning the bundle's learning rate and weight decay, and the size of an update (the fourth bundle tries 262,144 tokens per update; good values change with the shape, so after the bundle's shape is settled; changing the size changes the learning rate too) | Unknown | Not in the third bundle. There is not enough evidence to decide without measuring, and tuning takes many training runs. It can be measured directly. For the update, the settings that did well in autoresearch's searches are all smaller than the current 524,288 (Karpathy's best is 262,144; [Recursive](../sources/recursive-first-steps.md) uses 147,456, and a form that rises from 131,072 to 262,144). But each came with other changes, and the effect of the size alone was not separated. By nanochat's law (the best size is proportional to the amount read to the power 0.383), the size that fits the current amount read is about 430,000 (our own calculation) | Small |
| Raising the input's learning rate while keeping the embeddings shared, or splitting them partway through training | Unknown | Only some records | Medium |
| TIDE (several per-token tables, mixed and added at each layer) | Unknown | 750 million and up, Adam only. residual-weights.md | Medium |
| Packing similar documents into the same row (BM25) | Unknown, possibly large | One study only, −3.2 to −4.1% at 1.3 billion. document-rows.md. Incompatible with attention within documents, which went into the fourth bundle on 2026-11-05 (documents cannot see each other through attention, so the packing has no effect) | Medium |
| The MLP's expansion factor (now 4×; in Primer, 12× helped a lot at about 35 million) | Unknown, possibly large | One study only, context 64 | Small |
| Better GPU efficiency (up to nanochat's level of about 46%; for example by dropping autocast) | About 1.5% | Medium | Medium |
| A cheaper H100 (the model does not change) | About 1.2% | By calculation | Small |
| Reading easy documents first | Unknown (1–3% in accuracy) | One study only, unverified on BPB | Medium |
| Attention Residuals | 0.3–1.7% | Medium | Medium |
| Training on patches | Up to about 4% | One study only, results at 970 tokens per parameter or more | Medium |
| Hybrids with linear attention (Gated DeltaNet, Kimi Delta Attention, [Mamba-3](../sources/mamba-3.md)) | 1–2%. Mamba-3: score +0.7 at 180 million, +2.5 at 880 million | Medium | Large |
| Large tables looked up by the hidden state (memory layers; [product keys](https://arxiv.org/abs/1907.05242), [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)) | Unknown. Put into 12 layers, beats 24 layers (2019). Beats ordinary models with twice the compute or more, and MoE matched in compute and parameters (2024, up to 8 billion, 1 trillion tokens) | Two studies, abstracts only. Numbers at the current scale unverified. Especially effective on tasks that ask about facts | Large |
| Leaving unimportant tokens out of backpropagation (CENTRIFUGE) | Time −34.7% (1.1 billion and up) | One study; needs a dedicated implementation | Large |
| Sparse or low-rank MLPs (Mixed Sparsity Training, CR-Net, Bi-Phase) | Reports of a quarter of the FLOPs. Wall-clock time unverified | Weak to medium | Large |
| Approximating normalization with constants (anGPT) | Up to 40% faster convergence (against a GPT with QK-norm). 3% more time | One study | Medium |
| Starting from a pruned or distilled larger model (a different starting point) | Unknown. Reported to help more the less is read | Pruning vs. Scratch, Where to Begin. There are also reports of no difference at the same compute. [Survey](../experiments/broad-research-survey.md) | Medium |
| mHC | Up to 1.3% | Weak to medium. Worse in the speedrun | Medium |
| Data selection (DCLM-Edu, PreSelect and others. ClimbMix was the best of those nanochat tried, and only 0.2% of its 400B is read now. Switching to v6's English lowered the mean over six tasks by .025, but the cause was not isolated. v6-english-data.md) | Unknown, possibly large | Split | Medium |
| Repeating data | Unknown | Medium | Small |
| Vocabulary size | Unknown, possibly large | Large scale only | Medium |
| SOAP-Muon | About 0.9% (steps only) | Medium | Medium |
| Shorter preparation time (helps only at small scale) | About 1.6% | Calculated from measurements | Medium |
| EMA-Nesterov | About 0.35% | Weak | Small |
| MoE | 0 to a few % | Weak at the current scale | Large |
| Context length (autoresearch uses 2,048; 2,048 makes an update 7.5% slower) | Unknown | Weak | Small |
| Growing the context from short | Unknown | Weak | Medium |
| Splitting attention into windows (SSSL) | Unknown (speed) | Weak | Small |
| Interruptible rental (the model does not change) | Up to about 6.9% | Stopped in another project's trial | Medium |
| Other GPUs (4090, 5090, B200 and others. [Survey](../experiments/broad-research-survey.md)) | About 1.8% if the 4090 gives about 1.4× in compute terms (a rough calculation). B200 about 3×, depending on price | A subproject. Not measured | Medium |

## Candidates likely to have small effects

| Candidate | Expected effect | Evidence |
|---|---|---|
| Value residual (overlaps with value embeddings) | About 0.4% | Weak |
| Layernorm scaling | 0–0.4% | Split |
| z-loss | Unknown | Weak |
| Muon+, MuonEq | Very small | Weak |
| Anvil's update rule (about 0.15% in record #92), FARO, Newton-Muon, radial brake, u/w floor, PowerCool | Small or unknown | Weak |
| Scion, Kron | Favored with more training | Medium |
| MuonH | Unknown (compared only with Adam-type optimizers) | Medium |
| NAdamW, Mars, AdEMAMix, Schedule-Free AdamW | Likely small under the current conditions | Split |
| Other data (Ultra-FineWeb, Essential-Web, FinePhrase and others) | Unknown | Not examined |
| Splitting numbers into single digits; the splitting regex | Unknown | Weak |
| Narrow QK (loss worsens in exchange for speed; record #92 makes QK 64-dimensional in 5 of its 7 attention layers and uses a modified FlashAttention-3) | In record #92, time −11%, loss +0.2% (including other attention changes). About 0.4% when converted to the same cost | Weak (record #92 is the only number at the current scale) |
| Liger, shorter compile time | Small | Weak |
| Raising the gradient to a power element by element ([GradPower](../sources/gradpower.md)) | The paper's p=1.2 likely makes things worse. The opposite, p=0.9, gives 0.1–0.3% (no larger than the seed noise of about 0.3%) | The paper used 131,072 tokens per update: 0.4–0.9% with AdamW, about 0.5% with Muon. It says the best p falls as updates grow, and at 524,288 tokens even AdamW gets about 0.3%. In [Parameter Golf](../sources/parameter-golf-gradpower.md) (17 million), with updates the same size as ours and Muon, p=1.2 was about 1.4% worse and 0.9 gave −0.14 to −0.27%. In [Marin](../sources/marin-belay.md) (MoE, MuonH), 0.9 made no difference. There are no numbers with NorMuon. Revisit if the update size changes |
| Orthogonalizing Q, K and V per head ([T-Muon](../sources/t-muon.md), [Group Muon](../sources/group-muon.md); currently Q, K and V are orthogonalized separately) | Likely small with NorMuon (about 0.04% in the speedrun). 0.6–0.7% with plain Muon | T-Muon, with 524,288 tokens per update as ours, context 1,024 and three seeds, gets 0.6–0.7% lower loss on LLaMA-type models of 60 million to 350 million. But with plain Muon. In NorMuon's [speedrun](../sources/modded-nanogpt-paired-head-muon.md), splitting K and V into groups of two heads gave −0.0012 (steps −0.7%). In Group Muon (plain Muon, no weight decay), splitting fully per head was slightly worse at the end (+0.06%). NorMuon's authors say splitting Q, K and V gave no improvement, and we think the gain from finer splitting overlaps with NorMuon (our own view). Revisit when dropping NorMuon |
| Updating the rows' norms separately with Adam ([Muown](../sources/muown.md)) | About 0.1% against the current NorMuon (124 million) | 0.6–1.1% lower than plain Muon (124 million to 2.7 billion; one seed at 1 billion and up). The comparison with NorMuon is only a figure at 124 million, both without weight decay. We now pair NorMuon with cautious weight decay, and Muown would replace NorMuon. The current bundle starts the output projections at 0, so it cannot be used as is (change the initialization, or exclude them). Revisit when dropping NorMuon |
| Mixing in a few rows that start mid-document | 0 or slightly worse on the measured scores | The weakness is only in the first 64 tokens of rows that start mid-document without BOS (+47% in the second bundle). The measured tasks and the correct-answer BPB all start from BOS. We have not found studies comparing ways of mixing. nanochat and modded-nanogpt start training rows with BOS. When reading from mid-document, try putting BOS at the start before changing training (document-rows.md) |
| Starting updates small and growing them (record #92 goes from about 57,000 to 590,000) | Small. nanochat got only a small gain and did not put it in (the sizes of the schedule tried are not given). The speedrun's record #46 cut the time to reach the same loss by 1.4% (about 0.1% roughly converted to the current scale) | nanochat's LOG, record #46, [Fast Catch-up](../sources/fast-catch-up.md) (AdamW, 33,000 to 66,000 tokens per update; numbers only in figures). [Note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/batch-size-schedule.md) |

## Passed over for the fourth bundle on 2026-11-05

Read when choosing from outside the speedrun, and not put in.

| Candidate | Why it was passed over |
|---|---|
| Canon layers (short convolutions added before and after layers; Physics of Language Models 4.1, Primer's MDHA) | On real data at 1.3 billion and 100 billion tokens, adding them to RoPE gave a difference within the noise. They help in setups without position information, or with linear attention. We have RoPE and the key shift |
| Normalizing the outputs too (Peri-LN, Gemma, OLMo 2) | With our output weights starting at 0, each block's output would suddenly jump to size 1 at the first update. Even if the attention gate brings the output close to 0, the normalization scales it back up |
| Differential Attention, Forgetting Transformer, Selective Attention | The results are at 800 million or more, or the attention computation has to be rebuilt, which would make it slower than the current FlashAttention |
| Memory layers (product keys, Memory Layers at Scale) | Heavy costs in implementation and speed. Measure STEM and the n-gram tables, from the same line of work, first |
| Prepending the URL to documents (MeCo) | FineWeb has URLs, but there is almost no difference at the smallest size, 600 million |
| Making attention local-only in some layers (Gemma 3, gpt-oss) | After attention within documents has already reduced the compute, the gain is smaller still |

## Candidates unlikely to work at the current scale

| Candidate | Why postponed |
|---|---|
| Engram's full form | At small scale, no different from a two-token table (nanochat's tables are small; engram.md) |
| Screening | Its main benefit is long context |
| H-Net | The structure changes a lot. Few comparisons |
| CompleteP | It helps deep models |
| Per-layer embeddings | Contents unverified |
| Sampled softmax | Little gain because the vocabulary is small. Record #92's −12% time is with a vocabulary of 50,304, and includes fusing the loss computation |
| Splitting long documents | Almost no difference |
| MTP, SwiGLU, LeakyReLU², XSA, orthogonal initialization | In nanochat they did not help, or lost on time |
| FP8 | Actually slower at 12 layers. Record #92 got −7% time with dedicated kernels |
| GQA | Mainly a technique for inference |
| RETRO-style retrieval | Estimated 2–6% worse (unless the database is made huge) |
| Pretraining on synthetic data | At the same compute, it did not beat training on natural text alone ([note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/pre-pretraining.md)) |
| Training with metadata prepended to documents and removing it at the end ([MeCo](../sources/meco.md), KoCo) | MeCo is almost no different at its smallest size, 600 million (score +0.2), and its perplexity is slightly worse even at 1.6 billion, 13.2 to 13.3. ClimbMix has no URLs, and making topics takes compute (about 1,500 GPU hours in the paper). KoCo and Beyond URLs: not read. [Survey](../experiments/broad-research-survey.md) |
| Starting shallow and deepening later ([Deep Progressive Training](../sources/deep-progressive-training.md)) | After growing, catching up with a model deep from the start takes about 16 billion tokens (said not to depend on model size), about 24× our whole training run (about 670 million). All results are at 50 tokens per parameter or more, with no reports of scores. Other studies on adding layers: not read. Revisit when training gets much longer |

## Candidates BPB cannot measure

Order them afresh when the task setup changes.

- Instruction tuning (SFT), preference learning (APO), instruction data
- Reinforcement learning on correct answers (RLVR)
- Tools, harnesses, and the ability to look up facts by search
- Building programs into the weights (Qwenputer)
- Setups that also count inference cost (Train-to-Test; training small models longer)
- Japanese, multiple languages, Japanese evaluation
- Confirming effects at a second scale

Views on individual items are in the same `topics/`, and many others are in `language-model`'s [`topics/`](https://github.com/tomato414941/language-model/tree/main/docs/research/topics).
