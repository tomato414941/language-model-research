# Broad survey

On 2026-11-02, because latest-research-survey.md from the same day leaned toward pretraining techniques, we searched again across more fields. It served as material for ordering the candidates.

## How we searched, and the limits

- **Per-conference lists by field (papernotes.org):** 79 papers in ICLR 2026's Pretraining, 27 in ICML 2026's Pretraining, 51 in NeurIPS 2025's Pretraining, 12 in ACL 2026's Pretraining, and ICML 2025's Pretraining. From ICLR 2026's and ICML 2026's LLM Efficiency and Optimization and ICML 2026's Model Compression, only the papers related to training were picked up, from their summaries.
- **Searches by field:** distillation, ways to reduce backpropagation, synthetic data, architectures other than the Transformer, compression, BabyLM, recursion, JEPA, prices, hardware, weighting during training, brain-inspired methods, test-time training, and Chinese and Japanese literature.

Limits:

- "List" in the tables below means only the list's machine summary was read, without checking the original. "Search" means only the search summary was read.
- NeurIPS 2026 is barely listed on this site. The official lists of ICML and NeurIPS could not be fetched. EMNLP, COLM and the second half of ICLR's Optimization were not looked at.
- The Chinese and Japanese searches found almost nothing. The basis for MiniMind's cost (about 64 million in about 2 hours) was not found. The NLP2026 list was not looked at.
- The current bundle reads 12 tokens per parameter. In most reports below, the amount read is orders of magnitude larger.

## Replacing computation with tables

| Work | Report | Depth of reading |
|---|---|---|
| STEM (ICLR 2026, stem.md) | Turns the MLP's input side into a per-token table. At 350M, mean score 49.72→50.90 | Full text |
| L³ (ICML 2026, l3.md) | Puts per-token tables inside the layers. At 809M with the same compute, 48.28→50.25 | Full text |
| Hierarchical Memories (arXiv 2510.02375) | A 160 million model that fetches and adds 18 million from a 4.6 billion memory matches ordinary models more than twice its size | Search |
| LMLM (arXiv 2505.15962) | Takes factual answers out of the loss and has them looked up. 382 million approaches 7 billion on factual tasks | Search |
| Scone, Memory Mosaics (NeurIPS 2025) | Tables of frequent n-grams, associative memory | List |

COUNT, record #92 and Recursive are in latest-research-survey.md.

## Reducing computation by changing the shape

| Work | Report | Depth of reading |
|---|---|---|
| Deep Progressive Training (ICML 2026, deep-progressive-training.md) | Trains with 0–1 layers up to 80%, then deepens. Worth 3–5× the compute | Part of the full text |
| Composer (arXiv 2510.00379, ICLR 2026) | A search finds a shape with attention and MLP at 1 to 2. From 350 million to 3 billion, scores rise by 1.1–3.1% on average | Search |
| AIRA-Compose (arXiv 2605.15871) | 11 agents search for shapes in 24 hours. At 1 billion, scores +2.4%, and +3.8% in a hybrid with Mamba | Abstract |
| Mixed Sparsity Training (arXiv 2408.11746) | Sparse training cuts GPT-2's FLOPs to a quarter. Wall-clock time is not in the abstract | Abstract |
| Sparser, Faster, Lighter (ICML 2026) | With L1, over 99% of the MLP becomes 0, with little downstream effect. Training time is not in the abstract | Abstract |
| BLaST (arXiv 2507.03117), CR-Net, Bi-Phase | Sparse or low-rank MLPs. BLaST is 1.11× in pretraining | Search, list |
| CENTRIFUGE (arXiv 2502.00340, ICLR 2026) | Drops unimportant tokens from backpropagation. Dropping 50% gives backpropagation −49.9% and overall −34.7%. 1.1 billion to 40 billion | Search |

## Other families

| Work | Report | Depth of reading |
|---|---|---|
| Mamba-3 (ICLR 2026, mamba-3.md) | At the same parameters, 180 million 42.8→43.5, 880 million 52.8→55.3 | Part of the full text |
| Diffusion language models (arXiv 2610.17590) | The gap in compute needed for the same loss shrinks from 16× to 3.0× with blocks of 4. Still more than ordinary models | Abstract |
| ParaRNN, Energy-Based Transformers, Dual-objective (ICLR 2026) | RNNs, energy, a mix of two objectives | List |
| Training without backpropagation, spiking, predictive coding | No results at language-model scale were found. Hybrid Pretraining, which uses zeroth-order estimates for three quarters of the layers, is known only from secondary sources | Search |
| Tiny Recursive Model (arXiv 2510.04871) | Results on puzzles; no results for language models were found | Search |

## Losses and targets

| Work | Report | Depth of reading |
|---|---|---|
| Dispersion Loss (ICML 2026, dispersion-loss.md) | Adds a loss that spreads the directions of the embeddings. 34.95→35.61 in mid-training of GPT-2 | Part of the full text |
| GradPower (ICML 2026, gradpower.md) | One line that raises the gradient to the power 1.2. At 2 billion, score 47.72→48.26, loss −0.022 | Summary |
| Lost in Backpropagation (lost-in-backpropagation.md) | Claims the output table loses 95–99% of the gradient | Part of the full text |
| Next-ToBE, Future Summaries, LLM-JEPA, CoCoMix (ICLR 2026) | Predicting the bag of future words, predicting a summary of the future, latent prediction, predicting SAE concepts | List |

## Data

| Work | Report | Depth of reading |
|---|---|---|
| MeCo (arXiv 2501.01956, ICML 2025) | Trains with URLs and the like prepended to documents, and removes them for the last 10%. At 1.6 billion, 33% less data needed. Also works with topics generated by a model | Search |
| KoCo (ACL 2026), Beyond URLs (ICLR 2026) | Attaching semantic coordinates, fine-grained metadata | List |
| ADAPT (arXiv 2605.05227, ICLR 2026) | Changes document weights during training. Claims to beat filtering beforehand at the same compute. No numbers in the abstract | Abstract |
| AC-ODM (ICML 2026), Demystifying Data Organization (ACL 2026), How Text Quality Interventions Reshape Neural Scaling Laws (ICLR 2026) | Automatic tuning of the mix, reading order, a comparison of quality interventions over 2,000 models | List |
| Beyond Rephrasing (arXiv 2607.28109) | Synthetic data shaped like books raises mid-training scores by about +1.09 | Search |
| It's All Training (NeurIPS 2026) | Synthesizes everything from encyclopedia seeds and trains in one stage | List |
| Is a Document Educational or Just Wikipedia-Style? (ACL 2026) | Quality classifiers are fooled by a Wikipedia-like style | List |

ClimbMix's only column is `text`, with no metadata such as URLs (checked on 2026-11-02).

## Starting from existing weights or larger models

| Work | Report | Depth of reading |
|---|---|---|
| Small LLMs: Pruning vs. Training from Scratch (arXiv 2606.14150) | Starting from a pruned Llama-3.1-8B beats random initialization at the same tokens. The gap shrinks as the amount read and the pruning rate grow, and nearly vanishes at a rate of 0.8 | Abstract |
| Where to Begin (arXiv 2510.07227) | With a subnetwork and distillation, FLOPs fall to 1/5.16 at 10 billion tokens and 1/1.26 at 100 billion. Whether the teacher's compute is counted is not in the abstract | Abstract |
| Knowledge Distillation vs. Pretraining from Scratch (arXiv 2404.19319) | At the same compute, ordinary distillation and training from scratch are about the same | Search |
| Don't Ignore the Tail, Energy Accounting of Distillation (ICML 2026) | A week of distillation on one H100 is feasible. Distillation uses about 2.4× the energy | List |

## Training procedure

| Work | Report | Depth of reading |
|---|---|---|
| anGPT (arXiv 2505.22014, NeurIPS 2025) | Approximates normalization with constants. Converges up to 40% faster than GPT with QK-norm, at 3% more time, and needs no weight decay or warmup | Search |
| Seesaw, WSM (ICLR 2026) | Growing the batch instead of lowering the learning rate, time −36%. Replacing the decay phase with a merge of weights from a constant learning rate | List |
| AdaGC (ICML 2026) | Per-tensor gradient clipping raises scores by +1.27–2.48% (large models) | List |
| SiameseNorm (arXiv 2602.08064, ICML 2026) | Two streams, Pre and Post. At 1.3 billion, arithmetic 28.1→39.6 | Search |
| Sharpness-Aware Pretraining (arXiv 2605.02105) | From 20 million to 150 million, SAM, a large learning rate and a short decay reduce forgetting in post-training by up to 80% | Abstract |
| Conditioned Initialization (ICLR 2026) | Sets the condition number of the attention matrices' initial values to 1. The figure of 20–30% faster is from secondary sources, and the difference for language is small | Search |
| RMNP (ICML 2026) | Replaces orthogonalization with row normalization and is 13–44× faster. Muon's cost is small now, so there is little to gain | List |
| Dropout Universality (arXiv 2605.21648) | Dropout that strengthens earlier layers. Results on MLPs and ViT; no language-model results | Search |

## Prices and hardware

- Comparing Vast's prices (gpu-prices.md), the RTX 4090's median of $0.47 against the H100's $2.40, the 4090 is about 1.4× better in price per compute (a rough calculation from the speeds of earlier measurements, with different model sizes).
- SemiAnalysis's 2026-07 price survey puts the B200 at $1.73 and the MI355X at $1.50 (per chip-hour; on the inference comparison page). The B200's peak is about 2.3× the H100's, so renting at this price would be about 3× better per compute. Vast's B200 prices were not checked.
- No source comparing the MI355X's training cost per compute was found.
- TPU Research Cloud gives TPUs for free on application. Its official repository was archived in 2026-04, and whether it still accepts applications was not checked. The code would need porting.

## Neighboring fields

- **BabyLM (abstract of the EMNLP 2025 report):** New objectives and architectures came out on top. Training FLOPs and results do not fully correlate. According to the summary of the 2024 report, most tricks with the reading order did not work.
- **Parameter Golf (search summary; the article could not be fetched):** 16MB, 10 minutes on 8 H100s. The top entries include Muon's weight decay, spectral initialization of the embeddings, and per-document test-time training.
- **Hutter Prize:** Advanced greatly in June–July 2026, with enwik9 at 100,424,672 bytes. It uses classic context mixing and preprocessing for text, and training and evaluation use the same data, so the BPB cannot be compared.
- **Test-time training (arXiv 2604.06169, ICLR 2026 and others):** Results on long-context tasks.
- **Automated search:** Recursive, AIRA-Compose. Within what was read, there is little breakdown of which changes worked.

## Research on measurement

- Relative Scaling Laws (arXiv 2510.24626, search): across 255 models, loss and accuracy are tightly linked in a sigmoid shape.
- Scale Efficiently (a note in `language-model`): loss is mostly set by size, but scores depend on the shape.
- Can Small Training Runs Reliably Guide Data Curation? (ICLR 2026, list): the ranking of small models flips with the learning rate.
- AI Progress Should Be Measured by Capability-Per-Resource (NeurIPS 2025, list): argues that progress should be measured by capability for the same resources.

## Findings

- **Many reports show higher scores:** STEM, L³, Mamba-3, Composer, AIRA-Compose, Dispersion Loss, COUNT, MeCo, LARE and others. The "only COUNT" at the time latest-research-survey.md was written was a mistake caused by the narrow search.
- **The trend of replacing computation with memory is the broadest:** its forms vary, from n-gram tables and per-token tables to memory banks and databases.
- **Reducing attention shows up independently:** Composer's search, AIRA-Compose, speedrun record #92.
- **Mind the difference in amount read:** we currently read 12 tokens per parameter. STEM's 350M reads about 285 and Deep Progressive Training's 124 million about 2,500. Close ones are COUNT (10–40), T-Muon (about 20), the speedrun and nanochat, and Dispersion Loss's mid-training also reads little.
- **Prices are also a major lever:** with the conversion of about 3.9% per doubling of compute, the choice of GPU alone could move results more than most of the model-improvement candidates.

## Not verified

- None of the methods was measured at the current scale.
- The original texts of the "List" and "Search" entries.
- The effects of combinations.
