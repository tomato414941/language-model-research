# Methods that look up tables (Engram, COUNT, STEM, L³)

Methods that look up a table from the token or the few preceding tokens and add to or replace the model's computation. The lookup takes almost no computation; what grows is mainly memory. COUNT was put into the third bundle on 2026-11-03, and the learned two-token table on 2026-11-04 (see "Its form in the third bundle" below). On 2026-11-04 the bundle was about 1% better than the second bundle on validation BPB read from the start of documents ([the third bundle](../experiments/third-bundle.md)), but the tables' own effect was not separated. COUNT makes each update about 18% slower.

## Tables looked up by the token alone

- [STEM](../sources/stem.md) replaces the input part of the MLP with a table; at 350 million the mean score goes from 49.72 to 50.90, and FLOPs also fall a little. It assumes a gated MLP and does not fit the current ReLU² form as is.
- [L³](../sources/l3.md) adds tables between layers; at the 800-million class with the same FLOPs, the score goes from 48.28 to 49.10 with one layer, 49.45 with two and 50.25 with three layers with wider tables. Speed is about 87%. The tables hold 710,000 embeddings per layer; counted with the current width of 640 and a value width of 512, that is about 800 million parameters per layer (about 10× the whole current model).
- STEM's amount read is orders of magnitude larger than ours. L³'s 800-million class has about 12 tokens per parameter, about the same as now. The current value embeddings (every other layer) are the same kind of table, and with a vocabulary of 8,192 the tables are small.

## Tables that are not learned (COUNT)

[COUNT](../sources/count-ngram.md) keeps a table of n-gram counts from a corpus in main memory, weights the output-table rows of likely next tokens by frequency and adds them to the input. No learned parameters are added.

- At 60M–1B and 10–40 tokens per parameter, the gain in validation loss is equivalent to 46–89% more compute and the gain in task loss to 6–56% more, and the scores on 41 multiple-choice tasks also rise at most budgets ([the `language-model` note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/count-ngram.md)). Converting at about 3.9% per doubling of compute, that is about 2–3.6% in validation loss and about 0.3–2.5% in task loss.
- The current bundle is 54 million at 12 tokens per parameter, with compute of about 2×10¹⁷. That is about the same as the paper's smallest point (60 million, D/N=10), where the difference is largest. Read off the figure, the accuracy over the 41 tasks goes 29.7→31.3 and the correct-answer BPB 2.05→1.91.
- There is only one study, and its baseline uses AdamW. The current bundle has already gained from Muon and others, so the effect may be smaller where they overlap. The tables are built from all of Dolma 3 (about 6 trillion tokens), and whether tables counted on less data work as well has not been tested.

## Learned tables

- **speedrun:** In [record #92](../sources/modded-nanogpt-anvil2.md), raising the number of rows from 377,280 to 84.6 million takes the validation loss from about 3.315 to 3.279 (about 1.1%). The time is almost the same, and most of the gain remains with a quarter of the rows.
- **Automated search:** [Recursive](../sources/recursive-first-steps.md)'s best solution spends about 2.8 billion parameters on two-token and three-token tables. The tables' own effect has not been separated.
- **Reducing memory:** [Low-Bit N-Gram](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/low-bit-ngram.md) says that going to 4 bits cuts memory by about 75% while over 99% of the effect remains.
- **Older results at small scale:** In nanochat's LOG (2026-01-27 and 28), the full form improved things only slightly, neither the gate nor three tokens helped, and putting it only in the first half of the layers made things worse. The best was a two-token table added to every layer; it won at 12 layers, but at 25 layers its effect vanished when compared by time, and it was removed. This form's table has 5× the vocabulary in rows, starts at 0 and is added to each layer's input multiplied by a per-layer weight (initial value 0.1). At 12 layers it beat the baseline whether compared by number of updates, compute or time, but no figure for the difference is given. With the current vocabulary that is about 41,000 rows and about 26 million parameters (our own calculation). In Marin, about 0.2% ([the bigram hash note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/bigram-hash-embedding.md)). The table is much smaller than record #92's.
- **Comparison with COUNT:** COUNT's authors say that when GPU memory is limited, it is better spent on the model itself than on a learned table.

## Large scale

- In [the Engram paper](../sources/engram.md), validation loss at the same compute is 0.7% and 1.5% lower than MoE's. That is about the same as raising compute about 1.14× and about 1.3× (our conversion). Even the smallest experiment uses about 730× the baseline's compute, and all the loss experiments are on top of MoE.
- In [Tokenizer-Agnostic Engram](../sources/tokenizer-agnostic-engram.md), the benchmark average rose, but wikitext BPB worsened from 0.861 to 0.919 (one run each). The yardstick here is BPB, so this needs care.
- Even when it is put in the last of 12 layers, 64% of the effect remains, so we think "it lightens the early layers' work" cannot fully explain it. In the experiment that cuts off the table, MATH also drops to 36%, so "the knowledge is in the table" is weakly supported too.
- DeepSeek-V4 (arXiv 2606.19348) did not use it and listed it as a direction to explore. DeepSeek-V4.1-Flash devotes 196 billion parameters to Engram (from Low-Bit N-Gram's abstract; the original report was not read). [Marin's post](../sources/marin-moe.md) passes on it.

Unlike RETRO (retrieval.md), it does not hold the text itself and can look up only short phrases. It needs no DB or search, and it cannot copy long passages verbatim.

## Applied to the current conditions

Our own calculations and reading; not measured.

- **Number of rows:** with a vocabulary of 8,192, there are at most about 67 million two-token pairs, fewer than the 2.5 billion for a vocabulary of 50,304. Fewer rows may suffice.
- **Table size ([counted](../experiments/count-table-check.md)):** from 2 billion tokens of ClimbMix, with the top 8 candidates, about 5.4GB at a minimum count of 4 and about 2.3GB at a minimum of 8. Smaller than the paper's 17GB. The third bundle's table, built from 2 billion tokens of FineWeb, is 5.3GB.
- **Memory:** a learned table lives on the GPU. The current bundle uses about 51GB of the H100's 80GB. COUNT's table can live in main memory, but at this size it also fits in the GPU's free space.
- **Input magnitude:** in the second bundle's weights, the input table's values are about 17.9 and the output table's about 0.12 (mean per-row RMS), a difference of about 150×. Normalizing after adding, as the paper does, would make COUNT's part vanish, so the two need to be scaled separately before being added.
- **Effect of duplication:** [speedrun's PR #367](../sources/modded-nanogpt-retrieval.md) used exact-match search over the training data as a hint and cut 40.5 seconds to 6.7 seconds on the same machine. The maintainers see this as showing the gap between validation loss and generalization. The table's improvement in validation loss may include a share from text that overlaps between training and validation, so check it on the scores too.
- **The table's text and the training text:** the view as of 2026-11-03 is to build the table from a part that training does not read.
  - When a table with a minimum count of 4 is built from the 2 billion tokens of training data, the share of found rows whose next token is in the top 8, at history 5, is 83% on the counted text and 57% on the validation text ([counted](../experiments/count-table-check.md)). The model may trust the table too much during training. The effect on scores was not measured.
  - The paper counts on the same corpus, but the table's source text is 150–10,000× the amount read, and its rows are histories that appeared hundreds of times or more, so this gap should be small. Here the text is about 3× the amount read, and the minimum count is also low.
  - Other fields have the same problem. [CatBoost](../sources/catboost.md) calls it target leakage and says that leaving out only the example itself does not prevent it. The drawback of splitting the data is that the data shrinks, but about 99.5% of ClimbMix is not read, so that drawback is small. [RETRO](../sources/retro.md) excludes neighbors from the same document, and [the speedrun PR](../sources/modded-nanogpt-retrieval.md) looks up only from previous steps or counts on shards it does not read.
  - Revisit this once it is measured side by side with a table built from the same text.
- **What to build it from:** the view as of 2026-11-03 is [FineWeb](../sources/fineweb.md)'s 10BT sample.
  - **Why not ClimbMix:** its mixture was chosen by the scores on the validation questions of 3 of the 6 tasks (PIQA, ARC-Easy, HellaSwag). A table can pass on the continuations of that text directly. Its license is also non-commercial.
  - **Why FineWeb:** it is general web text made by a single pipeline, with no optimization of the mixture. ODC-By. The experiments that chose the pipeline used the scores on 8 tasks including HellaSwag, PIQA, ARC and MMLU, so it is not entirely neutral.
  - **Not chosen:** FineWeb-Edu, Nemotron-CC and Cosmopedia are sources of ClimbMix. DCLM's quality classifier was chosen by task scores and is also used in Nemotron-CC's quality judgments. [Common Pile](../sources/common-pile.md) is heavy on papers, code and patents, its weights were also set from small models' scores, and it is weak on HellaSwag and PIQA. Dolma 3, which the paper used, is 76% CC, and a secondary account says its mixture was optimized from task scores with small models, but this was not checked in primary sources. Domain-restricted data is not used, since choosing domains close to the 6 tasks would mean fitting to the evaluation. In [MassiveDS](../sources/massiveds.md) too, a datastore that mixes domains is as good as or better than per-domain ones.
  - **What is lost instead:** with tables of the same size, the hit rate on the ClimbMix text the model reads is 20–40% lower for the preceding 3–5 tokens. It was about the same with tables from 2 billion tokens. Even on v6's English, which is neither, the ClimbMix table hits more often ([measured](../experiments/count-table-check.md)). This departs from the form, as in the paper and [kNN-LM](../sources/knn-lm.md), that counts a lot of text of the same kind as the training text. The effect on scores was not measured.
  - **What is gained:** the table becomes data separate from both the training data and the validation data, so its hit rates during training and in actual use match. In [Infini-gram](../sources/infini-gram.md), the improvement is smaller for models trained on the same data as the reference. The table need not be rebuilt when the training data changes.
  - Revisit this once it is measured side by side with the ClimbMix table.
- **How much to count:** for tables of the same size, each doubling of the text raises the hit rate by 1–3 points (measured up to 2 billion). With kNN-LM, MassiveDS and RETRO's DB too, results keep improving as they grow. The training cost does not change, so it can grow as far as the processing machine can count. As of 2026-11-03, start with 2 billion tokens, the same as the training data. With the current method the disk runs out at around 4 billion tokens, and going further requires changing the method. Increase it once COUNT is known to work at the current scale.
- **Leakage of evaluation questions:** the continuation can be looked up directly from a five-token history, so if evaluation questions are in the table, they may show up in the scores more readily than when they remain in the weights. Since accuracy is what is looked at, documents containing evaluation questions (0.114% in the training data) are not counted. The text the table is built from is searched in the same way.
- **Cost:** building and transferring the table are not part of the current comparison, which matches training time. They are recorded separately as costs. Transferring 5.3GB takes about 4 minutes at the last transfer's speed, about $0.16. The transfer is needed for every training run. The table can be built on the processing machine; for 2 billion tokens of FineWeb, preparing the text took about 34 minutes and counting about 1.6 hours ([breakdown](../experiments/count-table-check.md)). Once built, it can be reused in any training run.

## Its form in the third bundle

The settings are in `configs/bundle3.py`; the table is built by `count_data.py` and `count_table.py`.

- **How it came in:** a free check that mixes the table into a saved model's output was proposed, but the item was included without the check (2026-11-03). Its effect is seen together with the rest in the bundle's measurement.
- **Table:** 2 billion tokens from the start of FineWeb's 10BT sample (revision `9bb295d`), excluding documents that contain evaluation questions. For histories of the preceding 0–5 tokens that appeared 4 or more times, the top 8 following tokens. The score is `C(h,w)/C(h)` rounded to 1 byte. Candidates are not filled in from shorter histories. A history may start with BOS but does not contain BOS in the middle. 5.3GB. On ClimbMix's validation text, the share of positions where the next token is among the candidates is 38% at history 3 and 9% at history 5 ([measured](../experiments/count-table-check.md)).
- **Paragraph breaks:** FineWeb separates paragraphs with a single newline, and no token contains two newlines. In ClimbMix's validation text, 0.95% of tokens are such tokens, and histories containing them (4.7% of positions at history 5) are not found in the table. In FineWeb, paragraph breaks cannot be told apart from line breaks, so the table is used without matching them.
- **Lookup:** during training, on the GPU, the keys are binary-searched for each length. The candidates' rows for the 6 lengths are weighted by score, summed and divided by 6. Counting the lengths that are not found is the same as in the paper. What is added is rows of the output table, and the gradient flows into the output table.
- **How it is added to the input:** the output table's rows are about 1/150 the magnitude of the input table's, so each is scaled separately, then multiplied by a learned weight (initial value 0.1, the same learning rate as the x0 weights) and added. The paper applies LayerNorm after adding.
- There are many differences from the paper (the table's source, the amount, paragraph breaks, how it is added, the optimizer), so if it does not work, review these first.

The learned two-token table (`bigram_table`) was also put in on 2026-11-04. Its role partly overlaps with COUNT's, but the bundle is measured as a whole, so it was added as something likely to help.

- **Form:** the same as speedrun's record 62 and nanochat. The pair of the previous and the current token is hashed to look up a table of 5× the vocabulary (40,960 rows, about 26 million parameters), which is multiplied by a per-layer weight (initial value 0.1) and added to each layer's input. The table starts at 0. The first position of a row has no previous token, so it uses the table's last row. Pairs that span a BOS are not separated either, as in record 62.
- **Learning rate:** 0.1 for the table, the same as value embeddings (in record 62 the table and value embeddings share a learning rate); the per-layer weights use the same rate as the x0 weights.
- **Reason:** in record 62 this alone cut the time by about 4.9%, about 0.3% when converted to the current scale. At nanochat's 12 layers it won whether compared by number of updates, compute or time.
- **Weak points:** at nanochat's 25 layers the effect vanished when compared by time. The part that overlaps with COUNT cannot be separated from the bundle's result.

## Conditions likely to change the view

- When it is measured at the current scale and the effect is smaller than expected. Especially if accuracy does not move.
- When MoE is added. The Engram paper's results are on top of MoE.
- When the task is changed to one that also counts the cost of inference. The table's memory becomes a cost.
