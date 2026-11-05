# How training rows are cut

Where in the data each training row (a sequence of tokens as long as the context) is taken from. The baseline and the first bundle keep nanoGPT's way and take rows from random positions in `train.bin`, so most rows start in the middle of a document. The scored questions start from BOS.

The second bundle aligned the start of each row with the start of a document (BOS).

## What is known

- **Measured result:** [The second bundle](../experiments/second-bundle.md) trained only on rows that start from BOS, and on rows that start in the middle of a document its loss over the first 64 tokens became +47% (−1.9% over the rest of the row). On rows that start at the start of a document it was −4.2% overall. `scoring.py`'s validation BPB reads rows that start midway, so it became +2.2%, while the correct-answer BPB, which starts from BOS, became −4.4%. This was not measured separately from the other items.

- **[modded-nanogpt's record 26](../sources/modded-nanogpt-records.md):** updates went from 1,770 to 1,750 (−1.1%), which converts to about −0.06%. Other changes came with it.
- **[nanochat](../sources/nanochat-log.md) (01-13):** made aligning rows and then packing documents the standard form. It gives no figure for the effect and says part of the drop in validation loss is only apparent.
- **autoresearch:** reads the data the same way as nanochat.
- **Direct comparison:** no paper has been found that compares by loss whether rows start mid-document or at the start of a document.

**The current data (our own calculation):** 2.68 million documents in 2 billion tokens. Their length averages 745 with a median of 617, and 11.3% of documents are longer than a row (1,025 tokens). Reading from the start, the 23.1% of tokens that do not fit in the first row are not read. Documents are chosen with equal probability, so short documents are read more than before, and the mean number of documents starting in a row rises from 1.37 to 2.4 ([the check's record](../experiments/second-bundle-code-check.md)). Finding the BOS positions takes 9 seconds on the processing machine; made once and kept in a file, it does not count toward the training cost.

**Yardstick:** `scoring.py`'s validation BPB reads rows cut from fixed positions, so it is unfavorable to a model trained with this way of cutting (scoring.md). `train.py`'s validation loss also cuts from random positions.

**Other forms of a similar idea:**

- Packing documents (nanochat's best-fit): documents are less often cut within a row, but the share of discarded documents and the code both grow.
- [Computing attention only within documents](../sources/sequence-composition.md): about −3.9% at 1.3 billion, and BPB from 0.85427 to 0.85407 for nanochat's 16 layers. The evidence is split, and it also adds computation.
- [Packing similar documents into the same row](../sources/sequence-composition.md): −3.2 to −4.1% at 1.3 billion. The attention computation is unchanged, but it needs preprocessing. There is only one study, with no results at the current scale.

## Current view

The view as of 2026-11 is to keep it. Reading starts from a random BOS and continues into the next document until the row ends. No packing is done. Before measuring, we put the effect at 0–0.2%, but the side effects were larger than expected. The effect itself was not measured separately. The reasons to keep it are these.

- In actual use the context starts at the start of a document, and the correct-answer BPB improved in that form.
- It takes little effort, and only a little of the 400B of data is used, so there is enough even with parts left unread.
- autoresearch's set of learning rates runs with rows read from the start of documents.

It is weak when the context starts in the middle of a document, and uses that read a long text cut to the context length may be affected. However, every measured task starts from BOS, so mixing rows that start mid-document into training does not improve the measured scores. The mixed form was moved to "Candidates likely to have small effects" (candidates.md).

If uses that read from the middle of a document are weighed heavily, first put a BOS at the front when reading. Models concentrate attention on the first token ([StreamingLLM](https://arxiv.org/abs/2309.17453)'s attention sink; only the abstract was read), so much of the +47% may come from the missing BOS at the front. Measuring again with the saved weights of the second bundle, with a BOS added at the front of rows cut at fixed positions, would check this on the processing machine. This has not been measured yet.

## Conditions likely to change the view

- When uses in which the context starts mid-document are weighed heavily.
- When the effect is measured item by item.
- When training grows enough to use up the data. The unread 23.1% becomes a waste.
