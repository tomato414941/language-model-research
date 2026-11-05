# Checking the code of the second bundle

On 2026-11-01, the code with RoPE, RMSNorm, QK-norm, ReLU² and zero initialization of the output added was checked to run, on the processing machine without a GPU (CPU). **Their effect was not measured.** The same checks were run each time an item was added. The numbers below are from when all 5 items were in.

## Checks on a small model

2 layers, width 64, 2 heads, context 64, 8 rows × 2 passes per update, float32, no compile.

| Check | Result |
|---|---|
| Tests | All 111 passed |
| Baseline, Muon, the first bundle (6 updates) | Loss exactly the same before and after the change |
| Removing the items added in the second bundle (6 updates) | Loss exactly the same before and after the change |
| The second bundle (everything in, 20 updates) | Validation loss from 9.02 to 7.56 |
| Resuming from partway, scoring | Worked. Validation BPB 2.72 |
| The saved weights of the baseline and the first bundle | Load, with the added items off |

## The start at the real size

On the second bundle with 10 layers and width 640 (RoPE, RMSNorm, QK-norm, ReLU², Value embeddings, shared embeddings), the output weights were initialized to the current small values and to 0, and the two were compared with the same seed. One update was 4 rows × context 256 (about 1/500 of the real one), float32, no compile. Only Muon's orthogonalization was changed to float32.

| Update | Small values | 0 |
|---|---|---|
| Validation loss before training | 9.14 | 12.82 |
| Training loss, 1st | 9.14 | 12.80 |
| 2nd | 8.42 | 8.16 |
| 3rd | 7.93 | 7.94 |
| Mean of 6th–10th | 8.13 | 8.11 |
| Mean of 11th–20th | 7.71 | 7.72 |
| Mean of 21st–30th | 7.50 | 7.55 |

The batch is small and the updates few, so this only checks the start.

## With separate embeddings added

On the same day, separating the input and output embeddings was added and the same checks were run. All 126 tests passed. With it removed, the loss was exactly the same as the earlier code's. The small model with everything in (20 updates) went from 9.01 to 7.41 in validation loss, and resuming from partway and scoring also work. At the real size, the parameters used in the computation stayed at 54.4 million, and the total went from 80.61 million to 85.85 million.

The start at the real size (same conditions as above, with the output weights from 0):

| Update | Shared, learning rate 0.004 | Separate, input 0.66, output 0.0044 |
|---|---|---|
| Validation loss before training | 12.80 | 9.01 |
| Training loss, 1st | 12.79 | 9.01 |
| 2nd | 8.21 | 8.91 |
| 3rd | 8.14 | 8.43 |
| Mean of 6th–10th | 8.06 | 7.58 |
| Mean of 11th–20th | 7.73 | 7.35 |
| Mean of 21st–30th | 7.48 | 7.33 |
| Mean of 31st–34th | 7.41 | 7.10 |

Even with an input learning rate of 0.66, training did not diverge in this range. The batch is small and the updates few, so this is not a comparison of quality.

## With the x0 and resid weights added

On the same day, the x0 and resid weights were added and the same checks were run. All 136 tests passed. With them removed, the loss was exactly the same as the earlier code's. The small model with everything in (20 updates) went from 9.01 to 7.45 in validation loss, and resuming from partway and scoring also work.

The start at the real size (same conditions as above, with separate embeddings; same seed):

| Update | Without the weights | With the weights |
|---|---|---|
| Training loss, 1st | 9.01 | 9.01 |
| 3rd | 8.43 | 8.31 |
| 4th | 8.04 | 9.36 |
| 5th | 8.04 | 9.10 |
| Mean of 6th–10th | 7.58 | 8.17 |
| Mean of 11th–20th | 7.35 | 7.57 |
| Mean of 21st–30th | 7.33 | 7.38 |
| Mean of 31st–34th | 7.10 | 7.16 |

With the weights, the loss jumped at the 4th update and nearly caught up by around the 30th. The x0 learning rate of 0.5 is larger than its initial value of 0.1, so this may be because the weights move a lot in the first few updates, but this has not been checked. The batch is small, so whether the same happens at the real size is unknown. This is not a comparison of quality.

## With the logit softcap added

On the same day, `15 * tanh(logits / 15)` was added and the same checks were run. All 143 tests passed. With it removed, the loss was exactly the same as the earlier code's. The small model with everything in (20 updates) went from 9.01 to 7.39 in validation loss, and resuming from partway and scoring also work.

The start at the real size (same conditions as above, with the x0 and resid weights; same seed):

| Update | Without softcap | With softcap |
|---|---|---|
| Training loss, 1st | 9.01 | 9.01 |
| 3rd | 8.31 | 9.00 |
| 4th | 9.36 | 8.44 |
| 5th | 9.10 | 8.36 |
| Mean of 6th–10th | 8.17 | 8.12 |
| Mean of 11th–20th | 7.57 | 7.55 |
| Mean of 21st–30th | 7.38 | 7.33 |
| Mean of 31st–33rd | 7.05 | 7.01 |

With softcap, the jump at the 4th update seen with the x0 and resid weights did not appear. Why has not been checked. The batch is small and the updates few, so this is not a comparison of quality.

## With the values flowing through the layers in bf16

On the same day, `train.py`'s `stream_bfloat16` was added. It works only on the GPU, so it cannot be checked with `train.py` on the CPU. All 149 tests passed, and with the CPU's autocast it was confirmed that the values flowing through the layers all become bf16, and stay fp32 when it is removed. The training losses and scoring values of what came before do not change.

With the same weights and the same batch, the flowing values were compared in fp32 and in bf16 (the CPU's autocast, everything in, with only the output weights starting from small values. Its list differs from that of the GPU's autocast).

| Model | Loss | Largest difference in logits | Cosine similarity of gradients |
|---|---|---|---|
| 3 layers, width 32, vocabulary 32 | | 0.0001 | Matrices 0.9999 or more, x0 weights 0.82, resid weights 0.56 |
| 4 layers, width 640, vocabulary 8,192, 4 rows × 128 | 9.01052 and 9.01052 | 0.001 | Matrices 0.9999 or more, x0 weights 0.98, resid weights 0.9999 |

The gradients of the x0 and resid weights are very small (because RMSNorm cancels out the magnitude of the stream), and in the small model they are buried in rounding. At width 640 they do not drift much.

## With reading from the heads of documents

On the same day, `train.py`'s `rows_from_document_heads` was added and the same checks were run. All 153 tests passed. With it removed, the loss was exactly the same as the earlier code's. The small model with everything in (20 updates) went from 9.01 to 7.68 in validation loss, and resuming from partway and scoring also work. The file of the positions of document heads (`train.heads.npy`) is 21MB and is created next to the training data.

2,000 rows were read from the real data in the same way as `train.py` and compared.

| How rows are read | Rows starting with BOS | Mean number of documents starting in a row |
|---|---|---|
| From random positions | 4 rows | 1.37 |
| From the heads of documents | 2,000 rows | 2.4 |

Documents are chosen with equal probability, so short documents are read more than before.

The start at the real size (same conditions as above. The data read differs, so the losses in the two columns cannot be compared directly):

| Update | From random positions | From the heads of documents |
|---|---|---|
| Mean of 1st–5th | 8.78 | 8.67 |
| Mean of 6th–10th | 8.12 | 8.12 |
| Mean of 11th–20th | 7.55 | 7.50 |
| Mean of 21st–30th | 7.33 | 7.30 |

## Not checked

- Whether it runs on a GPU. Speed was not measured either. The cost of QK-norm is unknown.
- Whether it trains stably to the end at the real size.

## Lessons about how to check

- When comparing losses before and after, use one thread (`OMP_NUM_THREADS=1`). With 2 threads, even the same code changes at the 4th decimal place.
- With Muon at the real size, bf16 matrix multiplications are slow on this CPU, and the first update had not finished after 280 s. Changed to float32, it takes about 7 s per update (4 threads).
- Logs written to a file are buffered, so training that will be stopped partway is run with `PYTHONUNBUFFERED=1`. Stopping it loses what was buffered.
- When scoring on the CPU, add `--device cpu --precision float32 --batch-tokens 4096`.
