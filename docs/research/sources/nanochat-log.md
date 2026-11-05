# Details of nanochat's LOG

dev/LOG.md and scripts/base_train.py in `karpathy/nanochat` (reread on 2026-11-01). The overall table is in `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/nanochat-log.md); only the numbers it lacks are written here. d is the number of layers, and the width is 64× the number of layers.

- **x0 lambdas (01-11):** validation BPB without them, and with them at the best learning rate.

| Depth | Without | With | Difference |
|---|---|---|---|
| d8 | 1.0885 | 1.0782 | −0.0103 |
| d12 | 0.9770 | 0.9693 | −0.0077 |
| d16 | 0.9059 | 0.9002 | −0.0057 |
| d20 | 0.8565 | 0.8526 | −0.0039 |

- **Weight decay (01-10):** the best value is about 0.40 at d8 (width 512), about 0.22 at d12 (width 768), about 0.10 at d16 and about 0.08 at d20. It says the value is inversely proportional to the square of the width. The cautious form was better than the ordinary one, and lowering it linearly from 1.0 to 0 was better than keeping it fixed. It says Polar Express is within the error, and NorMuon a very small improvement.
- **Sweep of the optimizer settings (01-19 to 22):** about 320 runs. Fine tuning that helped at d12 hurt at d20, and the gain was about 0.002 at d12 and about 0.0007 at d20. It warns against tuning too much on a small proxy.
- **Aligning to the starts of documents (01-13):** each row starts with BOS and is packed with documents. With a context of 2,048, about 35% of the tokens are discarded. Since it is always the ends of long documents that are discarded, the data is slightly skewed, but it says this does not seem to affect downstream performance. It gives as the reason that contexts starting partway through a document do not occur at scoring time. The validation loss falls, but the change makes direct comparison with earlier runs impossible, and it says part of the drop is only apparent. Cutting attention between documents makes almost no difference (0.85427 and 0.85407 at d16).
- **Value embeddings (01-17):** every attempt to shrink the tables failed, the gate helps, and every other layer is best.
- **Softcap (01-16, 03-02):** comparing 5 to 30 at 24 layers, 5 is terrible, the rest are about the same, and 20 is the best (validation loss from 0.716 to 0.715). There is no comparison with no softcap. `23 * sigmoid((x + 5) / 7.5)` is slightly worse than `15 * tanh(x / 15)`, and it says it may help only with FP8.
- **Dropping autocast (03-04):** on the grounds that in effect it only converted the weights to bf16, the types were made explicit. There are no numbers on speed.
- **Data (02-17, 03-04):** after five failed attempts to beat FineWeb-Edu, ClimbMix cut the time to reach GPT-2's performance from 2 hours 46 minutes to 2 hours 1 minute.
- **Learning-rate conversion (`base_train.py`):** the learning rate is made proportional to √B, where B is the number of tokens per update (it says this has not been checked for Muon). Weight decay is made proportional to √B and to the inverse of the number of training tokens D. The baseline is d12, with B at 2^19.

Read: the entries for the dates above, and the conversion part of `base_train.py`.
