# Checking the code of Muon and the bundle

On 2026-11-01, `muon.py`, the Value embeddings in `model.py`, `configs/muon.py` and `configs/bundle.py` were checked to run, on the processing machine without a GPU (CPU). **Their effect was not measured.** These checks ran a small model, so the loss values do not predict real training.

| Check | Result |
|---|---|
| Tests | All 52 passed |
| Baseline (AdamW, 2 layers, width 64, 6 updates) | Loss exactly the same before and after the change |
| The baseline's saved weights | Loaded by the changed code, with normal output |
| Muon (the same small model, 20 updates) | Training loss from 9.02 to 7.36. Resuming from partway also worked |
| Bundle (the same small model, 20 updates) | Training loss from 9.03 to 7.51. Resuming from partway also worked |
| Removing each item | No error with any of Value embeddings, Polar Express, NorMuon or cautious weight decay removed |

The parameters at the real size are as follows.

| | Used in the computation | Total | Computation per token |
|---|---|---|---|
| Baseline | 91.25 million | 92 million | 661 million |
| Bundle | 54.41 million | 81.3 million | 405 million |

Not checked:

- Speed on a GPU. For Muon's added computation there is only the original post's calculation, an upper bound of 0.7%, and no measurement.
- Training stability at the real size. The matrix learning rate of 0.04 and the table learning rate of 0.1 are untuned.
- A measurement at the real size on the CPU was stopped partway, since it says nothing about the GPU and takes long.
