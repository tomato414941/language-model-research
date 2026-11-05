# modded nanoGPT WR

djdumpling's post (2026-05-27, `djdumpling.github.io/2026/05/27/modded-nanoGPT-WR.html`). A report on cutting the record with gated XSA; its list of what was tried and did not work is useful. 8 H200s or H100s, vocabulary 50,304.

What worked: with learned gated XSA, the validation loss stays below 3.28 even with the updates cut from 1,440 to 1,410 (3.27865, 10 runs).

What did not work (numbers are mostly not given).

| Tried | Result |
|---|---|
| AdEMAMix, AdaMuon, schedule-free Adam, Dion | Worse. Attributed to the slow state needing 1,500 or more steps to ramp up |
| Power-law learning-rate decay, schedule-free averaging, LAWA, a separate learning-rate floor | Unchanged or worse |
| Growing the context from short to long | Helps per step, but loses to the time of 3 `torch.compile` recompilations. The variability also grows |
| Learning the attention temperature, SSMax | Settles near 1, no difference |
| An explicit attention sink | No difference. BOS plays that role |
| GQA, register tokens and others | Cancelled out by the cost of compile and kernel shapes |
| Dynamic Tanh | Diverges at every initial value |
| xIELU | About 10% slower even after 3 rounds of kernel improvement |
| Narrowing the MLP by 2× | About 15% faster, but the loss is 0.04 worse |
| mHC | Worse |

The post's main text was read through a summary; the code was not read.
