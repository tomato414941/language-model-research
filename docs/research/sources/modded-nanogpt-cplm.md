# modded-nanogpt's CPLM (PR #379)

PR #379 of `KellerJordan/modded-nanogpt` (unmerged as of 2026-11-03). On top of record #92, it adds a pointer to earlier tokens in the document. The paper is described as "coming soon" and cannot be read yet.

## How it works

- The next-token distribution is a mixture of the LM's softmax and a pointer to earlier tokens in the same document. `p(y) = p_lm(y)·(1 − α(1 − a_sink)) + α·p_copy(y)`.
- The mixing weight α is the probability the softmax gives to an unused row of the output embedding table (`<copy>`). It goes through the same softcap.
- The pointer is a single head (128 dimensions) on top of the last hidden state, and it looks only at earlier tokens in the same document. The pointer reaches 2,048 tokens in training and 8,192 tokens in validation. There is a learned sink (a_sink), whose share goes back to the LM.
- Q and K are RMS-normalized and multiplied by a learned gain. Without this, the pointer's values diverged for some seeds.
- Parameters: +196,737. Memory is about 0.1GB in training, compute is at most 0.8% of the training FLOPs, and the time per step is +1.3%.

## Results

- Record #92 on the same machine (13 runs): 1,194 steps, 40.575 ± 0.024 seconds, validation loss 3.2765 ± 0.0016. The PR (8 runs): 1,050 steps, 36.009 ± 0.039 seconds, 3.2769 ± 0.0016. The time is −11.25%, and the loss difference is +0.33 millinats (p = 0.66).
- The step count was chosen on this machine, and the author writes that the loss came out about 3 millinats higher on another machine.
- There are no comparisons such as removing the pointer or setting the validation range to 2,048. There are no downstream scores either.
- In record #92's attention, 5 of the 7 layers have windows of 128 to 768 tokens (1 to 6 blocks of 128 tokens), and even the 2 long-window layers reach at most about 1,700 tokens during training and 2,560 tokens in the final validation (computed from the settings in `train_gpt.py`, which count in blocks of 128 tokens). The pointer reaches further than most layers' windows.

The PR's description and comments, and the window settings recorded in record #92's log, were read. The code was neither read nor run.
