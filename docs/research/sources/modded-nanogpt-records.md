# The history of modded-nanogpt's records

The table of records in the README of `KellerJordan/modded-nanogpt`, and the code and logs of some records (commit `4ea6b93`, read on 2026-11-01). It is a race for the time to reach a FineWeb validation loss of 3.28 on 8 H100s. The starting point is llm.c's reproduction of GPT-2 (45 minutes, 10 billion tokens), in the line of nanoGPT. According to the README, it now gets there in under 40 seconds and under 330 million tokens.

| Record | Time | Change (as the table describes it) |
|---|---|---|
| 1 | 45 minutes | llm.c's baseline |
| 2 | 31.4 minutes | Learning-rate tuning and RoPE |
| 3 | 24.9 minutes | Muon |
| 4 | 22.3 minutes | Improvements to Muon |
| 5 | 15.2 minutes | Vocabulary padding, ReLU², zero initialization of the outputs, QK-norm |
| 6, 7 | 13.1 minutes, 12.0 minutes | Distributing Muon, PyTorch 2.5 |
| 8 | 10.8 minutes | Separate input and output embeddings |
| 9 | 8.2 minutes | Value and embedding shortcuts, momentum warmup, logit softcap |
| 10 | 7.8 minutes | Activations in bf16 |
| 11 | 7.2 minutes | U-shaped shortcuts, doubled learning rate |
| 12 | 5.03 minutes | Dense attention with a context of 1,024 replaced by FlexAttention with a context of 64K |
| 18 | 3.4 minutes | Softcap from 30 to 15 |
| 26 | 2.863 minutes | Reading aligned to the starts of documents (2.896 minutes before) |
| 49 | 2.146 minutes | Partial Key Offset |
| 51 | 2.075 minutes | Embeddings shared again |
| 53 | 1.988 minutes | Embeddings separated at 2/3 of training, multi-token prediction |

Code and logs read:

- **Records 4 to 5 (the diff of `train_gpt2.py`):** from 1,339 seconds, 6,200 updates and 216 milliseconds per update to 910 seconds, 5,100 updates and 179 milliseconds per update. The changes: zero initialization of the attention and MLP outputs, RMSNorm on Q and K, GELU to ReLU² (a code comment says "~1-2% better than GELU"), heads from 12 to 6 (128 dimensions per head), and the vocabulary to 50,304. RoPE and RMSNorm without learned weights were already in record 4. There is no number for zero initialization alone, and the README only says it is "muP-like". Record 5 still shares the embeddings, and its training loss is 16.0 at the first update, 9.5 at the second and 7.6 at the fifth (log).
- **Record 8 (README):** separate embeddings, RMSNorm after the embedding, and zero initialization of `lm_head`. The active parameters stay the same, and 39 million are added. The training loss is 10.8 at the first update, 10.4 at the second and 7.5 at the sixth (log).
- **Records 7 and 8 (logs):** both have 12 layers, width 768, 524,288 tokens per update and Muon, and no Value embeddings yet. The shared table's learning rate is 0.0036. Record 8 uses 0.3 for the input and 0.002 for the output, with RMSNorm after the input table. The updates needed to reach 3.28 went from 5,100 to 4,578. Validation loss at the same number of updates (the differences are our own calculation; both keep the learning rate constant up to 3,000 updates):

| Update | Shared (record 7) | Separate (record 8) | Difference |
|---|---|---|---|
| 125 | 4.952 | 4.831 | −2.45% |
| 250 | 4.336 | 4.210 | −2.89% |
| 500 | 3.920 | 3.872 | −1.24% |
| 750 | 3.773 | 3.741 | −0.84% |
| 1,000 | 3.684 | 3.658 | −0.69% |
| 1,500 | 3.580 | 3.560 | −0.56% |
| 3,000 | 3.444 | 3.430 | −0.42% |

- **Record 51 (README):** drops the separation and the zero `lm_head` and goes back to sharing (keeping the RMSNorm after the input table). The shared table's learning rate is 0.008 and its standard deviation 0.005, and there are Value embeddings. Together with a tuning of the FP8 coefficients, it cut 55 updates, from 127.1 seconds to 123.9 seconds. The author says sharing cut the number of updates substantially.
- **Record 53 (README):** separates the embeddings at 2/3 of training. Together with multi-token prediction and other changes, from 2,035 updates to 1,920. It says a learning-rate multiplier on the separated input table brought no benefit.
- **Later records (the relevant rows of the README):** 2026-06-11 goes back to sharing throughout, and 2026-08-30 separates them in the final stage.
- **`train_gpt_simple.py` of the optimization track:** separate. The input has standard deviation 1.0 followed by RMSNorm, with learning rate 0.7; the output starts from 0, with learning rate 0.004; Muon uses 0.025.
- **Record 9 (logs):** validation loss with the updates fixed at 3,200, adding one change at a time.

| Added | Validation loss | Time per update |
|---|---|---|
| Shortened only | 3.3274 | 142 ms |
| value residual | 3.3151 | 145 ms |
| Learned weights | 3.3065 | 146 ms |
| Embedding shortcut | 3.2972 | 152 ms |
| Momentum warmup | 3.2844 | 152 ms |
| tanh softcap | 3.2791 | 155 ms |

  "Learned weights" are the value residual's weights, and "embedding shortcut" is the per-block `x = lambdas[0] * x + lambdas[1] * x0` (from `[1., 0.]`, with an Adam learning rate of 0.02).
  The tanh softcap is the single line `logits = 30 * torch.tanh(logits / 30)`.
- **Record 18 (README):** lowering the softcap from 30 to 15 cut the updates from 1,490 to 1,390. The author sees it as imposing a structure that pays off at small scale.
- **The optimization track's README:** for simplicity, it removes Value embeddings and the shortcut weights as unusual components. The logits in `train_gpt_simple.py` are `15 * logits * (logits.square() + 15**2).rsqrt()`, so a softcap remains in a form other than tanh.
- **Later records (the relevant parts of the README):** switching to `23 * sigmoid((logits + 5) / 7.5)` cuts 40 updates (in the period that uses an FP8 `lm_head`). Later still, fusing the softcap and the loss into one Triton kernel saves 0.9 seconds.
- **Record 10 (logs):** `model.cuda().bfloat16()` puts the weights outside the linear layers and the activations in bf16; the linear layers' weights stay in fp32 and are cast at compute time with `F.linear(x, self.weight.to(x.dtype))`; autocast is dropped. An update went from 154.76 milliseconds to 144.67 milliseconds (−6.5%). The updates needed to reach 3.28 went from 3,200 (3.2791) to 3,242 (3.2781), and the total from 493.7 seconds to 467.6 seconds (−5.3%).
- **Record 26 (README):** aligns the start of each training row with the start of a document, on the grounds that learning from contexts that start partway through is wasteful. It cut the updates from 1,770 to 1,750, together with a change to the learning-rate decay and someone else's speedup.
- **Record 49 (README):** Partial Key Offset shifts the key in the dimensions without RoPE back by one token. It is used only in the long-window layers. With 5 changes combined, from 131.2 seconds to 128.9 seconds.
- **Record 12 (README):** the details of the time saving were not read. It says the noise between runs grew.

On 2026-11-04, the part of the README after the table of records and the descriptions of the following PRs were read.

- **Record 28 (PR #117, sparse attention gate):** multiplies the attention output by a gate computed from the first 12 dimensions of the residual, so that attention can choose to do nothing depending on the context. The updates needed to reach 3.28 went from 1,750 to 1,695, and the author estimates the gate's share at about 50. The time per update rises slightly. The author's blog says that with the gate, there is less need to use BOS as a place to send attention (attention sink).
- **Record 34 (PR #130, smear):** adds the previous token's embedding, through a gate, to the current token's. The updates fell by 15, but the author says it is worth about 5 at matched loss.
- **Record 40 (PR #140, backout):** before the prediction, subtracts the residual's contribution up to layer 8, scaled by a learned coefficient. It came with other tuning, and its effect was not separated.
- **Record 49 (PR #169, Partial Key Offset):** shifts the key in the unrotated dimensions (32–64 and 96–128 of the 128) back by one token, so that a single layer can do induction. It is better than applying it to all dimensions, and worse when applied to only some heads. Applying it to every layer gives the lowest loss, but for speed it is used only in the long-window layers. The 5 changes went from 131.2 seconds to 128.8 seconds, and most of that is attributed to the drop in updates (50) that this brings.
- **A record of the medium track (PR #139, backout):** adds the output of layer 12, scaled by a learned coefficient, to the output of the final layer before the prediction. The updates needed to reach the target loss went from 5,590 to 5,550, and the mean over 22 runs is 2.9194 (standard deviation 0.00069).
- **Record 62 (PR #201, bigram hash embedding):** hashes the pair of the previous and current tokens, looks it up in a table with 5× as many rows as the vocabulary, and adds the result to each layer's input. The table starts from 0 and the per-layer coefficients from 0.1. The table alone saves about 5.1 seconds (about 4.9%), and the updates went from 1,765 to 1,600 (including other changes). It says adding the result as is worked better than a gate, 3 tokens or several hashes.
- **Record 81 (PR #259, MUDD):** only the last 2 layers mix the hidden states of earlier layers, with weights set per position by a small MLP, and add the result to V and to the residual (starting from 0). At the same 1,430 updates, the loss went from 3.2867 to 3.2783 (each the mean of 10 runs). The time per update rises by about 1.8%, and the total is about 1.5 seconds shorter. The author is one of the authors of MUDDFormer.

The posts on X were not read. Record 44's [Profiling 101](https://blog.underfit.ai/profiling-101-nanogpt) returned 403 and could not be read.
