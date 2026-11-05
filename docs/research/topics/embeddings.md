# Input and output embeddings

The table that reads tokens (`wte`) and the table that predicts the next token (`lm_head`). The baseline and the first bundle share one table, with a learning rate of 0.004, set for the output.

- **Shared:** GPT-2, Gemma, Qwen3 at 4B and below, SmolLM3.
- **Separate:** Llama 3, OLMo 2 and 3, DeepSeek-V3, Qwen3 at 8B and above. [nanochat](../sources/nanochat-log.md), [autoresearch](../sources/autoresearch.md) and modded-nanogpt's optimization track. The last three are code of the same lineage: the input starts at a standard deviation of around 1.0, has RMSNorm applied right after and gets a large learning rate (0.3–0.7). The output starts at 0 or 0.001 and gets a small learning rate (0.002–0.008).

`language-model` also has short notes ([separation](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/untied-embeddings.md), [learning rate](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/embedding-learning-rate.md)).

## What is known

**The closest comparison:** [modded-nanogpt's records 7 and 8](../sources/modded-nanogpt-records.md). 524,288 tokens per update, the same as now; Muon; and a shared learning rate of 0.0036, close to ours. The difference in validation loss at the same number of updates is −2.45% at 125, −0.84% at 750, −0.56% at 1,500 and −0.42% at 3,000. Separation, the input RMSNorm, the learning rates and `lm_head` starting at 0 are combined in one change. The updates needed to reach 3.28 fell by 10%, which converts to about −0.6%.

**Contrary results:**

- The later record 51 went back to sharing, and its updates fell by 2.6% (equivalent to +0.14% when converted). An adjustment of the FP8 coefficients came with it, and the shared learning rate is 0.008. Having value embeddings makes it close to the current bundle, and with value embeddings the benefit of a separate input table may be small (our reading). Later records also go back and forth between sharing and separating partway through training.
- [SmolLM3](../sources/smol-training-playbook.md)'s 1.2 billion and 1.46 billion are about the same except on WinoGrande. It is a comparison with matched learning rates; the input's learning rate was not raised.
- [Rethinking Embedding Coupling](../sources/rethinking-embedding-coupling.md): from 62.3 to 62.7 in the evaluation after pretraining.
- The [old](../sources/using-the-output-embedding.md) [comparisons](../sources/tying-word-vectors.md) on LSTMs favor sharing, but they are about less overfitting on small data, which differs from the current conditions.

Of SLlama (separation is better in a small comparison on 10 million tokens), only the summary was read.

**The input's learning rate:** [Importance of Embedding Layer Learning Rate](../sources/hyperparameter-transfer-embedding.md) says that with AdamW the input's learning rate is the most important difference and that `lm_head`'s initial value barely matters. [Hayou and Liu](../sources/embedding-lr-vocabulary.md) say that √width times the rate is better for a model of 1 billion. With sharing, the input and the output get the same learning rate, so it cannot be raised. In theory it could be raised even with sharing by multiplying `lm_head`'s output by a coefficient, but no result of an actual comparison was read.

**Mechanism:** [Weight Tying Biases](../sources/weight-tying-biases.md) says a shared table is determined by the output's gradients, which makes the first few layers work worse. Increasing the input's gradients did not improve the downstream evaluations.

**Cost:** large models separate the tables because the tables' share becomes small. Now, with a vocabulary of 8,192, the second table is 5.24 million (about 6% of the total). What grows is the memory and updating it; the attention and MLP computation does not change. SmolLM3's disadvantage from comparing at the same parameter count (fewer layers) does not apply to the current conditions, which compare at the same cost.

**Learning-rate values:** [autoresearch](../sources/autoresearch.md) multiplies 0.6 for the input and 0.004 for the output by (width/768)^−0.5. At width 640 that gives 0.66 and 0.0044. It runs with the same ClimbMix, a vocabulary of 8,192 and Muon.

**The start of training:** With the second bundle's shape, a trial of 34 updates at a batch about 1/500 of the real one did not diverge even with an input learning rate of 0.66, and the loss stayed below the shared table's ([the check's record](../experiments/second-bundle-code-check.md)). It is not a comparison of quality.

## Current view

The view as of 2026-11 is that the second bundle separates the tables. The input starts at a standard deviation of 1.0, with RMSNorm right after and a learning rate of 0.66. The output starts at a standard deviation of 0.001, with a learning rate of 0.0044. We think the effect at the same cost is 0–0.6%. The reasons are these.

- A bundle is measured only once, so it follows an example that worked with Muon. autoresearch's set of learning rates is for the separated form.
- Only memory grows, and with the current vocabulary that is small.
- In the current shared table, the input's learning rate is about 1/160 of autoresearch's and may be too low.

Raising the learning rate while keeping the table shared, and separating partway through training, stay candidates. Neither has a result compared on its own with the current shape and Muon.

## Conditions likely to change the view

- When sharing and separation are compared in the current shape. With value embeddings the difference may be small.
- When the vocabulary is enlarged. The second table grows.
- When training becomes unstable. The input's learning rate is large, so suspect it first.
- When the effect is measured item by item.
