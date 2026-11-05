# A post comparing RMSNorm with and without weights

Lucas Lingle, [Parametric or Nonparametric RMSNorm?](https://lucaslingle.substack.com/p/parametric-or-nonparametric-rmsnorm) (2025-04-30).

- LLaMA's shape, 28 layers, width 3,584. AdamW (β 0.9, 0.95, ε 1e-8, weight decay 0.1). 1,048,576 tokens per update, context 256, 100,000 updates. The learning rate was searched for each.
- The lowest losses are 2.240 with weights and 2.253 without.
- It does not compare with LayerNorm. The number of seeds is not stated, and neither is the data.
- The author admits the result is the opposite of their own earlier paper (mu-transfer.md), and says that having weights is the consensus choice.

The main text was read through a summarizing tool.
