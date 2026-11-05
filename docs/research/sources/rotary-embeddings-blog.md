# EleutherAI's post on RoPE

Biderman et al., [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) (2021-04-20). A post without peer review, and a comparison by the people who popularized RoPE.

Results of comparing it with learned position embeddings, in the GPT shape.

| Model | Data | Tokens | Per parameter | Learned table | RoPE | Difference |
|---|---|---|---|---|---|---|
| 125 million (150 million in the table heading) | OpenWebText2 | About 30 billion | About 240 | 2.809 | 2.759 | −1.8% |
| 1.4 billion (1.5 billion in the table heading) | The Pile | About 8 billion | About 6 | 2.240 | 2.173 | −3.0% |

- The numbers are validation loss. At 1.4 billion, it says convergence is also about 30% faster.
- It says validation loss also fell substantially for a small character-level model (enwiki8) with 8 layers and width 512, but there is only a figure, with no numbers.
- Cost: with some care, about 1–3% more in total. Written naively, it takes 4–5× the time of an additive position embedding. It has to be applied in every layer.

The number of seeds is not stated. Read: the experiment and cost sections through a summarizing tool. The figures were not seen.
