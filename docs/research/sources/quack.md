# QuacK

Controlling changes to attention logits (arXiv 2511.21377).

- Taking the growth of the Q and K weights as the cause of instability, it lowers the learning rates of Q and K according to the size of their weights, instead of using QK-norm.
- In ordinary attention (MHA), QK-norm is the best overall, with QuacK slightly below it. At a small learning rate (3e-4), the methods hardly differ. With nothing, training was unstable at a learning rate of 3e-3.
- Removing QK-norm's two RMSNorms made training about 10% faster.
- QK-norm is hard to use with Multi Latent Attention, so QuacK is the best there.

The experiments use AdamW-style learning rates; they are not comparisons under a method that evens out the size of updates, as Muon does (our own reading). The abstract, the introduction and the main text of the results were read; the numbers in the figures were not read. The model sizes were not checked.
