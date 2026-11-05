# modded-nanogpt's lambdas

A post by snimu that looks at modded-nanogpt's learned lambdas (snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html). The values come from one training run of the medium track, with no comparison of having them against not having them.

- In `x = x_lambda * x + x0_lambda * x0`, x0_lambda is large in all layers but three, and larger than x_lambda in many layers. In layers 8, 14 and 15 it crosses 0 during training, but the validation loss does not jump.
- The first version of the post had the labels of the two values swapped; they were corrected the next day.
- The role of x0 (serving as a reference for comparison, and so on) is the author's hypothesis.

Read: a summary extracted with the fetching tool. The post's figures were not seen.
