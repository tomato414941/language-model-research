# The μ-Transfer paper

Lingle, An Empirical Study of μP Learning Rate Transfer (arXiv 2404.05728). A study of whether the learning rate carries over to other scales, with models of up to 1 billion parameters and 190 billion tokens.

- Under the μP setting that makes `lm_head`'s initial values small, RMSNorm with and without weights was compared (4.2.3). In the largest model at the optimal learning rate, the weights lowered quality. Under μP too, the form without weights beat the usual form with weights (4.5).
- With the weights, the learning rate is harder to carry over to other scales.
- It also compares starting the Q weights at zero (Tables 4 and 9). The validation loss at the best learning rate goes from 3.695 to 3.692 at width 128, from 2.953 to 2.949 at width 512 and from 2.511 to 2.510 at width 2,048. The author calls it "a slight yet consistent improvement".
- Table 11 has 96 layers with the residual blocks starting at zero (widths 128 and 512), with best values of 3.456 and 2.753. There is no comparison with a nonzero start at the same 96 layers.

The text of 4.2.3, the part on the model settings, Table 4's baseline and Q rows, Table 9 and Table 11 were read.
