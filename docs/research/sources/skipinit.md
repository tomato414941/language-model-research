# SkipInit

De, Smith, Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks (arXiv 2002.10444). It is about residual CNNs without normalization, not language models.

- The residual branch's output is multiplied by a single learned value α. Starting it at 1/√depth or below makes deep networks trainable. Because 0 works whatever the depth, it recommends 0 in practice.
- Wide-ResNet accuracy on CIFAR-10 is about the same with 1/√depth and with 0 (16 layers 93.0 and 93.3, 100 layers 94.2 and 94.2, 1,000 layers 94.2 and 94.3). With α=1, networks deeper than 16 layers cannot be trained.

Section 2 and Table 1 were read.
