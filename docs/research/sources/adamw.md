# AdamW

Loshchilov and Hutter, Decoupled Weight Decay Regularization (arXiv 1711.05101, ICLR 2019).

- A fix that decouples Adam's weight decay from the gradient update. In SGD, L2 regularization and weight decay are the same, but in Adam they are not. With L2, the decay acts more weakly on weights whose past gradients are larger.
- On CIFAR-10 (ResNet, 11.6 million and 25.6 million parameters) and ImageNet32x32, at a learning rate of 0.001, the test error fell by 15% relative to Adam.
- The best values of the learning rate and the weight decay become nearly independent of each other.
- Learning-rate schedules also help Adam, and cosine annealing did better than a fixed rate or step decay. The best weight decay is smaller the more updates there are.
- The experiments are image classification; there is no comparison with other methods on language models. It cites the Transformer (Radford et al.) as an example of use. The authors write that it needs to be checked on other tasks as well.

The main text was read. The appendix was not read.
