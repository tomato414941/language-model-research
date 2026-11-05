# Stochastic activations

Lomeli et al., Stochastic activations (arXiv 2509.22358).

- During training, the MLP activation is chosen at random between SiLU and ReLU. Fine-tuning with ReLU at the end works better than training with ReLU alone, and lets inference be sparse.
- The goal is faster inference (about 1.65× in the MLP). The quality does not go beyond SiLU's.

The abstract and part of the results text were read.
