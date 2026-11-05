# Physics of Language Models 4.1

Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers (arXiv 2512.17351). Its main subject is adding short convolution layers.

The appendix's description of the models has a short passage on initialization.

- Turning off `rescale_prenorm_residual` from GPT-2 (which multiplies the initial output weights by 1/√N) improved performance slightly. No numbers are given. HF's Llama and Mamba2 have it off by default, and GLA's authors also turned it off following this.
- Setting `initializer_range` (the initial scale of all the weights) in GLA's implementation to 0.006 made things worse, and it was set back to 0.02.

Only this part of the appendix was read.
