# A comparison of training-stability methods

Methods of improving LLM training stability (arXiv 2410.16682).

- At 830 million parameters, the learning rate was raised until training diverged, and the methods were compared.
- QK-norm and the logit softcap are the most stable of the baseline methods, diverging at a learning rate of 60e-3. σReparam and LayerScale diverge at 40e-3.
- Combining QK-norm and the softcap, and normalizing after QKV, allow the learning rate to go a further 1.5× higher.

What is compared is the point of divergence, not the difference in loss at the same cost. Read: the abstract and the body of the results.
