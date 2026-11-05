# Delphi

Open Athena's post, Scaling Laws That Extrapolate 300× Past the Fit (2026-05, https://www.openathena.ai/blog/delphi/). Marin's first scaling suite.

- At each compute budget from 3e18 to 3e20, the model size and token count were varied to find the optimum, and those points were used to predict 1e21, 1e22 and 1e23, which were then checked. The architecture is Qwen 3, with context 4,096.
- The difference across three seeds is about 0.1%, about 1/10 of the prediction's 95% confidence interval.
- It also estimates the cost of straying from the optimum. Training 10× longer needs about 6× the compute for the same loss. However, the underlying data goes only up to 2×, and the authors also say this number has low confidence.
- Counting inference costs, many organizations lean toward training longer. It says Llama 3 8B was trained on about 90× the optimal number of tokens and OLMo 3 7B on about 40×.

Most of the main text was read. The values in the figures were not read.
