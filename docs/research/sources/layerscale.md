# LayerScale (CaiT)

Touvron et al., Going deeper with Image Transformers (arXiv 2103.17239). Image classification, not language models.

The output of each residual branch is multiplied by values learned per channel (LayerScale). They start close to 0 but not at 0: 0.1 up to 18 layers, 1e-5 at 24 layers, and 1e-6 for deeper models.

Accuracy of DeiT-S on ImageNet (higher is better). The forms that multiply by one value per layer are compared with LayerNorm and warm-up restored.

| Layers | Baseline | One value, from 0 (ReZero) | One value, from a small value | LayerScale |
|---|---|---|---|---|
| 12 | 79.9 | 78.3 | 80.4 | 80.5 |
| 18 | 80.1 | 80.1 | 81.6 | 81.7 |
| 24 | Failed | 80.8 | 81.1 | 82.4 |
| 36 | Failed | 81.6 | 81.6 | 82.9 |

The authors say that starting from a small value is better than starting from 0. At 12 layers, starting from 0 is worse than the baseline.

Read: the relevant parts of Sections 2 and 3, Table 1, and the description of Table 4.
