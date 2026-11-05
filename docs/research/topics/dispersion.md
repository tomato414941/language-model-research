# Dispersion Loss

A method that adds one loss term so that the directions of the token vectors within the same text spread apart ([source](../sources/dispersion-loss.md)). It was put into the third bundle on 2026-11-03. On 2026-11-04 the bundle was about 1% better than the second bundle on validation BPB read from the start of documents ([the third bundle](../experiments/third-bundle.md)), but this item's own effect was not separated.

## Why it was included, and its weak points

- A report says that the smaller the model, the more its token directions gather in one place, and the current model (54.4 million) is small. In the report, the mean multiple-choice score is +0.7 to 1.5.
- There is only one study, and it does not report BPB. Whether the directions gather in the current model was not measured. A free check on a saved model was proposed, but the item was included without the check.
- The paper's training from scratch has an amount read about 200× ours. How it relates to the current bundle's normalization, such as QK-norm, is unknown.

## Its form in the third bundle

The settings are in `configs/bundle3.py`.

- At every eighth position (128 positions per row) of each of the 10 layers' outputs, the loss is computed from the angles of all pairs within the row and averaged over layers and rows. The weight 0.1 and τ=1 are the paper's default values.
- The paper takes the log of the sum over pairs, but here the log of the number of pairs is subtracted, making it the log of the mean. The gradient is the same, and the loss value falls between −1 and 0.
- It is added only in training and is not part of the validation loss. It is part of the training loss that is displayed.

## Conditions likely to change the view

- When the bundle is measured and its scores drop.
- When the directions turn out not to gather in the current model.
