# modded-nanogpt's first Muon record

`records/track_1_short/2024-10-10_Muon` in `KellerJordan/modded-nanogpt`.

- Model: 12 layers, 12 heads, width 768. The embedding and output layers are shared. RMSNorm (without learned weights), RoPE, and a correction for the number of layers on the attention output.
- Training: 524,288 tokens per update (64 rows × 8, context 1,024), 6,200 updates (about 3.25 billion tokens). No gradient clipping.
- Embedding and output: AdamW, learning rate 0.0036, β (0.9, 0.95), weight decay 0.
- The blocks' matrices: Muon, learning rate 0.00036, momentum 0.95, Nesterov, 5 Newton-Schulz iterations, no weight decay. The update is rescaled to a mean square of 1 before it is applied. Q, K and V are orthogonalized separately.
- Learning rate: constant with no warmup, then falling linearly over the last 1,800 updates.
- Result: validation loss 3.2785, 1,339 seconds of training.

Read: the model and optimization parts of `train_gpt2.py`, and the end of the log.
