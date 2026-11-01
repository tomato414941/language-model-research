# modded-nanogpt の最初の Muon の記録

`KellerJordan/modded-nanogpt` の `records/track_1_short/2024-10-10_Muon`。

- モデル：12層、12ヘッド、幅768。埋め込みと出力の層は共有。RMSNorm（学習する重みなし）、RoPE、注意の出力に層の数の補正。
- 学習：更新1回524,288トークン（64本×8、文脈1,024）、6,200回（約32.5億トークン）。勾配のクリップなし。
- 埋め込みと出力：AdamW、学習率0.0036、β (0.9, 0.95)、weight decay 0。
- ブロックの行列：Muon、学習率0.00036、momentum 0.95、Nesterov、Newton-Schulz 5回、weight decay なし。更新の2乗平均が1になるように大きさをそろえてから掛ける。Q、K、V は別々に直交化する。
- 学習率：warmup なしで一定にし、最後の1,800回で線形に下げる。
- 結果：検証損失3.2785、学習1,339秒。

読んだのは `train_gpt2.py` のモデルと最適化の部分と、ログの最後。
