# FlashAttention-3 の論文

Shah ら、FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision（arXiv 2407.08608）。

- FlashAttention-2 の H100 での使用率は35%にとどまる。
- FlashAttention-3 は H100 の新しい命令を使い、注意の計算を FP16 で1.5〜2倍にする。最大740 TFLOPs/s（使用率75%）。FP8 では約1.2 PFLOPs/s に近づく。
- 数字は注意の計算だけのもので、学習全体の速さは示していない。
- 著者は、低精度の注意が大規模な学習に与える影響の理解を今後の課題としている。

読んだのは要旨、導入、結論、実験の節の見出しと図の説明。アルゴリズムの詳細は読んでいない。
