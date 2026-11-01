# Gated DeltaNet

Yang ら（arXiv 2412.06464、2024-12）。線形注意に、記憶を消すゲートと、狙って書き換える delta rule を入れる。

- 1.3B、FineWeb-Edu 1,000億トークン、AdamW で、Mamba2 や DeltaNet を複数の課題で上回るとしている。
- 学習の速さは DeltaNet とほぼ同じで、Mamba2 より少し遅い。文脈2,000トークンの範囲では、FlashAttention-2 を使う Transformer が一番速く、短い窓の注意と混ぜた形はその中間になる。

読んだのは要旨と学習の速さの節。実験の本体は読んでいない。
