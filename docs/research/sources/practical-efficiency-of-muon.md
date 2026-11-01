# Practical Efficiency of Muon

Essential AI、Practical Efficiency of Muon for Pretraining（arXiv 2505.02222、2025-05）。

- 1億〜40億パラメータで、Muon は同じ loss に AdamW より10〜15%少ないトークンで届く。バッチを大きくしても、この差は保つか広がる。
- 1次のモーメントだけを持つので、比べた中でメモリが一番少ない（AdamW より少ない）。計算の増分はバッチに反比例して減る。
- 埋め込みと正規化は Adam で更新する。
- Muon と µP の併用で、37億パラメータ、文脈8,192まで学習率と weight decay を移せた。

読んだのは要旨、導入の一部、実験の設定の一部、結論。結果の節は読んでいない。
