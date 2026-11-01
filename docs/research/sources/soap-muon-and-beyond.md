# SOAP, Muon, and Beyond

Khona ら（NVIDIA）、SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales（arXiv 2607.20548、2026-07）。

- 全体300億（動くのは30億）の MoE を3兆トークン、全体720億（動くのは80億）の Mamba との混成を1兆トークンなどで学習する。
- 更新の大きさをそろえて学習率を移し、AdamW、Muon、SOAP を比べる。AdamW はあるバッチを超えると悪くなり、Muon と SOAP は更新1回1億トークンまで保つとしている。
- SOAP は大きなバッチで loss が跳ねる。毎ステップの QR 直交化と KL-Shampoo の共分散の推定で解消した。
- Qwen3 の全体300億の形で比べると、KL-SOAP が Muon にわずかに勝つ。メモリに余裕があれば KL-SOAP を勧めている。この比較では Q、K、V を分けていない。
- Nesterov 型の momentum は収束を良くしなかった。ε は調整していないと断っている。

読んだのは要旨、導入、比較の方法の冒頭、5.5節、実装の冒頭、結論。5.3節の結果と表は読んでいない。
