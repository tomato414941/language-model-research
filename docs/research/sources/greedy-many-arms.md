# The Unreasonable Effectiveness of Greedy Algorithms in Multi-Armed Bandit with Many Arms

Bayati ら（arXiv 2002.10121、NeurIPS 2020）。

- 候補の数 k が予算 T の平方根以上の場合を扱う。全部を1回ずつ試すだけで損が k に比例するので、普通の UCB は試しすぎる。√T 個程度に絞ると、増え方の上で最適になる。
- 実験では、絞った UCB より、一番良く見えるものを選び続ける貪欲な方法のほうが良かった。著者は、候補が多いと良い候補が早く見つかること（自由な探索）で説明している。
- 候補の報酬は独立と仮定している。

読んだのは要旨、導入、主な図の説明。証明は読んでいない。
