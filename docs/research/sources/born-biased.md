# Transformers Are Born Biased

Transformers Are Born Biased: Structural Inductive Biases at Random Initialization and Their Practical Consequences（arXiv 2602.05927）。

- ランダムな初期値のモデルは、特定のトークンを強く好む。12億の GPT-2 の形では、一番好まれるトークンが文脈1,024で10.2%。原因は MLP の非対称な活性化と注意の集約で、MLP だけでも起きる。
- 種ごとの偏りは、弱くなるが学習のあとも残る。論文はこれをモデルの指紋に使う。
- 初期値は GPT-2 の形で、出力の重みは 0.02/√(2×層の数)。出力の重みを0にした比較はない。0なら学習の始めは各層が値を足さないのでこの偏りも出ないはずだが、これは自分たちの読みで、品質への影響は分からない。

読んだのは1、3、4.1、5節の始め、6.2.2、7節、付録の初期値の部分。
