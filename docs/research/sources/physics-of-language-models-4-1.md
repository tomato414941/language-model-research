# Physics of Language Models 4.1

Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers（arXiv 2512.17351）。本題は短い畳み込みの層を足す話。

付録のモデルの説明に、初期値についての短い記述がある。

- GPT-2 由来の `rescale_prenorm_residual`（出力の重みの初期値を 1/√N 倍する）を切ると、性能がわずかに良くなった。数字はない。HF の Llama と Mamba2 の既定は切ってあり、GLA の著者もこれを受けて切った。
- GLA の実装の `initializer_range`（重み全体の初期値）を0.006にすると悪くなり、0.02に戻した。

読んだのは付録のこの部分だけ。
