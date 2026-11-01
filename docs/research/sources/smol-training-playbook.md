# The Smol Training Playbook

Hugging Face の SmolLM3 の学習の記録（huggingfacetb-smol-training-playbook.hf.space）。

埋め込みの節：

- 基準は共有。分けると2 × 語彙 × 幅のパラメータになる。分けた表の割合は Llama 3.2 8B で13%、Llama 3.1 70B で3%。
- 比較は3つ。共有の12億（16層）、分けた14.6億（16層）、分けた12億（12層）。
- 共有の12億は、パラメータが約18%少ないのに、分けた14.6億と WinoGrande 以外のベンチマークで同じくらい。分けた12億は両方より悪い。
- 著者は、同じパラメータ数なら分けるより深くするほうが効くとし、SmolLM3-3B は共有にした。
- MobileLLM の1.25億の結果として、共有でパラメータが11.8%減り、精度の低下はわずかと引いている。

数字は本文の説明だけで、表の値は読んでいない。読んだのは取得の道具で抜き出した埋め込みの部分で、残りは読んでいない。`language-model` の[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/smol-training-playbook.md)は別の読書ノートから書いたもの。
