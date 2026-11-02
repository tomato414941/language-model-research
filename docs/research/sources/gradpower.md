# GradPower

GradPower: Powering Gradients for Faster Language Model Pre-Training（arXiv 2505.24275、ICML 2026）。

オプティマイザに渡す前に勾配を要素ごとに `sign(g)·|g|^p` にする1行の変更。p=1.2 を2億のモデルで決めて固定した。6,600万〜20億の LLaMA と MoE で最後の損失が下がるとする。

- LLaMA 20億（C4）の AdamW：損失 −0.022、点の平均 47.72→48.26（ARC-E 60.02→60.35、HellaSwag 44.65→44.93、WinoGrande 56.83→59.43）。
- Qwen2MoE 20億：損失 −0.028。

読んだのは papernotes.org の要約だけで、原文は読んでいない。読んだ範囲に Muon との組み合わせの数字はない。
