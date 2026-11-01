# Improving our LLM Pretraining Efficiency

Larry Dial の記事（Open Athena、2026-06-03、https://www.openathena.ai/blog/pretraining-speedup/）。Marin の密なモデルから MoE への移行と、その先の改善。

倍率は計算量での値で、かっこ内は実時間での値。

| 変更 | 倍率 |
|---|---|
| 密から Marin MoE V1 へ（1e23） | 6.7（3.6） |
| 専門家を64から256へ | 1.4（1.3） |
| AdamH から MuonH へ | 1.3（1.25） |
| Partial Key Offset | 1.2（1.2） |
| 専門家の正規化と倍率 | 1.04（1.04） |

- 密から MoE への差は、1e18〜3e20 では計算量で約4倍。最後の4つは1つずつ測り、重ねると 3e19 で2.1倍だった。
- MoE の法則は loss = 1.6 + 95.18·C^−0.0941、密の法則は loss = 1.65 + 117.8·C^−0.0966。
- Adam の学習率は lr = 1.680·tokens^−0.282·dim^−0.372·bs^0.5 に当てはまった。
- Partial Key Offset の効果の半分は、ヘッドの半分だけに RoPE をかける形から来るとしている。
- Attention Residuals、per-layer embeddings、MUDD、mHC、Engram も15%以上速くできそうだが、複雑さを抑えるために見送っている。

読んだのは要約、MoE の節、MuonH と Partial Key Offset の節の冒頭。図の値は読んでいない。
