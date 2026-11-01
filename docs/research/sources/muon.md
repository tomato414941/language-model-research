# Muon

Keller Jordan ら、Muon: An optimizer for hidden layers in neural networks（ブログ記事、2024-12-08、https://kellerjordan.github.io/posts/muon/）。論文にはなっていない。

- 2次元の重みのために、SGD-momentum の更新へ Newton-Schulz を5回かけて直交に近づける（特異値分解の UV^T に当たる）。係数は (3.4445, −4.7750, 2.0315) で、bfloat16 で動く。
- スカラー、ベクトル、入力と出力の層は AdamW などで更新する。埋め込みは modular norm の理論から説明でき、出力の層は実験からそうしている。
- 実験では、Nesterov 型の momentum が少し良く、Q、K、V は別々の行列にかけるほうが良い。
- 計算の増分の上限は T·m/B（T は反復回数、m は幅、B は更新1回のトークン数）。nanoGPT の記録（幅768、524,288トークン）で0.7%、Llama 405B で0.5%。メモリは SGD-momentum と同じ。
- 結果：FineWeb で検証 loss 3.28 に届く時間の記録（NanoGPT speedrun）を1.35倍速くした。記事の時点で、その後の12回の記録は7人によるもので、すべて Muon を使っている。CIFAR-10 の94%到達は A100 で3.3秒から2.6秒。15億パラメータを HellaSwag で GPT-2 XL の水準にするのに 8×H100 で10時間（AdamW は13.3時間）。
- なぜ効くかについては、Transformer の更新は少数の方向に偏っていて（ほぼ低ランク）、直交化で弱い方向が大きくなるのでは、という推測にとどめている。
- 論文の多くは AdamW の基準の調整が足りないとして、競技の記録で示すことを求めている。記録なら、AdamW のほうが速ければ戻すだけで新記録になるため。
- 記事の時点の未解決の問い：200億パラメータ以上で効くか、分散できるか、事前学習以外でも効くか。

読んだのは本文の全部。図は見ていない。
