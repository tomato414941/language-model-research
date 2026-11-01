# Benchmarking Optimizers

Semenov ら（EPFL ほか）、Benchmarking Optimizers for Large Language Model Pretraining（arXiv 2509.01440、2025-09）。

- Llama 形（weight tying、SwiGLU、RMSNorm、RoPE）、FineWeb で、手法ごとに調整して比べる。
- 1.24億パラメータ、更新1回16,000または131,000トークンで、10億〜168億トークン学習する。
- 16,000トークンのとき、10億トークンでは SOAP、ADOPT、AdEMAMix、D-Muon、Prodigy、SF-AdamW が AdamW に勝ち、D-Muon が一番良い。トークンを増やすと AdEMAMix が先頭になり、AdamW が差を詰める。
- 7.2億パラメータ、更新1回100万トークンでは、AdEMAMix と MARS が一番良い。
- 全体では、AdEMAMix が安定して良い。Signum、Lion、MARS はバッチが大きいと伸びる。Sophia は小さいバッチで長く学習すると発散する。SOAP はどちらのバッチでも安定している。

読んだのは要旨、図の説明、結果の一部。2.1億の結果と付録は読んでいない。
