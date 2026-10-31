# autoresearch

Karpathy の `karpathy/autoresearch`（2026-03）。AI エージェントが1枚の GPU で nanochat を簡略にした学習コードを直しては学習し、結果が良ければ残す。

- 固定（`prepare.py`、変更禁止）：学習5分（起動と compile を除く）、ClimbMix（`karpathy/climbmix-400b-shuffle`、最後のファイルが検証用）、語彙8,192の BPE、文脈2,048、検証の BPB を測る関数。
- 自由（`train.py`）：モデルの形と大きさ、最適化、バッチなど。パッケージの追加は不可で、メモリは緩い制約。
- 条件は「シンプルなほうがよい」。同じ性能でコードが減れば改善として扱う。
- 標準の設定は8層、幅512、4ヘッド（1ヘッド128次元）、更新1回524,288トークン、Muon と AdamW、FlashAttention-3。
- `program.md` の実行例：val_bpb 0.9979、学習300秒、全体326秒、メモリ45GB、mfu 39.80%、4億9,960万トークン、953回の更新、5,030万パラメータ。README は H100 で確かめたとしている。

読んだのは README、`program.md`、`prepare.py` と `train.py` の定数とモデルの定義。実験の成果（約700回で約20個の改善、という話）は二次情報でしか見ていない。
