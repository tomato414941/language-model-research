# autoresearch

Karpathy の `karpathy/autoresearch`（2026-03）。AI エージェントが1枚の GPU で nanochat を簡略にした学習コードを直しては学習し、結果が良ければ残す。

- 固定（`prepare.py`、変更禁止）：学習5分（起動と compile を除く）、ClimbMix（`karpathy/climbmix-400b-shuffle`、最後のファイルが検証用）、語彙8,192の BPE、文脈2,048、検証の BPB を測る関数。
- 自由（`train.py`）：モデルの形と大きさ、最適化、バッチなど。パッケージの追加は不可で、メモリは緩い制約。
- 条件は「シンプルなほうがよい」。同じ性能でコードが減れば改善として扱う。
- 標準の設定は8層、幅512、4ヘッド（1ヘッド128次元）、更新1回524,288トークン、Muon と AdamW、FlashAttention-3。
- 最適化の設定（`train.py`）：行列は Muon（Polar Express、NorMuon、cautious weight decay）、それ以外は AdamW。学習率は行列0.04、埋め込み0.6、出力0.004、スカラー0.5で、AdamW の値には (幅/768)^−0.5 を掛ける。β (0.8, 0.95)、ε 1e-10。momentum は最初の300回で0.85から0.95へ上げる。weight decay は0.2から線形に0まで下げる。学習率は warmup なしで前半を一定にし、後半で線形に0まで下げる。勾配のクリップなし。埋め込みと出力は共有せず、初期値の大きさも別々。
- モデル（`train.py`）：RoPE、Q と K の RMSNorm、学習する重みのない RMSNorm、ReLU²、注意と MLP の出力のゼロ初期化、入力と出力の埋め込みの分離（入力の標準偏差1.0のあと RMSNorm、出力は0.001）、x0 と resid の重み、logit の softcap 15、埋め込みは bf16、窓は4層中3層を半分に。
- 測り方（`prepare.py`）：BPB は最後のファイルの約2,100万トークン、文脈2,048で、行は BOS から始まり文書を詰める。学習も同じ読み方で、`prepare.py` の既定は10ファイル。
- `program.md` の実行例：val_bpb 0.9979、学習300秒、全体326秒、メモリ45GB、mfu 39.80%、4億9,960万トークン、953回の更新、5,030万パラメータ。README は H100 で確かめたとしている。

読んだのは README、`program.md`、`prepare.py` と `train.py` の定数とモデルの定義。実験の成果（約700回で約20個の改善、という話）は二次情報でしか見ていない。
