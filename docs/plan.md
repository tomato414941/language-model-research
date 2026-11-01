# 計画

## 目的

変更を入れると、同じお金でより良い言語モデルができるかを確かめる。目的は安くて良いモデルを作ることで、競技ではない。ルールは少なくする。多いと抜け道を探すことになるため。ハックより、大規模でも通じる改善を重視する。

## 方針

- 候補を種類で外さない。モデルや学習法のほか、速度、料金、データ、課題設定も候補にする。一覧は `research/topics/candidates.md`。
- 改善1つごとに測らない。読んでふるい、束にして1回測る。進め方は `research/topics/experiment-planning.md`。
- 候補も物差しも増える前提で、全体の計画を固定せず、次の束までを決める。物差しを変えても比べられるように、基準の重みを残して採点し直す。
- 判断は仮のものとして書き、時点と、見方が変わりそうな条件を添える。
- 読む、決める、変えるのと同じコミットで記録を更新する。この記録だけで続きができる状態を保つ。
- GPU を借りるなど費用のかかる操作は、内容と金額を確認してから行う。GPU の条件は `gpu.md`。

## 基準

nanoGPT の12層・幅768、AdamW、ClimbMix、5億トークン、H100 1枚で約830秒。検証の BPB と、英語6つのベンチマークの正解の BPB で測る。結果は `research/experiments/baseline.md`。

## 今の状態（2026-11-01）

- コード済み、未測定：Muon と学習率、Polar Express、NorMuon、cautious weight decay、Value embeddings、ヘッド128次元、10層・幅640（`configs/bundle.py`）。
- 確認の結果は `research/experiments/bundle-check.md`。

## 次にやること

1. FlashAttention-3 のコードと、束を GPU で動かす手順を用意する。
2. GPU を1回借りて、時間の使い方、1回に流す本数、注意の計算方式、文脈の長さごとの速度を測り、同じお金で回せる更新回数を決める。
3. 束を学習して採点し、基準と比べる。悪ければ項目を外して原因を探す。
4. 結果をもとに、候補の順番と次の束を決める。
5. 課題設定の変更を検討する（推論の費用、第2の規模、BPB で測れない候補）。

## 動かし方

- テスト：`python -m unittest discover -s tests`
- データの用意：`python prepare.py`
- 基準：`./baseline.sh 種...`（学習、採点、重みの保存までを行う）
- Muon だけ：`python train.py configs/muon.py`
- 束：`python train.py configs/bundle.py`。外し方は `configs/bundle.py` の説明にある。
- 採点：`python scoring.py 出力先 --tokenizer data/climbmix/tokenizer.json --validation data/climbmix/val.bin --questions data/questions.jsonl --output 結果.json`
- 比較：`compare.py`
