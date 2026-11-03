# 小さいモデルを伸ばす線の評価

Qwen3-1.7B を強化学習などで伸ばす線の評価。2026-11-03 に、測る前に決めた。language-model の `docs/evaluation.md` の決まりに従う。

## 決めたこと

- **形式：** 作者が手順を定めている課題は、作者のコードと設定で問う。Swallow の評価（swallow-evaluation-instruct）は思考するモデル向けに形式を作り直しているので、作者の手順がある課題には使わない。課題ごとの照合と使う道具は下の「各課題の形式」にある。点は Swallow の表と比べられない。
- **比べ方：** 元のモデルと伸ばしたモデルを、同じ環境と設定で測る。思考ありと思考なしの両方で測る。答えの長さ（トークン数）も比べる。
- **合格の線：** 思考ありで、日本語の課題の平均が +0.05 以上上がる。かつ、思考ありと思考なしのどちらでも、下のどの課題も2つのモデルの差の標準誤差の3倍を超えて下がらない。
- **日本語の課題（平均に入れる）：** JamC-QA、WMT20 英日、WMT20 日英、M-IFEval-Ja、MMLU-ProX（日本語）、BenchMAX Science（日本語）、PolyMath（日本語）、JHumanEval。
- **英語の課題：** HellaSwag、IFBench、MMLU-Pro、GPQA Diamond、MATH-500、AIME 2024–2025、LiveCodeBench。
- **他の言語：** MMLU-ProX の軽量版（日英以外の27言語）、MGSM（日英以外の9言語）、PolyMath（日英以外の16言語）。PolyMath の作者の採点は、問いと同じ言語で答えたかも数える。
- **安全性：** AnswerCarefully の v2.0 と borderline-v1.0、XSTest。
- **道具の呼び出し：** BFCL。
- **長い文脈：** RULER。日本語の文章で追加の事前学習をしたときだけ測る。
- **測る時期：** 日本語と英語の課題は毎回測る。ほかは、日本語の平均が +0.05 に届いたときに、元のモデルと並べて測る。
- **測れないもの：** MT-Bench。作者が採点に使った GPT-4（gpt-4-0613）は 2026-10-23 に提供を終えた。
- **確かめていないこと：** XSTest と HellaSwag の作者の手順。
- **決めていないこと：** AnswerCarefully の採点に、OpenAI の有料 API を使うか。

## 各課題の形式

2026-11-03 に、各課題の README とデータのカード、MMLU-Pro の評価コード、lm-evaluation-harness の MMLU-ProX の定義を読んだ。論文の本文は読んでいない。

Swallow の評価は、例を付けずに長く思考させる形に作り直している。たとえば MMLU-Pro と MMLU-ProX は例なし、温度 0.6、最長 32,768 トークンで、作者の定め（5例、温度0、出力は 2,048 トークンまで）と違う。IFBench は温度 0.6 で指示ごとの strict を、LiveCodeBench はプロンプトとコードの取り出し方を変えて測っている。

| 課題 | 作者の定め | 問い方 |
|---|---|---|
| MMLU-ProX（日本語と、他の言語の軽量版） | カードが lm-evaluation-harness の `mmlu_prox_{言語}` を指定している。日本語は5例、温度0、出力は 2,048 トークンまでで、「答えは (X) です」から取り出す | harness。軽量版は作者が公開した `mmlu_prox_lite_{言語}` の全問 |
| MMLU-Pro | 作者の `evaluate_from_local.py`。5例、温度0、出力は 2,048 トークンまでで、`answer is (X)` から取り出す | 作者のコード |
| IFBench | 作者の `run_eval`。温度0で、思考するモデルは出力の上限を広げ、思考を除いて採点する。論文は設問ごとの loose を報告する | 作者のコード |
| LiveCodeBench | 作者の `lcb_runner`。pass@1 と pass@5 | 作者のコード。問題は LiveCodeBench の v6 で追加された175問 |
| PolyMath | 作者のプロンプト（`instruction.py`）と採点コード。正解率と、問いと同じ言語で答えたかを出す | 作者のコード |
| JamC-QA | 選択肢の記号でなく文字列を出力させ、完全一致を見る。FlexEval の `jamcqa` で測れる | FlexEval |
| BenchMAX Science | lm-evaluation-harness と BenchMAX のコードで測る | 作者のコード |
| GPQA Diamond | 作者のコードに zero_shot、5_shot、zero_shot_chain_of_thought、chain_of_thought の型がある | 作者のコードの zero_shot_chain_of_thought |
| JHumanEval | HumanEval と同じく、関数の頭と説明文を入力にする | HumanEval と同じ補完の形 |
| M-IFEval-Ja | 公式の採点コード | 公式の採点コード（language-model の v6 と同じ） |
| MT-Bench（英語、日本語） | GPT-4 が採点する。日本語版は Stability AI の FastChat | 測れない。gpt-4-0613 は 2026-10-23 に提供を終えた |
| AnswerCarefully | 作者の llm-jp-judge。採点は OpenAI か Azure の API で、例は gpt-4o-2024-08-06 | llm-jp-judge |
| BFCL | 作者の `bfcl generate` と `bfcl evaluate` | 作者のコード |
| RULER | 作者の生成と採点のスクリプト | 作者のコード |

作者が手順を定めていない課題は、公開の道具で問う。

- MGSM：harness（language-model の v6 と同じ）
- MATH-500、AIME 2024–2025、WMT20 英日・日英：Swallow の評価

作者の手順を確かめられなかった課題：

- HellaSwag：作者のリポジトリが DMCA の通知で見られない（2026-09-14 から）。language-model の v6 の外との比較と同じく harness で問う。
- XSTest：作者の説明を取得できなかった。
