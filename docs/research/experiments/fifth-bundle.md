# The fifth bundle (research-v5)

Chosen on 2026-11-05, after research-v4 became the current best. The three OLMES suites weigh QA, math and code equally, so this bundle works on what the model reads and how far it reads; v4's model and optimizer stay as they are.

## What v4 read as code

The code that v4 mixed in, the first two files of Common Pile's `stackv2_edu_filtered`, was all Markdown. The set's 95 files are ordered by language: 0000–0019 Markdown, then C++, C#, Go, Java, JavaScript, PHP, Python, Ruby, Swift and TypeScript, each in its own run of files (the first document of every fifth file, read on 2026-11-05). `prepare_extra.py` took the first files by name.

In the tokenizer trained on FineWeb, code takes 1.7–2.0 bytes a token (Markdown 2.5, FineWeb about 3.8), measured on the first 40 documents of each file.

## The changes

1. **Code from every file.** The first 2,600 documents of each of the 95 files (497M tokens), so the languages come in as the set holds them, without choosing by content. The code suite's tasks are in 17 languages (MultiPL-E style), and multilingual code pretraining helps across languages (Scaling Laws for Code, arXiv 2512.13472).
2. **A quarter each of math and code.** 50% FineWeb, 25% FineMath 4plus, 25% code, from 80/10/10. Llama 3 reads about 25% math and reasoning and 17% code (arXiv 2407.21783); To Code or Not To Code finds natural-language reasoning best at about 25% code (arXiv 2408.10914); up to half code costs natural language little (arXiv 2305.16264).
3. **More math and code at the end.** The last 20% of training reads 30% FineWeb and 35% each of math and code: domain upsampling over the final 10–20% (arXiv 2406.03476, 7B), annealing on math and code in Llama 3, code in the cooldown in arXiv 2408.10914, and OLMo 3's midtraining (arXiv 2512.13961).
4. **Rows of 2,048 tokens.** The suites' few-shot prompts often run past 1,024 tokens (math answers up to 2,960), which the model read in windows that see only 512 to 1,024 tokens back. Updates keep 262,144 tokens (32 rows × 4 passes).
5. **Attention windows that grow (SkyLadder, arXiv 2503.15450, NeurIPS 2025).** At a fixed number of tokens, shorter windows train better; SkyLadder starts with windows of 32 positions and widens them in a straight line to the whole row, here at 60% of training (the paper's α = 1/8 reaches its 8K rows at about 64%). Its 120M model gained 1.1 points on average accuracy, 360M 2.4. Sequence-length warmup reports similar gains at 125M (arXiv 2108.06084). The windows cut each document at multiples of the window, as the paper's block-wise mask does, and the key offset does not reach across a cut.

## Considered and left out

- **Retraining the tokenizer on the mix.** Code's 1.9 bytes a token suggests a large gain for the code suite, but it means retokenizing FineWeb and rebuilding the COUNT table (about 1.6 hours and 11 GB on the processing machine). Next, once this bundle shows what real code does.
- **Normalized value residual (IMU-1, arXiv 2602.02522).** −0.38% at 70M, against a model without value embeddings; the speedrun replaced value residual with value embeddings, which v4 has.
- **LayerNorm scaling.** −0.36% at 70M in IMU-1; split elsewhere, and the model has 10 layers.
- **Averaging the last weights.** Worse by 1.2% in the third bundle.

## Expected effects and risks

- The validation BPB is measured in rows of the model's context, so v5's are rows of 2,048 tokens against v4's 1,024. The decision uses the three suites.
- Half the rows are FineWeb against 80% before, so QA may lose some of what math and code gain.
- FineMath 4plus (398M tokens) is read about 0.6 times; repeats below four epochs cost little (arXiv 2305.16264).
- The evaluation reads 76M tokens at a context of 2,048 against 58M at 1,024.
