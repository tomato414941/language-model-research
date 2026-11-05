# Survey of breakthroughs

From 2026-10-31 to 11-01, we read general breakthroughs in machine learning and language models, picked by awards, reviews and adoption. They served as material for ordering the candidates. The depth of reading varied, mostly abstracts and main results. The numbers below were copied from the notes taken while reading.

| Year | Work | Findings |
|---|---|---|
| 2017 | Attention (arXiv 1706.03762) | English-German BLEU of 27.3 with the base model; the best single model at the time had 26.03 with about 6× the compute |
| 2020 | Scaling Laws (2001.08361) | Loss follows power laws in compute, parameters and data; the shape has little effect |
| 2022 | Chinchilla (2203.15556) | Scale parameters and tokens in the same proportion. About 20 tokens/parameter. 70B beats Gopher 280B at the same compute. The evidence is one set at large scale, whose smallest budget is 6e18 (the current baseline is 2.7e17) |
| 2022 | InstructGPT (2203.02155) | 1.3B is preferred over the 175B GPT-3 |
| 2022 | Chain-of-Thought (2201.11903) | PaLM's GSM8K goes from 18% to 57% |
| 2023 | LLaMA (2302.13971) | 13B beats GPT-3, which is 10× larger. Uses RMSNorm, SwiGLU and RoPE, but gives no numbers for their individual effects |
| 2023 | QLoRA (2305.14314) | Fine-tuning 65B on one 48GB GPU |
| 2023 | phi-1 (2306.11644) | 1.3B, 4 days on 8 A100s, HumanEval 50.6%. Emphasizes data quality |
| 2023 | Mamba (2312.00752) | 3B beats Transformers of the same size and matches ones twice its size |
| 2024 | FineWeb-Edu (2406.17557) | About a tenth of the tokens needed for the same level |
| 2024 | DeepSeek-V2, V3 (2405.04434, 2412.19437) | Cut costs with MoE. V3 is 671B (37B active), 14.8 trillion tokens, 2.664 million H800 hours |
| 2025 | DeepSeek-R1 (2501.12948) | Reinforcement learning takes AIME from 15.6% to 77.9%. The cost is about 5.5% of V3's pretraining |
| 2025 | Gated Attention (NeurIPS 2025 Best) | Works at large scale but is weak at small scale |
| 2025 | Superposition (NeurIPS 2025 runner-up) | An explanation of scaling laws |
| 2025 | On-policy distillation | Qwen3-8B's AIME24 reaches 74.4% in 1.8k GPU hours (reinforcement learning: 67.6% in 17.9k) |
| 2026 | mHC (2512.24880) | At 27B, loss −0.021 and MMLU from 59.0 to 63.4. Adopted by DeepSeek-V4 |
| 2026 | Attention Residuals (2603.15031) | Worth 1.25×. At 194 million, from 1.931 to 1.899 (1.06–1.12× in Marin's small-scale results) |
| 2026 | Olmo Hybrid (2604.03444) | olmo-hybrid.md |
| 2026 | The Flexibility Trap (ICML 2026 Outstanding) | Diffusion language models struggle to combine flexibility and quality. LLaDA 2.0 100B is at about Qwen3 30B's level |
| 2026 | Harness (GLM-5.1's report) | From 52.5% to 65.5% without changing the weights. The gap between harnesses is 7.8× the gap between models |

The optimizer survey is in optimizer.md, retrieval in retrieval.md, Engram in engram.md, MoE in moe.md and RoPE in rope.md.

Findings:

- No single invention as groundbreaking as Attention can be found in the last five years. Raschka's year-by-year overview also attributes progress to accumulation rather than single breakthroughs (2022 RLHF and PPO, 2023 LoRA and SFT, 2024 Mid-Training, 2025 RLVR and GRPO).
- Many of the changes with large effects, such as post-training and data quality, cannot be measured with the current BPB or do not show without changing the scale.
