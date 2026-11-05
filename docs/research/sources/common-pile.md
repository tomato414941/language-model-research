# Common Pile v0.1

Kandpal et al., The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text (arXiv 2506.05209, 2025). Public-domain and openly licensed text gathered from 30 sources. The 7B Comma v0.1-1T and 2T are trained on it.

## How the training data is mixed

- The main stage of the training data (`common-pile/comma_v0.1_training_dataset`) weights the original 463.6 billion tokens into an effective 1.0344 trillion tokens (README).
- By effective size, largest first: peS2o (papers) 260 billion, StackExchange 143.2 billion, Stack v2's educational code 135.5 billion, CC-licensed web (CCCC) 91.4 billion, USPTO (patents) 39.4 billion, PubMed 36.6 billion.
- How the weights were set (§4.2): a small model was trained on 28 billion tokens of each source, and the weights were raised or lowered by hand from its scores. Repetition is capped at six times over 1 trillion tokens. MixMin did not beat the hand-set mix. The tasks scored read as the same as §4.3's, but this is not stated.
- Filters (§4.1): language identification, DCLM's quality classifier on CCCC (with a low threshold), OCR errors, toxicity, PII and per-source boilerplate. Documents sharing 90% or more of their 20-grams are removed as duplicates.

## Results

- Small-scale comparison (§4.3): better than OLC, Common Corpus and KL3M on every task. FineWeb is the best on most tasks, and Comma is the best on MMLU and ARC. On HellaSwag, PIQA and CommonsenseQA it is much worse than Pile, OSCAR and FineWeb.
- Comma v0.1-1T beats models of the same budget on more than half of the tasks. It is strong on ARC-C and MMLU and lags on HellaSwag and PIQA (§4.4).
- No step that removes evaluation questions is described. WinoGrande was left out of the evaluation because it is in the Data Provenance Initiative's data.

§4.1–4.4 of the HTML version and the relevant part of the training data's README (from quotes extracted with a tool) were read. The tables of numbers and the appendix were not read.
