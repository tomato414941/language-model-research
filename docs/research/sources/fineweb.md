# FineWeb

Penedo et al., The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (arXiv 2406.17557, 2024). The data is `HuggingFaceFW/fineweb`.

## How it is made

- Text is extracted with trafilatura from the WARC files of 96 Common Crawl snapshots (§3.2).
- fastText English detection (0.65 or above), a URL blocklist, and MassiveText's quality and repetition filters (§3.3). Then C4's filters (except the rule on terminal punctuation) and their own heuristics (§3.5, §3.6).
- MinHash deduplication is done separately for each snapshot (§3.4). Email addresses and IP addresses are anonymized (§3.7).
- The experiments that chose the processing compared scores on 8 tasks: CommonSense QA, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande, ARC and MMLU (§3.1).
- No step that removes evaluation questions is described.

## Release

- About 15 trillion tokens (counted with the GPT-2 tokenizer). The license is ODC-By 1.0, and Common Crawl's terms of use also apply.
- There are uniformly drawn samples (10BT, 100BT, 350BT). 10BT is 15 files, about 30.6GB (revision `9bb295d`).
- FineWeb-Edu is a 1.3-trillion-token subset of FineWeb selected with a classifier of educational value (trained on Llama-3-70B annotations) (§4).

Read: the relevant sentences of §1, §3.1–3.7 and §4 of the HTML version (from quotes extracted with a tool), and the file listing on Hugging Face.
