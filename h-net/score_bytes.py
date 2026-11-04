"""Scores a model of the H-Net comparison: each language's bits per byte on the same validation texts, and the
harness's accuracy and correct answers' bits per byte on harness.py's English and Japanese benchmarks.

A validation text is the start of a validation document, as much of it as both models read whole: at most a row of
bytes for H-Net and 1,024 tokens for the BPE model, cut at a character. Each model reads it after BOS, so that the two
read the same texts and can be compared text by text.

Usage: python h-net/score_bytes.py OUTPUT (--bpe DIRECTORY | --hnet DIRECTORY) [--limit N]
"""

import argparse
import json
import math
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path[:0] = [".", "h-net"]
from harness import BENCHMARKS, score_in_full  # noqa: E402
from hnet_model import byte_language_model, load  # noqa: E402
from loading import loglikelihoods, nanogpt_model  # noqa: E402

BPE_TOKENIZER = "data/hnet-bpe-stable/tokenizer.json"
ROWS = "data/hnet-bytes-stable/rows.json"
VALIDATION = "data/hnet-validation.jsonl"
BPE_CONTEXT = 1024
VALIDATION_BYTES = 2_000_000  # of each language


def validation_texts(tokenizer, byte_row, budget=VALIDATION_BYTES):
    """Each language's texts: the start of each validation document, as much as both models read whole."""
    texts, sizes = {}, {}
    for line in open(VALIDATION, encoding="utf-8"):
        row = json.loads(line)
        language = row["language"]
        if sizes.get(language, 0) >= budget:
            continue
        # One place for BOS in each model's row.
        text = row["text"].encode("utf-8")[: byte_row - 1].decode("utf-8", errors="ignore")
        while text and len(tokenizer.encode(text, add_special_tokens=False).ids) > BPE_CONTEXT - 1:
            text = text[: int(len(text) * 0.9)]
        if text:
            texts.setdefault(language, []).append(text)
            sizes[language] = sizes.get(language, 0) + len(text.encode("utf-8"))
    return texts


def validation_bits(model, texts):
    results = loglikelihoods(model, [([model.prefix], model.encode(text)) for text in texts])
    bits = [-loglikelihood / math.log(2) for loglikelihood, _ in results]
    size = sum(len(text.encode("utf-8")) for text in texts)
    return {"bits_per_byte": sum(bits) / size, "texts": len(texts), "bytes": size, "bits": [round(b, 4) for b in bits]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("--bpe", help="train.py's output directory of the BPE model")
    parser.add_argument("--hnet", help="train.py's output directory of an H-Net model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--limit", type=int, help="ask only this many questions of each task, to try it")
    arguments = parser.parse_args()
    if arguments.bpe:
        model = nanogpt_model(arguments.bpe, BPE_TOKENIZER, arguments.device, "bfloat16", arguments.batch_tokens)
    else:
        network = load(arguments.hnet, arguments.device)
        model = byte_language_model(network, arguments.device, arguments.batch_tokens)
    texts = validation_texts(Tokenizer.from_file(BPE_TOKENIZER), json.load(open(ROWS))["block_size"])
    result = {"validation": {language: validation_bits(model, chosen) for language, chosen in texts.items()}}
    result["harness"] = {
        language: score_in_full(model, {name: row for name, row in BENCHMARKS.items() if row[0] == language},
                                arguments.limit)
        for language in ("en", "ja")
    }
    Path(arguments.output).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {language: round(row["bits_per_byte"], 4) for language, row in result["validation"].items()}
    for language, scored in result["harness"].items():
        for name, row in scored["benchmarks"].items():
            summary[name] = {"accuracy": round(row["accuracy"], 4), "gold_bpb": round(row["gold_bits_per_byte"], 4)}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
